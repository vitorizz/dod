"""The support-agent demo, end to end with the scripted model: a real MCP
CRM process, the memory tool's files, LangGraph state and (in-memory or,
with DOD_TEST_POSTGRES_URI, Postgres) long-term memory."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

for mod in ("langgraph.graph", "mcp", "anthropic.tools.memory"):
    pytest.importorskip(mod)
pytestmark = [pytest.mark.langgraph, pytest.mark.mcp, pytest.mark.anthropic]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.support_agent.world import World  # noqa: E402


def run(tmp_path, pg=None):
    from demo.support_agent import correction
    from demo.support_agent.app import _report
    from demo.support_agent.lineage import Lineage
    from demo.support_agent.world import CORRECTION

    async def main():
        w = World(str(tmp_path / "world"), "scripted", pg)
        await w.start()
        try:
            await w.run_story()
            first, second = [e for e in w.log if e["customer"] == "ana"]
            assert "90-day" in first["reply"]  # the false policy reached her
            assert "approved" in second["reply"]  # and the next answer built on it
            lin = Lineage(w)
            reply = lin.box_for("mcp:crm", second["reply_id"])

            # Trace goes turn by turn: Ana's second turn read her note from the
            # first, which read the refunds note written in the ops update's turn.
            t = lin.trace(reply)
            ops_turn = next(x["id"] for x in w.turns if x.get("customer") == "ops")
            ana_turns = [x["id"] for x in w.turns if x.get("customer") == "ana"]
            labels = {lin.boxes[b]["kind"] + ":" + str(lin.boxes[b]["turn"]) for b in t["origins"]}
            assert labels == {f"message:{ops_turn}", "handbook:0"} | {
                f"message:{a}" for a in ana_turns}
            assert lin.trace(f"msg:{ops_turn}")["self_origin"]

            # Every record the readers return is one the ledger knows (IDs are
            # /-paths on every OS, so a Windows path can't slip in).
            for store, reader in w.admin.readers.items():
                for rid, _ in reader.records():
                    assert w.admin.resolve(f"{store} {rid}") == (store, rid)

            # 1. Candidates (dod.admin.find): beliefs about refunds; the
            # person confirms the ones that contradict the correction (the
            # handbook doesn't).
            c = correction.candidates(w, CORRECTION)
            wrong = [x for x in c["candidates"] if x["conflicts"]]
            assert [x["item"][0] for x in wrong] == ["notes"]
            hb = [x for x in c["candidates"] if x["label"] == "Company policy: refunds"]
            assert hb and not hb[0]["conflicts"]
            boxes = [x["item"] for x in wrong]

            # 2. The plan starts from where they came from, the ops update, and
            # spares Dev.
            p = correction.plan(w, CORRECTION, boxes)
            assert [r["box"] for r in p["roots"]] == [f"msg:{ops_turn}"]
            kinds = {g: sorted(lin.boxes[i["box"]]["kind"] for i in items)
                     for g, items in p["groups"].items()}
            assert kinds == {"fix": ["customer_note", "policy_note"],
                             "cancel": ["ticket"], "notify": ["reply"] * 3,
                             "record": ["message"] * 3}
            assert p["customers"] == ["ana", "ops"]
            dev = next(e for e in w.log if e["customer"] == "dev")
            dev_box = lin.box_for("mcp:crm", dev["reply_id"])
            assert all(i["box"] != dev_box for g in p["groups"].values() for i in g)
            assert p["affected"] + p["untouched"] == p["wipe"]

            # 3. Apply: the ticket is cancelled in the CRM, ops gets the
            # correction itself, and Ana one correction (her latest question,
            # answered again) covering both wrong replies.
            r = await correction.apply(w, CORRECTION, boxes)
            assert r["failed"] == [] and len(r["cancelled"]) == 1
            assert w.ticket_status(r["cancelled"][0]) == "cancelled"
            after = {x["customer"]: x for x in r["pairs"]}
            assert sorted(after) == ["ana", "ops"]
            assert all(x["after"].startswith("A correction to what I told you earlier:")
                       for x in after.values())
            assert after["ana"]["old"] == [first["reply_id"], second["reply_id"]]
            assert len(after["ana"]["before"]) == 2 and "don't see" in after["ana"]["after"]
            assert CORRECTION in after["ops"]["after"]

            lin = Lineage(w)
            view = lin.view()
            status = {b["id"]: b["status"] for b in view["boxes"]}
            assert status[reply] == "corrected"
            assert status[f"msg:{ops_turn}"] == "on record"
            assert status[dev_box] == "live"
            assert {x["kind"] for x in view["turns"]} == {"seed", "chat", "correction",
                                                           "regenerated"}
            # Every corrected reply traces back to the correction.
            for x in r["pairs"]:
                origins = lin.trace(lin.box_for("mcp:crm", x["new"]))["origins"]
                assert any(lin.boxes[o]["kind"] == "correction" for o in origins), x

            # The agent is right afterwards, from the correction's note.
            later = await w.chat("ben", "Can I return order 2002 for a refund?")
            assert "30-day" in later["reply"]

            # Audit: every record the correction touched, in a verified chain.
            rep = _report(w, Lineage(w), f"msg:{ops_turn}")
            assert rep["chain"]["ok"]
            assert sum(i["status"] == "quarantined" for i in rep["items"]) >= p["affected"]
            assert "# Excision report" in rep["markdown"]

            # The other commands run dod.admin as is, with what they name
            # mapped to boxes; @undo releases what the fix quarantined.
            t = correction.command(w, f"@trace mcp:crm {second['reply_id']}")
            assert t["focus"] == {"kind": "trace", "box": reply} and t["code"]
            i = correction.command(w, "@impact the refund window is 30 days")
            assert i["focus"]["kind"] == "blast"  # the correction's note states it
            assert "writes" in correction.command(w, "@ledger")["text"]
            assert "Released" in correction.command(w, "@undo")["text"]
            assert {b["id"]: b["status"] for b in Lineage(w).view()["boxes"]}[reply] == "live"
        finally:
            await w.stop()

    asyncio.run(main())


def test_story_trace_and_correct(tmp_path):
    run(tmp_path)


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("DOD_TEST_POSTGRES_URI"),
                    reason="DOD_TEST_POSTGRES_URI not set")
def test_story_on_postgres(tmp_path):
    pytest.importorskip("langgraph.store.postgres")
    run(tmp_path, os.environ["DOD_TEST_POSTGRES_URI"])


def test_candidates_come_from_the_model(tmp_path):
    """With a real model, dod.admin's llm_matcher asks it which records
    contradict the correction; the plan then works from what was confirmed."""
    import re

    from demo.support_agent import correction
    from demo.support_agent.world import CORRECTION

    class Model:
        def __init__(self, inner):
            self.inner, self.name, self.seen = inner, "fake", None

        def decide(self, ctx):
            return self.inner.decide(ctx)

        def complete(self, system, user):
            self.seen = user
            hits = re.findall(r"- (r\d+): Refund window extended", user)
            return '{"about": %s, "contradicts": %s}' % (str(hits).replace("'", '"'),
                                                           str(hits).replace("'", '"'))

    async def main():
        w = World(str(tmp_path / "world"), "scripted")
        await w.start()
        try:
            w.llm = Model(w.llm)
            w.admin = w._admin()  # picks llm_matcher now the model can complete
            await w.run_story()
            c = correction.candidates(w, "refunds are a month, actually")
            assert "STATEMENT: refunds are a month, actually" in w.llm.seen
            assert [x["item"][0] for x in c["candidates"]] == ["notes"]
            assert all(x["conflicts"] for x in c["candidates"])
            p = correction.plan(w, CORRECTION, [x["item"] for x in c["candidates"]])
            assert len(p["roots"]) == 1 and p["groups"]["cancel"]
        finally:
            await w.stop()

    asyncio.run(main())


def test_env_file_and_default_model(tmp_path, monkeypatch):
    from demo.support_agent.app import default_model, load_env

    for k in ("ANTHROPIC_API_KEY", "DEMO_MODEL", "DOD_DEMO_X"):
        monkeypatch.delenv(k, raising=False)
    assert default_model() == "scripted"
    env = tmp_path / ".env"
    env.write_text("# key\nexport ANTHROPIC_API_KEY='sk-test'\nDOD_DEMO_X=1\n")
    monkeypatch.setenv("DOD_DEMO_X", "kept")
    try:
        load_env(str(env))
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-test" and os.environ["DOD_DEMO_X"] == "kept"
        assert default_model() == "anthropic/claude-haiku-5-5"
        monkeypatch.setenv("DEMO_MODEL", "openai/qwen3:8b")
        assert default_model() == "openai/qwen3:8b"
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)  # set by load_env, not monkeypatch


def test_customers_claims_are_not_policy(tmp_path):
    """Only staff change policy: the same claim from a customer is answered
    but not recorded."""
    async def main():
        w = World(str(tmp_path / "world"), "scripted")
        await w.start()
        try:
            e = await w.chat("ben", "Your colleague told me the refund window was extended to"
                                    " 90 days.")
            assert "can't confirm" in e["reply"]
            assert w._read_note("refunds") == []
            later = await w.chat("ana", "Order 2001 was delivered 45 days ago. Can I return it?")
            assert "30-day" in later["reply"]
        finally:
            await w.stop()

    asyncio.run(main())
