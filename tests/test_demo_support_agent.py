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
            # first, which read the refunds note written in Ben's turn.
            t = lin.trace(reply)
            ben_turn = next(x["id"] for x in w.turns if x.get("customer") == "ben")
            ana_turns = [x["id"] for x in w.turns if x.get("customer") == "ana"]
            labels = {lin.boxes[b]["kind"] + ":" + str(lin.boxes[b]["turn"]) for b in t["origins"]}
            assert labels == {f"message:{ben_turn}", "handbook:0"} | {
                f"message:{a}" for a in ana_turns}
            assert lin.trace(f"msg:{ben_turn}")["self_origin"]

            # 1. Candidates: beliefs about refunds; the person confirms the
            # ones that contradict the correction (the handbook doesn't).
            c = correction.candidates(w, CORRECTION)
            assert c["topic"] == "refunds"
            wrong = [x for x in c["candidates"] if x["conflicts"]]
            assert sorted(x["kind"] for x in wrong) == ["customer_note", "policy_note"]
            assert not any(x["conflicts"] for x in c["candidates"] if x["kind"] == "handbook")
            boxes = [x["box"] for x in wrong]

            # 2. The plan starts from where they came from, Ben's message, and
            # spares Dev.
            p = correction.plan(w, CORRECTION, boxes)
            assert [r["box"] for r in p["roots"]] == [f"msg:{ben_turn}"]
            kinds = {g: sorted(lin.boxes[i["box"]]["kind"] for i in items)
                     for g, items in p["groups"].items()}
            assert kinds == {"fix": ["customer_note", "customer_note", "policy_note"],
                             "cancel": ["ticket"], "notify": ["reply"] * 3,
                             "record": ["message"] * 3}
            assert p["customers"] == ["ana", "ben"]
            dev = next(e for e in w.log if e["customer"] == "dev")
            dev_box = lin.box_for("mcp:crm", dev["reply_id"])
            assert all(i["box"] != dev_box for g in p["groups"].values() for i in g)
            assert p["affected"] + p["untouched"] == p["wipe"]

            # 3. Apply: the ticket is cancelled in the CRM, Ben gets the
            # correction itself, Ana gets her questions answered again.
            r = await correction.apply(w, CORRECTION, boxes)
            assert r["failed"] == [] and len(r["cancelled"]) == 1
            assert w.ticket_status(r["cancelled"][0]) == "cancelled"
            after = {(x["customer"], x["old"]): x["after"] for x in r["pairs"]}
            assert len(after) == 3 and all(a.startswith("Correction to my earlier reply:")
                                           for a in after.values())
            assert after[("ana", first["reply_id"])].endswith("outside our 30-day refund window.")
            assert "don't see" in after[("ana", second["reply_id"])]
            assert CORRECTION in next(a for (cu, _), a in after.items() if cu == "ben")

            lin = Lineage(w)
            view = lin.view()
            status = {b["id"]: b["status"] for b in view["boxes"]}
            assert status[reply] == "corrected"
            assert status[f"msg:{ben_turn}"] == "on record"
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
            rep = _report(w, Lineage(w), f"msg:{ben_turn}")
            assert rep["chain"]["ok"]
            assert sum(i["status"] == "quarantined" for i in rep["items"]) >= p["affected"]
            assert "# Excision report" in rep["markdown"]
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


def test_model_match_is_checked_against_the_records():
    from demo.support_agent.llm import parse_match

    recs = [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]
    out = parse_match('Sure: {"topic": "refunds", "relevant": ["a"], "contradicts": ["b", "zzz"]}',
                      recs, ["refunds", "shipping"])
    assert out == {"topic": "refunds", "relevant": ["a", "b"], "contradicts": ["b"]}
    assert parse_match('{"topic": "nope"}', recs, ["refunds"])["topic"] is None


def test_candidates_come_from_the_model(tmp_path):
    """With a real model, the model picks what contradicts the correction;
    the plan and apply then work from what the person confirmed."""
    from demo.support_agent import correction
    from demo.support_agent.world import CORRECTION

    class Model:
        def __init__(self, inner):
            self.inner, self.name, self.seen = inner, "fake", None

        def decide(self, ctx):
            return self.inner.decide(ctx)

        def match(self, text, records, topics):
            self.seen = text
            ids = [r["id"] for r in records if r["kind"] == "policy_note"]
            return {"topic": "refunds", "relevant": ids, "contradicts": ids}

    async def main():
        w = World(str(tmp_path / "world"), "scripted")
        await w.start()
        try:
            w.llm = Model(w.llm)
            await w.run_story()
            c = correction.candidates(w, "refunds are a month, actually")
            assert w.llm.seen == "refunds are a month, actually"
            assert [x["kind"] for x in c["candidates"]] == ["policy_note"]
            assert all(x["conflicts"] for x in c["candidates"])
            p = correction.plan(w, CORRECTION, [x["box"] for x in c["candidates"]], c["topic"])
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
        assert default_model() == "anthropic/claude-sonnet-5-5"
        monkeypatch.setenv("DEMO_MODEL", "openai/qwen3:8b")
        assert default_model() == "openai/qwen3:8b"
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)  # set by load_env, not monkeypatch
