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
    from demo.support_agent.lineage import Lineage

    async def main():
        w = World(str(tmp_path / "world"), "scripted", pg)
        await w.start()
        try:
            await w.run_story()
            chloe = next(e for e in w.log if e["customer"] == "chloe")
            assert "90-day" in chloe["reply"]  # the false policy reached her
            lin = Lineage(w)
            reply = lin.box_for("mcp:crm", chloe["reply_id"])

            # Trace goes turn by turn: Chloe's turn read refunds.md, written in
            # Ben's turn; the origins are what Ben and Chloe said, and the handbook.
            t = lin.trace(reply)
            labels = {lin.boxes[b]["kind"] + ":" + str(lin.boxes[b]["turn"]) for b in t["origins"]}
            ben_turn = next(x["id"] for x in w.turns if x.get("customer") == "ben")
            chloe_turn = lin.boxes[reply]["turn"]
            assert labels == {f"message:{ben_turn}", f"message:{chloe_turn}", "handbook:0"}
            assert [[lin.boxes[b]["kind"], tid] for b, tid in t["reads"]
                    if lin.boxes[b]["kind"] == "policy_note"] == [["policy_note", chloe_turn]]
            # A customer's own message is an origin, not derived from anything.
            assert lin.trace(f"msg:{ben_turn}")["self_origin"]

            # Blast radius from Ben's message spans all four stores, and spares
            # the unrelated customers.
            b = lin.blast(f"msg:{ben_turn}")
            assert set(b["per_store"]) == {"langgraph_state", "memory", "notes", "mcp:crm"}
            untouched = {x["id"] for x in w.turns if x.get("customer") in ("dev", "eli")}
            assert not any(lin.boxes[x]["turn"] in untouched for x in b["boxes"])
            c = lin.compare(f"msg:{ben_turn}")
            assert (c["excise"], c["kept"], c["live"]) == (13, 10, 23)

            g = await w.regenerate(*lin._items_of(f"msg:{ben_turn}")[0])
            assert g["failed"] == [] and len(g["replaced"]) == 3
            new = [e for e in w.log if e["replaces"]]
            assert len(new) == 3 and all(
                "outside our 30-day" in e["reply"] or "don't see" in e["reply"] for e in new)
            view = Lineage(w).view()
            dev = next(e for e in w.log if e["customer"] == "dev")
            assert not any(x["quarantined"] for x in view["boxes"]
                           if x["item"] == ["mcp:crm", dev["reply_id"]])
            assert {x["kind"] for x in view["turns"]} == {"seed", "chat", "regenerated"}

            # Proof of removal: every excised record, sealed in a verified chain.
            from demo.support_agent.app import _report
            rep = _report(w, Lineage(w), f"msg:{ben_turn}")
            assert rep["chain"]["ok"]
            quarantined = [i for i in rep["items"] if i["status"] == "quarantined"]
            assert len(quarantined) >= 13  # items: a box can hold several versions
            assert {i["label"] for i in quarantined} >= set(c["lost_labels"])
            assert all(a["entry"] for i in quarantined for a in i["actions"])
            assert "# Excision report" in rep["markdown"]
        finally:
            await w.stop()

    asyncio.run(main())


def test_story_trace_blast_and_regenerate(tmp_path):
    run(tmp_path)


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("DOD_TEST_POSTGRES_URI"),
                    reason="DOD_TEST_POSTGRES_URI not set")
def test_story_on_postgres(tmp_path):
    pytest.importorskip("langgraph.store.postgres")
    run(tmp_path, os.environ["DOD_TEST_POSTGRES_URI"])
