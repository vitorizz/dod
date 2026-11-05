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
    async def main():
        w = World(str(tmp_path / "world"), "scripted", pg)
        await w.start()
        try:
            await w.run_story()
            replies = {e["customer"] + str(i): e for i, e in enumerate(w.log)}
            chloe = next(e for e in w.log if e["customer"] == "chloe")
            assert "90-day" in chloe["reply"]  # the false policy reached her
            t = w.trace_view("mcp:crm", chloe["reply_id"])
            users = [r for r in t["roots"] if r["source"] == "user"]
            assert {r["label"] for r in users} == {"ben's conversation", "chloe's conversation"}
            ben = next(r for r in users if r["label"] == "ben's conversation")
            b = w.blast_view(ben["store"], ben["record_id"])
            assert set(b["per_store"]) == {"langgraph_state", "memory", "notes", "mcp:crm"}
            c = w.compare(ben["store"], ben["record_id"])
            assert c["excise"] < c["live"] and c["kept"] >= 10
            g = await w.regenerate(ben["store"], ben["record_id"])
            assert g["failed"] == [] and len(g["replaced"]) == 3
            new = w.log[-3:]
            assert all("outside our 30-day" in e["reply"] or "don't see" in e["reply"]
                       for e in new)
            dev = next(e for e in w.log if e["customer"] == "dev")
            assert not any(n["quarantined"] for n in w.graph_view()["nodes"]
                           if n["record_id"] == dev["reply_id"])
            return replies
        finally:
            await w.stop()

    return asyncio.run(main())


def test_story_trace_blast_and_regenerate(tmp_path):
    run(tmp_path)


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("DOD_TEST_POSTGRES_URI"),
                    reason="DOD_TEST_POSTGRES_URI not set")
def test_story_on_postgres(tmp_path):
    pytest.importorskip("langgraph.store.postgres")
    run(tmp_path, os.environ["DOD_TEST_POSTGRES_URI"])
