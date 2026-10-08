"""One ledger, one scope, two stores: lineage crosses the store boundary."""

import pytest

pytest.importorskip("langgraph.store.memory")
pytestmark = pytest.mark.langgraph

from langgraph.store.memory import InMemoryStore  # noqa: E402
from test_crewai import StubMemory  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.integrations import crewai, langgraph  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402

NS = ("users", "u1")


def test_crewai_is_core_scope():
    assert crewai.agent_task_scope is agent_task_scope


def test_trace_and_blast_radius_cross_stores(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger)
        mem = crewai.instrument(StubMemory(pool=False), cap)
        store = langgraph.instrument(InMemoryStore(), cap)

        with agent_task_scope("agent", "plant"):
            root = mem.remember("the user is allergic to peanuts")  # false

        # Retrieve from CrewAI, write to LangGraph, in one scope.
        with agent_task_scope("agent", "plan"):
            mem.results["diet"] = [root]
            mem.recall("diet")
            derived = mem.remember("avoid Thai restaurants for the user")
            store.put(NS, "diet", {"text": "no peanuts in meal plans"})

        # And back: retrieve from LangGraph, write to CrewAI.
        with agent_task_scope("agent", "shop"):
            store.get(NS, "diet")
            back = mem.remember("removed peanut butter from the grocery list")

        lg_id = langgraph.record_id(NS, "diet")
        t = trace(ledger, "langgraph", lg_id)
        (root_write,) = trace(ledger, "crewai", root.id).items
        assert [w.write_id for w in t.roots] == [root_write.write_id]
        assert [w.store for w in t.ancestors] == ["crewai"]
        assert [(e.parent_store, e.parent_record_id) for e in t.edges] == [("crewai", root.id)]

        t_back = trace(ledger, "crewai", back.id)
        assert [w.write_id for w in t_back.roots] == [root_write.write_id]
        assert [w.store for w in t_back.ancestors] == ["langgraph", "crewai"]

        b = blast_radius(ledger, "crewai", root.id)
        assert b.store_counts == {"crewai": 2, "langgraph": 1}
        reached = {item for items in b.bindings.values() for item in items}
        assert {("crewai", derived.id), ("langgraph", lg_id), ("crewai", back.id)} <= reached
