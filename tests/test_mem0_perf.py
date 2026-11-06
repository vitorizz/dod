"""dod's own overhead on Mem0 calls: Mem0's work (LLM, embedding, vector
store) is replaced with instant stubs, so the timing is the wrapper's."""

import os
import types
import uuid

import pytest

os.environ["MEM0_TELEMETRY"] = "False"
pytest.importorskip("mem0")
pytestmark = pytest.mark.mem0

from mem0 import Memory  # noqa: E402
from perf import p99_ms  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.integrations import mem0 as m0  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402

HITS = [types.SimpleNamespace(id=f"m{i}", payload={"data": f"existing memory number {i}"})
        for i in range(10)]
RESULTS = {"results": [{"id": f"m{i}", "memory": f"existing memory number {i}"}
                       for i in range(10)]}


def stubbed():
    m = Memory.__new__(Memory)  # no clients, no files
    m.vector_store = types.SimpleNamespace(search=lambda *a, **k: HITS)

    def add(messages, **kwargs):
        m.vector_store.search(query=messages)  # Mem0's internal retrieval
        return {"results": [{"id": str(uuid.uuid4()), "memory": "a new fact", "event": "ADD"}]}

    m.add = add
    m.search = lambda query, **kwargs: RESULTS
    for name in ("get", "get_all", "update", "_delete_memory"):
        setattr(m, name, lambda *a, **k: None)
    return m


def test_add_overhead_p99_under_1ms(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        m = m0.instrument(stubbed(), Capture(ledger))
        with agent_task_scope("agent", "task"):
            p99 = p99_ms(lambda: m.add("hello", user_id="u1"), n=5_000, rounds=3,
                         label="instrumented mem0 add (10 internal hits, on-disk)")
    assert p99 < 1.0


def test_search_overhead_p99_under_1ms(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        m = m0.instrument(stubbed(), Capture(ledger))
        with agent_task_scope("agent", "task"):
            p99 = p99_ms(lambda: m.search("q", filters={"user_id": "u1"}), rounds=3,
                         label="instrumented mem0 search (10 results)")
    assert p99 < 1.0
