import itertools

import pytest

pytest.importorskip("langgraph.store.memory")
pytestmark = pytest.mark.langgraph

from langgraph.store.memory import InMemoryStore  # noqa: E402
from perf import p99_ms  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.integrations.langgraph import instrument  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402

NS = ("users", "u1")


def _seeded(store):
    for i in range(10):
        store.put(("facts",), f"f{i}", {"text": f"fact {i}"})
    return store


def _put(store):
    keys = itertools.count()

    def run():
        store.put(NS, f"k{next(keys)}", {"text": "the user prefers window seats"})

    return run


def _search(store):
    def run():
        assert len(store.search(("facts",), limit=10)) == 10

    return run


def test_put_overhead_p99_under_1ms(tmp_path):
    # A new key each call: one pre-read, then two commits (the write, then its
    # bind), each carrying the 10 items retrieved in scope as parents.
    base = p99_ms(_put(_seeded(InMemoryStore())), n=5_000, rounds=3, label="raw put")
    with Ledger(tmp_path / "ledger.db") as ledger:
        store = instrument(_seeded(InMemoryStore()), Capture(ledger))
        with agent_task_scope("agent", "task"):
            store.search(("facts",), limit=10)
            p99 = p99_ms(
                _put(store), n=5_000, rounds=3,
                label="instrumented put (10 context parents, on-disk)",
            )
    print(f"put overhead p99 ~ {p99 - base:.4f} ms")
    assert p99 - base < 1.0


def test_search_overhead_p99_under_1ms(tmp_path):
    base = p99_ms(_search(_seeded(InMemoryStore())), rounds=3, label="raw search (10 results)")
    with Ledger(tmp_path / "ledger.db") as ledger:
        store = instrument(_seeded(InMemoryStore()), Capture(ledger))
        with agent_task_scope("agent", "task"):
            p99 = p99_ms(
                _search(store), rounds=3,
                label="instrumented search (10 results, on-disk)",
            )
    print(f"search overhead p99 ~ {p99 - base:.4f} ms")
    assert p99 - base < 1.0
