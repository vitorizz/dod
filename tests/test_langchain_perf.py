import itertools

import pytest

pytest.importorskip("langchain_core.vectorstores")
pytestmark = pytest.mark.langchain

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.vectorstores import InMemoryVectorStore  # noqa: E402
from perf import p99_ms  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.integrations.langchain import instrument  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402


def _store():
    vs = InMemoryVectorStore(DeterministicFakeEmbedding(size=8))
    vs.add_texts([f"fact {i}" for i in range(10)], ids=[f"f{i}" for i in range(10)])
    return vs


def _add(vs):
    ids = itertools.count()

    def run():
        vs.add_texts(["the user prefers window seats"], ids=[f"k{next(ids)}"])

    return run


def _search(vs):
    def run():
        assert len(vs.similarity_search("fact", k=10)) == 10

    return run


def test_add_overhead_p99_under_1ms(tmp_path):
    # Pre-read, the write, the store call (embedding), then the bind.
    base = p99_ms(_add(_store()), n=5_000, rounds=3, label="raw add_texts")
    with Ledger(tmp_path / "ledger.db") as ledger:
        vs = _store()
        instrument(vs, Capture(ledger))
        with agent_task_scope("agent", "task"):
            vs.similarity_search("fact", k=10)
            p99 = p99_ms(_add(vs), n=5_000, rounds=3,
                         label="instrumented add_texts (10 context parents, on-disk)")
    print(f"add_texts overhead p99 ~ {p99 - base:.4f} ms")
    assert p99 - base < 1.0


def test_search_overhead_p99_under_1ms(tmp_path):
    vs = _store()
    base = p99_ms(_search(vs), rounds=3, label="raw similarity_search (10 results)")
    with Ledger(tmp_path / "ledger.db") as ledger:
        instrument(vs, Capture(ledger))
        with agent_task_scope("agent", "task"):
            p99 = p99_ms(_search(vs), rounds=3,
                         label="instrumented similarity_search (10 results)")
    print(f"search overhead p99 ~ {p99 - base:.4f} ms")
    assert p99 - base < 1.0
