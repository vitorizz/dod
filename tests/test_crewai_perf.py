from perf import p99_ms
from test_crewai import StubMemory

from dod.capture import Capture
from dod.integrations.crewai import agent_task_scope, instrument
from dod.ledger import Ledger


def _recall_and_remember(mem):
    def run():
        mem.recall("q")
        mem.remember("the user prefers window seats")

    return run


def test_wrapper_overhead_p99_under_1ms(tmp_path):
    # No save pool: the stub's own cost is near zero, so the instrumented
    # call's p99 bounds the wrapper's overhead from above.
    plain = StubMemory(pool=False)
    plain.results["q"] = plain.add(*(f"rec-{i}" for i in range(10)))
    base = p99_ms(_recall_and_remember(plain), label="stub recall (10 results) + remember")

    mem = StubMemory(pool=False)
    mem.results["q"] = mem.add(*(f"rec-{i}" for i in range(10)))
    with Ledger(tmp_path / "ledger.db") as ledger:
        instrument(mem, Capture(ledger))
        with agent_task_scope("agent", "task"):
            p99 = p99_ms(
                _recall_and_remember(mem),
                label="instrumented recall (10 results) + remember (on-disk)",
            )
    print(f"wrapper overhead p99 <= {p99:.4f} ms (stub alone p99={base:.4f} ms)")
    assert p99 < 1.0


def _consolidating_remember(mem):
    # Consults 5 records, rewrites 2, deletes 1 and inserts 1: one write plus
    # three compactions, each with the 5 consulted records as parents.
    def run():
        mem.add("rec-0", "rec-1", "rec-2", "rec-3", "rec-4")
        mem._storage.similar = ["rec-0", "rec-1", "rec-2", "rec-3", "rec-4"]
        mem.plan = {"update": {"rec-0": "merged 0", "rec-1": "merged 1"},
                    "delete": ["rec-2"], "insert": True}
        mem.remember("the user prefers window seats")

    return run


def test_consolidating_remember_p99_under_1ms(tmp_path):
    plain = StubMemory(pool=False)
    base = p99_ms(_consolidating_remember(plain), label="stub consolidating remember")

    mem = StubMemory(pool=False)
    with Ledger(tmp_path / "ledger.db") as ledger:
        instrument(mem, Capture(ledger))
        with agent_task_scope("agent", "task"):
            mem.recall("q")
            p99 = p99_ms(
                _consolidating_remember(mem),
                label="instrumented consolidating remember (on-disk)",
            )
    print(f"consolidating remember p99={p99:.4f} ms (stub alone p99={base:.4f} ms)")
    assert p99 < 1.0


def test_remember_many_p99_under_1ms(tmp_path):
    # No save pool: the timed call is both transactions, the writes at call
    # time and the binds after the save. Each call writes ~80 KB of WAL; back
    # to back that is ~200 MB/s and the kernel throttles writeback, so pause
    # between calls (a real batch spends far longer embedding and in the LLM).
    items = [f"fact {i}" for i in range(5)]
    mem = StubMemory(pool=False)
    mem.results["q"] = mem.add(*(f"rec-{i}" for i in range(10)))
    with Ledger(tmp_path / "ledger.db") as ledger:
        instrument(mem, Capture(ledger))
        with agent_task_scope("agent", "task"):
            mem.recall("q")
            p99 = p99_ms(
                lambda: mem.remember_many(items),
                n=3_000,
                pause_s=0.002,
                label="instrumented remember_many (5 items, 10 parents, on-disk)",
            )
    assert p99 < 1.0
