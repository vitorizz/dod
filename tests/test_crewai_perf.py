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
