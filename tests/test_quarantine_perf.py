import itertools
import time

from perf import p99_ms, split_p99_ms
from test_crewai import StubMemory

from dod.capture import Capture
from dod.integrations.crewai import agent_task_scope, instrument
from dod.ledger import Ledger


def test_quarantine_100_items_p99_under_1ms_per_item(tmp_path):
    batch = itertools.count()
    with Ledger(tmp_path / "ledger.db") as ledger:

        def run():
            b = next(batch)
            ledger.quarantine([("crewai", f"rec-{b}-{i}") for i in range(100)],
                              reason="benchmark", actor="perf")

        p99 = p99_ms(run, n=2_000, warmup=200, rounds=3,
                     label="Ledger.quarantine (100 items, on-disk)")
    print(f"per item p99={p99 / 100:.5f} ms")
    assert p99 / 100 < 1.0


def test_recall_with_1000_quarantined_p99_under_1ms(tmp_path):
    # 10 results, 2 of them quarantined, among 1,000 quarantined items; the
    # cached set is reused, so each recall only checks for newer actions.
    mem = StubMemory(pool=False)
    mem.results["q"] = mem.add(*(f"rec-{i}" for i in range(10)))
    with Ledger(tmp_path / "ledger.db") as ledger:
        ledger.quarantine([("crewai", f"other-{i}") for i in range(998)])
        ledger.quarantine([("crewai", "rec-3"), ("crewai", "rec-7")])
        instrument(mem, Capture(ledger))
        with agent_task_scope("agent", "task"):
            assert len(mem.recall("q")) == 8
            p99 = p99_ms(lambda: mem.recall("q"),
                         label="instrumented recall (10 results, 1,000 quarantined)")
    assert p99 < 1.0


def test_recall_after_quarantine_change_p99_under_1ms(tmp_path):
    # Worst case: every recall follows a quarantine change, among 1,000
    # quarantined items. Only the recall is held to the bar; the change (a
    # ledger commit) is timed separately.
    mem = StubMemory(pool=False)
    mem.results["q"] = mem.add(*(f"rec-{i}" for i in range(10)))
    flip, inner = itertools.count(), []
    clock = time.perf_counter_ns
    with Ledger(tmp_path / "ledger.db") as ledger:
        ledger.quarantine([("crewai", f"other-{i}") for i in range(1_000)])
        instrument(mem, Capture(ledger))
        with agent_task_scope("agent", "task"):

            def run():
                # Alternate so the set stays at ~1,000 items.
                action = ledger.release if next(flip) % 2 else ledger.quarantine
                action([("crewai", "rec-3")])
                start = clock()
                mem.recall("q")
                inner.append(clock() - start)

            _, recall_p99 = split_p99_ms(
                run, inner, n=2_000, warmup=200, rounds=3,
                labels=("quarantine change (1 item, on-disk)",
                        "instrumented recall after a change (1,000 quarantined)"),
            )
    assert recall_p99 < 1.0
