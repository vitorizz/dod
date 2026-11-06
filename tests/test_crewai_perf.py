import time

import perf
from perf import p99_ms, split_p99_ms
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
                n=5_000,
                rounds=3,
                label="instrumented consolidating remember (on-disk)",
            )
    print(f"consolidating remember p99={p99:.4f} ms (stub alone p99={base:.4f} ms)")
    assert p99 < 1.0


def test_forget_one_record_p99_under_1ms(tmp_path):
    # The common delete: get_record before and after, and one tombstone
    # commit. Each call adds a fresh record (a dict insert) and forgets it.
    mem = StubMemory(pool=False)
    mem.add(*(f"rec-{i}" for i in range(100)))  # unrelated records stay put
    ids = iter(range(10**9))

    def forget():
        rid = f"del-{next(ids)}"
        mem.add(rid)
        mem.forget(record_ids=[rid])

    with Ledger(tmp_path / "ledger.db") as ledger:
        instrument(mem, Capture(ledger))
        with agent_task_scope("agent", "task"):
            p99 = p99_ms(forget, n=5_000, rounds=3,
                         label="instrumented forget(record_ids=[x]) (on-disk)")
        (count,) = ledger._conn.execute("SELECT COUNT(*) FROM tombstones").fetchone()
    assert count == 1_000 + 5_000 * 3
    assert len(mem._storage.records) == 100
    assert p99 < 1.0


def test_filter_delete_1000_records(tmp_path):
    # Off the hot path, so held to a per-call ceiling instead of the 1 ms bar
    # (docs/decisions.md): reset() lists the store before and after deleting,
    # and tombstones all 1,000 records in one transaction. Each call writes
    # ~5 MB of WAL, so every dozen calls or so one pays the WAL restart
    # (a checkpoint and a disk sync): that is checkpoint cost, not capture, and
    # it sets this op's p99 on a slow disk. The guard is on the median (~15 ms
    # on a dev machine), which a regression in the call itself moves.
    import statistics

    mem = StubMemory(pool=False)
    with Ledger(tmp_path / "ledger.db") as ledger:
        instrument(mem, Capture(ledger))
        ids = [f"rec-{i}" for i in range(1_000)]
        times = []
        with agent_task_scope("agent", "task"):
            for i in range(220):
                mem.add(*ids)
                t = time.perf_counter_ns()
                mem.reset()
                if i >= 20:
                    times.append((time.perf_counter_ns() - t) / 1e6)
        (count,) = ledger._conn.execute("SELECT COUNT(*) FROM tombstones").fetchone()
    median = statistics.median(times) / perf.SCALE
    print(f"instrumented reset (1,000 records, on-disk): median={median:.2f} ms"
          f" max={max(times):.1f} ms (n={len(times)})")
    assert count == 1_000 * 220
    assert median < 60.0
    # Tail ceiling: a WAL restart (copy, sync, truncate) on a slow CI disk.
    assert sorted(times)[int(len(times) * 0.99)] / perf.SCALE < 1_000.0


def test_update_p99_under_1ms(tmp_path):
    # Two commits, like remember: the compaction before the store write, the
    # bind after it.
    mem = StubMemory(pool=False)
    mem.results["q"] = mem.add(*(f"rec-{i}" for i in range(10)))
    with Ledger(tmp_path / "ledger.db") as ledger:
        instrument(mem, Capture(ledger))
        with agent_task_scope("agent", "task"):
            mem.recall("q")
            p99 = p99_ms(
                lambda: mem.update("rec-0", "the user prefers aisle seats"),
                n=5_000,
                rounds=3,
                label="instrumented update (10 context parents, on-disk)",
            )
    assert p99 < 1.0


def test_remember_many_p99_under_1ms(tmp_path):
    # The caller waits only for the call-time commit (the writes); the bind
    # commit runs later on crewai's save pool. Time each on its own. No save
    # pool here, so the background save runs inside the call: a shim around
    # the instrumented _background_encode_batch times it, and the rest of the
    # call is the call-time part. Each call writes ~80 KB of WAL; back to back
    # that is ~200 MB/s and the kernel throttles writeback, so pause between
    # calls (a real batch spends far longer embedding and in the LLM).
    items = [f"fact {i}" for i in range(5)]
    mem = StubMemory(pool=False)
    mem.results["q"] = mem.add(*(f"rec-{i}" for i in range(10)))
    with Ledger(tmp_path / "ledger.db") as ledger:
        instrument(mem, Capture(ledger))
        background, inner = mem._background_encode_batch, []
        clock = time.perf_counter_ns

        def timed_background(contents):
            start = clock()
            try:
                return background(contents)
            finally:
                inner.append(clock() - start)

        mem._background_encode_batch = timed_background
        with agent_task_scope("agent", "task"):
            mem.recall("q")
            call_p99, bind_p99 = split_p99_ms(
                lambda: mem.remember_many(items),
                inner,
                n=2_000,
                pause_s=0.002,
                rounds=3,
                labels=(
                    "instrumented remember_many call-time commit (5 items, 10 parents, on-disk)",
                    "instrumented remember_many background bind commit (5 items, on-disk)",
                ),
            )
    # The items share the scope's context, stored once as a context set, so
    # the whole call-time commit is held to the 1 ms bar.
    assert call_p99 < 1.0
    assert bind_p99 < 1.0
