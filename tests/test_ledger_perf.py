import random
import time

import pytest
from perf import p99_ms

from dod.ledger import Ledger


@pytest.mark.parametrize("as_", ["parents", "context"])
@pytest.mark.parametrize("n_parents", [5, 50])
def test_append_write_p99_under_1ms(tmp_path, n_parents, as_):
    # "parents": per-write direct parents, one row each per write. "context":
    # the scope's context (what Capture passes), stored once as a context set
    # and referenced by every later write with the same context.
    edges = [("mem", f"rec-{i}", "retrieved_in_scope") for i in range(n_parents)]
    with Ledger(tmp_path / "ledger.db") as ledger:

        def append():
            ledger.append_write(
                content="the user prefers window seats",
                store="mem",
                source="agent_inference",
                agent_id="a1",
                session_id="s1",
                **{as_: edges},
            )

        p99 = p99_ms(append, label=f"append_write ({n_parents} {as_}, on-disk)")
    assert p99 < 1.0


def _growing_context_p99(tmp_path, size):
    # A long session that retrieves before every write: the context grows by
    # one item per write, so each write references a new set. At ``size``
    # items the scope ends and the next one starts empty.
    counter = iter(range(10**9))
    context = []
    with Ledger(tmp_path / "ledger.db") as ledger:

        def append():
            if len(context) == size:
                context.clear()
            context.append(("mem", f"rec-{next(counter)}", "retrieved_in_scope"))
            ledger.append_write(
                content="the user prefers window seats",
                store="mem",
                source="agent_inference",
                agent_id="a1",
                session_id="s1",
                context=list(context),
            )

        return p99_ms(append, n=5_000, rounds=3,
                      label=f"append_write (context growing to {size}, new set every write)")


def test_append_write_growing_context_p99_under_half_ms(tmp_path):
    assert _growing_context_p99(tmp_path, 50) < 0.5


def test_append_write_growing_500_context_p99_under_1ms(tmp_path):
    assert _growing_context_p99(tmp_path, 500) < 1.0


@pytest.mark.parametrize("as_", ["parents", "context"])
def test_append_write_fresh_random_parents_p99_under_1ms(tmp_path, as_):
    # 50 parents drawn at random from 100k items for every write: the case an
    # inline parent->child index would make dirty one B-tree leaf per parent.
    rng = random.Random(50)
    with Ledger(tmp_path / "ledger.db") as ledger:

        def append():
            ledger.append_write(
                content="the user prefers window seats",
                store="mem",
                source="agent_inference",
                **{as_: [("mem", f"rec-{rng.randrange(100_000)}", "retrieved_in_scope")
                         for _ in range(50)]},
            )

        p99 = p99_ms(append, n=5_000, rounds=3,
                     label=f"append_write (50 fresh random {as_}, on-disk)")
    assert p99 < 1.0


def _watermark(ledger):
    with ledger._index_lock:
        return ledger._index_watermarks(ledger._index_conn)[0]


@pytest.mark.parametrize("as_", ["parents", "context"])
def test_append_write_during_100k_backlog_fill_p99_under_1ms(tmp_path, as_):
    # A ledger whose 100k writes (2 random parents each) aren't in
    # the lineage index yet, reopened so the background thread starts filling it,
    # then appends with 50 fresh random parents timed while the fill runs.
    path = tmp_path / "ledger.db"
    rng = random.Random(100)
    with Ledger(path, checkpoint_interval=3600, background_index=False) as ledger:  # never fills
        for b in range(100):
            ledger.append_batch([
                dict(content="x", store="mem", source="agent_inference",
                     record_id=f"rec-{i}",
                     parents=[("mem", f"rec-{rng.randrange(100_000)}", "in_prompt")
                              for _ in range(2)])
                for i in range(b * 1_000, (b + 1) * 1_000)
            ])
    with Ledger(path) as ledger:
        deadline = time.monotonic() + 30
        while _watermark(ledger) == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        before = _watermark(ledger)
        assert before > 0, "the fill never started"

        def append():
            ledger.append_write(
                content="the user prefers window seats",
                store="mem",
                source="agent_inference",
                **{as_: [("mem", f"rec-{rng.randrange(100_000)}", "retrieved_in_scope")
                         for _ in range(50)]},
            )

        p99 = p99_ms(append, n=5_000, rounds=3,
                     label=f"append_write (50 fresh random {as_}, during 100k backlog fill)")
        after = _watermark(ledger)
        print(f"fill progress during the run: seq {before} -> {after}")
    assert before < 100_000 and after > before  # measured while the fill ran
    assert p99 < 1.0


def test_append_tombstone_p99_under_1ms(tmp_path):
    counter = iter(range(100_000))
    with Ledger(tmp_path / "ledger.db") as ledger:
        wid = ledger.append_write(content="x", store="mem", source="agent_inference")

        def append():
            ledger.append_tombstones([dict(
                store="mem", record_id=f"rec-{next(counter)}", reason="consolidation",
                write_id=wid, agent_id="a1", session_id="s1",
            )])

        p99 = p99_ms(append, label="append_tombstones (1 tombstone, on-disk)")
    assert p99 < 1.0


def test_bind_p99_under_1ms(tmp_path):
    n, warmup = 10_000, 1_000
    with Ledger(tmp_path / "ledger.db") as ledger:
        # One bind per write, as in real use.
        write_ids = iter(
            [
                ledger.append_write(content=str(i), store="mem", source="agent_inference")
                for i in range(n + warmup)
            ]
        )
        counter = iter(range(n + warmup))

        def bind():
            ledger.bind(next(write_ids), "mem", f"rec-{next(counter)}")

        p99 = p99_ms(bind, n=n, warmup=warmup, label="bind (on-disk)")
    assert p99 < 1.0
