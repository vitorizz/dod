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


def test_append_write_growing_context_p99_under_1ms(tmp_path):
    # Worst case for context sets: the context grows before every write, so
    # each write stores a new set of ~50 members (as many rows as before
    # context sets). Retrievals keep it at 50 by sliding the window.
    counter = iter(range(10**9))
    with Ledger(tmp_path / "ledger.db") as ledger:

        def append():
            i = next(counter)
            ledger.append_write(
                content="the user prefers window seats",
                store="mem",
                source="agent_inference",
                context=[("mem", f"rec-{j}", "retrieved_in_scope") for j in range(i, i + 50)],
            )

        p99 = p99_ms(append, n=5_000, rounds=3,
                     label="append_write (50 context, new set every write, on-disk)")
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
