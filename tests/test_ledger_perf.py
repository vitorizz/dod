import pytest
from perf import p99_ms

from dod.ledger import Ledger


@pytest.mark.parametrize("n_parents", [5, 50])
def test_append_write_p99_under_1ms(tmp_path, n_parents):
    parents = [("mem", f"rec-{i}", "retrieved_in_scope") for i in range(n_parents)]
    with Ledger(tmp_path / "ledger.db") as ledger:

        def append():
            ledger.append_write(
                content="the user prefers window seats",
                store="mem",
                source="agent_inference",
                agent_id="a1",
                session_id="s1",
                parents=parents,
            )

        p99 = p99_ms(append, label=f"append_write ({n_parents} parents, on-disk)")
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
