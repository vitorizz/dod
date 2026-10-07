from perf import p99_ms

from dod.ledger import Ledger

PARENTS = [("mem", f"rec-{i}", "retrieved_in_scope") for i in range(5)]


def test_append_write_p99_under_1ms(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:

        def append():
            ledger.append_write(
                content="the user prefers window seats",
                store="mem",
                source="agent_inference",
                agent_id="a1",
                session_id="s1",
                parents=PARENTS,
            )

        p99 = p99_ms(append, label="append_write (5 parents, on-disk)")
    assert p99 < 1.0
