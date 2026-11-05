"""Capture overhead with the audit chain on (``Ledger(path, audit=True)``)."""

import pytest
from perf import p99_ms

from dod.ledger import Ledger


@pytest.mark.parametrize("n", [5, 50])
def test_audited_append_write_p99_under_1ms(tmp_path, n):
    edges = [("mem", f"rec-{i}", "retrieved_in_scope") for i in range(n)]
    with Ledger(tmp_path / "ledger.db", audit=True) as ledger:

        def append():
            ledger.append_write(content="the user prefers window seats", store="mem",
                                source="agent_inference", agent_id="a1", session_id="s1",
                                context=edges)

        p99 = p99_ms(append, label=f"audited append_write ({n} context, on-disk)")
    assert p99 < 1.0


def test_audited_growing_context_p99_under_1ms(tmp_path):
    # A new context set every write: the chain hashes the set and its parts.
    counter = iter(range(10**9))
    context = []
    with Ledger(tmp_path / "ledger.db", audit=True) as ledger:

        def append():
            if len(context) == 50:
                context.clear()
            context.append(("mem", f"rec-{next(counter)}", "retrieved_in_scope"))
            ledger.append_write(content="x", store="mem", source="agent_inference",
                                agent_id="a1", session_id="s1", context=list(context))

        p99 = p99_ms(append, n=5_000, rounds=3,
                     label="audited append_write (context growing to 50)")
    assert p99 < 1.0


def test_audited_bind_and_quarantine_p99_under_1ms(tmp_path):
    with Ledger(tmp_path / "ledger.db", audit=True) as ledger:
        wids = iter([ledger.append_write(content=str(i), store="mem", source="user")
                     for i in range(11_000)])
        counter = iter(range(10**9))

        def bind():
            ledger.bind(next(wids), "mem", f"rec-{next(counter)}")

        def quarantine():
            ledger.quarantine([("mem", f"rec-{next(counter)}")], reason="test")

        b = p99_ms(bind, label="audited bind (on-disk)")
        q = p99_ms(quarantine, label="audited quarantine (1 item, on-disk)")
    assert b < 1.0 and q < 1.0
