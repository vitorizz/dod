from perf import p99_ms

from dod.capture import Capture
from dod.ledger import Ledger

ITEMS = [("mem", f"rec-{i}") for i in range(10)]


def test_retrieve_and_write_p99_under_1ms(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger)

        def retrieve_and_write():
            cap.reset("task")
            cap.on_retrieve("task", ITEMS)
            cap.on_write("task", "the user prefers window seats", "mem", "agent_inference", "a1", "s1")

        p99 = p99_ms(retrieve_and_write, label="on_retrieve (10 items) + on_write (on-disk)")
    assert p99 < 1.0


def test_on_delete_p99_under_1ms(tmp_path):
    counter = iter(range(100_000))
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger)
        p99 = p99_ms(
            lambda: cap.on_delete("mem", [f"rec-{next(counter)}"], "forget", None, "a1", "s1"),
            label="on_delete (1 record, on-disk)",
        )
    assert p99 < 1.0


def test_bind_p99_under_1ms(tmp_path):
    n, warmup = 10_000, 1_000
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger)
        cap.on_retrieve("task", ITEMS)
        write_ids = iter(
            [
                cap.on_write("task", f"fact {i}", "mem", "agent_inference", "a1", "s1")
                for i in range(n + warmup)
            ]
        )
        record_ids = iter(range(n + warmup))

        def bind():
            cap.bind(next(write_ids), f"rec-{next(record_ids)}")

        p99 = p99_ms(bind, n=n, warmup=warmup, label="bind (on-disk)")
    assert p99 < 1.0


def test_on_retrieve_does_not_touch_database(tmp_path):
    class NoDb:
        def __getattr__(self, name):
            raise AssertionError(f"on_retrieve touched the ledger: {name}")

    cap = Capture(NoDb())
    cap.on_retrieve("task", ITEMS)
    cap.on_retrieve("task", ITEMS, tag="in_prompt")
    cap.reset("task")
