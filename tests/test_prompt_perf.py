import random

from perf import p99_ms

from dod.capture import Capture
from dod.ledger import Ledger

random.seed(0)
WORDS = "refund policy order delivery shipping warranty customer days approved note".split()


def text(n=300):
    return " ".join(random.choice(WORDS) for _ in range(n // 7))


def test_on_retrieve_with_texts_p99_under_1ms(tmp_path):
    # A search returning 10 results with ~300 chars each: probes computed
    # and held in memory on the retrieval path.
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger)
        items = [("mem", f"r{i}") for i in range(10)]
        texts = [text() for _ in items]
        key = ("agent", "t")

        def run():
            cap.reset(key)
            cap.on_retrieve(key, items, texts=texts)

        assert p99_ms(run, rounds=3, label="on_retrieve, 10 results with text") < 1.0


def test_confirm_p99_under_1ms(tmp_path):
    # 50 retrieved items in scope, checked against an 8k-character prompt that
    # holds 10 of them.
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger)
        items = [("mem", f"r{i}") for i in range(50)]
        texts = [f"item {i}: " + text() for i in range(50)]
        prompt = "\n".join(texts[:10]) + "\n" + text(8000 - sum(map(len, texts[:10])))
        key = ("agent", "t")

        def run():
            cap.reset(key)
            cap.on_retrieve(key, items, texts=texts)
            assert len(cap.confirm(key, prompt)) == 10

        assert p99_ms(run, n=5_000, rounds=3, label="retrieve 50 + confirm (8k prompt)") < 1.0
