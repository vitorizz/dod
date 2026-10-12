import itertools

import pytest

pytest.importorskip("opentelemetry.sdk.trace")
pytestmark = pytest.mark.otel

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import (  # noqa: E402
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from perf import p50_p99_ms, p99_ms  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.export.otel import OTelObserver  # noqa: E402
from dod.ledger import Ledger  # noqa: E402

ITEMS = [("mem", f"rec-{i}") for i in range(10)]
TEXT = "the user prefers window seats"


class StubLedger:
    """Commits instantly, so the comparison measures the observer, not the disk."""

    def __init__(self):
        self._ids = itertools.count()

    def append_write(self, **_):
        return f"{next(self._ids):032x}"


@pytest.fixture(autouse=True)
def _sdk_enabled(monkeypatch):
    # test_crewai disables the OTel SDK process-wide; a disabled SDK would
    # benchmark a no-op tracer.
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)


class CountingExporter(SpanExporter):
    """Counts finished spans and keeps none, so a long benchmark neither grows
    memory nor needs an untimed clear inside the timed call."""

    def __init__(self):
        self.count = 0

    def export(self, spans):
        self.count += len(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def _tracer(exporter=None):
    exporter = exporter or InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("bench"), exporter


def _write(cap):
    def run():
        cap.reset("task")
        cap.on_retrieve("task", ITEMS)
        cap.on_write("task", TEXT, "mem", "agent_inference", "a1", "s1")

    return run


ROUNDS = 5


def _overhead_ms(base, with_observer, label):
    """Median over interleaved rounds of (p50 with the observer - p50 without).

    The observer adds a fixed amount of work per write, which the median shows
    directly; a difference of two p99s mostly measures runner noise in the tail.
    The tail is guarded by the absolute bar in
    ``test_capture_with_otel_p99_under_1ms``. Each round measures both sides back
    to back, so a burst of noise shifts one round, and the median of rounds
    ignores it. A real regression raises every round, so the median catches it.
    """
    diffs = []
    for r in range(ROUNDS):
        b, _ = p50_p99_ms(base, label=f"capture, no observer (round {r + 1}/{ROUNDS})")
        w, _ = p50_p99_ms(with_observer, label=f"{label} (round {r + 1}/{ROUNDS})")
        diffs.append(w - b)
    return sorted(diffs)[ROUNDS // 2]


@pytest.mark.parametrize("active_span", [True, False], ids=["active-span", "standalone-span"])
def test_otel_observer_adds_under_0_1ms(active_span):
    base = _write(Capture(StubLedger()))
    if active_span:
        tracer, exporter = _tracer()
        cap = Capture(StubLedger(), observers=[OTelObserver(tracer=tracer)])
        with tracer.start_as_current_span("step"):
            added = _overhead_ms(base, _write(cap), "capture + OTel (span event)")
        assert exporter.get_finished_spans()[0].events, "SDK recorded nothing"
    else:
        tracer, exporter = _tracer(CountingExporter())
        cap = Capture(StubLedger(), observers=[OTelObserver(tracer=tracer)])
        added = _overhead_ms(base, _write(cap), "capture + OTel (standalone span)")
        assert exporter.count > 0, "SDK exported nothing"
    print(f"OTel observer adds p50 {added:.4f} ms (median of {ROUNDS} interleaved rounds)")
    assert added < 0.1


def test_capture_with_otel_p99_under_1ms(tmp_path):
    tracer, _ = _tracer()
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger, observers=[OTelObserver(tracer=tracer)])
        with tracer.start_as_current_span("step"):
            p99 = p99_ms(_write(cap), label="on_retrieve + on_write + OTel (on-disk)", rounds=3)
    assert p99 < 1.0
