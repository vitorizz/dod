import itertools

import pytest

pytest.importorskip("opentelemetry.sdk.trace")
pytestmark = pytest.mark.otel

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from perf import p99_ms  # noqa: E402

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


def _tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("bench"), exporter


def _write(cap):
    def run():
        cap.reset("task")
        cap.on_retrieve("task", ITEMS)
        cap.on_write("task", TEXT, "mem", "agent_inference", "a1", "s1")

    return run


@pytest.mark.parametrize("active_span", [True, False], ids=["active-span", "standalone-span"])
def test_otel_observer_adds_under_0_1ms(active_span):
    tracer, exporter = _tracer()
    base = p99_ms(_write(Capture(StubLedger())), label="capture, no observer", rounds=3)
    cap = Capture(StubLedger(), observers=[OTelObserver(tracer=tracer)])
    if active_span:
        with tracer.start_as_current_span("step"):
            with_otel = p99_ms(_write(cap), label="capture + OTel (span event)", rounds=3)
        assert exporter.get_finished_spans()[0].events, "SDK recorded nothing"
    else:
        run = _write(cap)

        def run_and_drop():
            run()
            exporter.clear()

        with_otel = p99_ms(run_and_drop, label="capture + OTel (standalone span)", rounds=3)
    print(f"OTel observer adds p99 {with_otel - base:.4f} ms")
    assert with_otel - base < 0.1


def test_capture_with_otel_p99_under_1ms(tmp_path):
    tracer, _ = _tracer()
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger, observers=[OTelObserver(tracer=tracer)])
        with tracer.start_as_current_span("step"):
            p99 = p99_ms(_write(cap), label="on_retrieve + on_write + OTel (on-disk)", rounds=3)
    assert p99 < 1.0
