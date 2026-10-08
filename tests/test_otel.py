import hashlib
import logging

import pytest

from dod.capture import BindEvent, Capture, TombstoneEvent, WriteEvent
from dod.ledger import Ledger

TEXT = "the user prefers window seats"
ITEMS = [("mem", "rec-1"), ("mem", "rec-2")]


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        yield ledger


class Recorder:
    def __init__(self):
        self.events = []

    def on_write(self, event):
        self.events.append(event)

    def on_bind(self, event):
        self.events.append(event)

    def on_tombstone(self, event):
        self.events.append(event)


class Raises:
    def on_write(self, event):
        raise RuntimeError("observer down")

    on_bind = on_tombstone = on_write


# Observer hook (core, no OpenTelemetry needed).


def test_write_event_after_commit(ledger):
    rec = Recorder()
    cap = Capture(ledger, observers=[rec])
    cap.on_retrieve("t", ITEMS)
    cap.on_retrieve("t", [("mem", "rec-2")], tag="in_prompt")
    w = cap.on_write("t", TEXT, "mem", "agent_inference", "a1", "s1")
    assert rec.events == [WriteEvent(
        w, "mem", None, "agent_inference", hashlib.sha256(TEXT.encode()).hexdigest(),
        "a1", "s1", (("mem", "rec-1", "retrieved_in_scope"), ("mem", "rec-2", "in_prompt")),
    )]
    assert ledger._conn.execute(
        "SELECT content_hash FROM writes WHERE write_id = ?", (w,)
    ).fetchone()[0] == rec.events[0].content_hash


def test_compaction_event_merges_replaced_and_context(ledger):
    rec = Recorder()
    cap = Capture(ledger, observers=[rec])
    cap.on_retrieve("t", ITEMS)
    cap.on_compaction("t", [("mem", "rec-1"), ("mem", "old")], b"merged", "mem", "agent_inference")
    assert rec.events[0].parents == (
        ("mem", "rec-1", "in_prompt"), ("mem", "rec-2", "retrieved_in_scope"),
        ("mem", "old", "in_prompt"),
    )


def test_bind_and_tombstone_events(ledger):
    rec = Recorder()
    cap = Capture(ledger, observers=[rec])
    w = cap.on_write("t", TEXT, "mem", "user")
    cap.bind(w, "rec-9")
    cap.on_delete("mem", ["rec-9"], "forget", w, "a1", "s1")
    assert rec.events[1:] == [
        BindEvent(w, "mem", "rec-9"),
        TombstoneEvent("mem", "rec-9", "forget", w, "a1", "s1"),
    ]


def test_batch_events(ledger):
    rec = Recorder()
    cap = Capture(ledger, observers=[rec])
    earlier = cap.on_write("t", "x", "mem", "user")
    cap.on_retrieve("t", ITEMS)
    ids = cap.on_batch(
        "t", "mem",
        writes=[((), TEXT, "agent_inference", "rec-a"), ([("mem", "rec-1")], "y", "tool", None)],
        binds=[(earlier, "rec-0")],
        deletes=[("rec-2", "forget", None)],
        agent_id="a1",
    )
    events = rec.events[1:]
    assert [type(e) for e in events] == [WriteEvent, WriteEvent, BindEvent, TombstoneEvent]
    assert [e.write_id for e in events[:2]] == ids
    assert events[0].record_id == "rec-a" and events[1].record_id is None
    assert events[1].parents[0] == ("mem", "rec-1", "in_prompt")
    assert events[2] == BindEvent(earlier, "mem", "rec-0")
    assert events[3] == TombstoneEvent("mem", "rec-2", "forget", None, "a1", None)


def test_no_event_when_ledger_fails(ledger):
    rec = Recorder()
    cap = Capture(ledger, observers=[rec])
    with pytest.raises(ValueError):
        cap.on_write("t", TEXT, "mem", "not-a-source")
    assert rec.events == []


def test_raising_observer_fails_open(ledger, caplog):
    rec = Recorder()
    cap = Capture(ledger, observers=[Raises(), rec])
    with caplog.at_level(logging.ERROR, logger="dod.capture"):
        w = cap.on_write("t", TEXT, "mem", "user")
        cap.bind(w, "rec-1")
        cap.on_delete("mem", ["rec-1"], "forget")
    assert len(rec.events) == 3
    assert ledger._conn.execute("SELECT count(*) FROM writes").fetchone()[0] == 1
    assert sum("observer down" in r.exc_text for r in caplog.records if r.exc_text) == 3


def test_observer_methods_are_optional(ledger):
    class OnlyWrites:
        def __init__(self):
            self.n = 0

        def on_write(self, event):
            self.n += 1

    obs = OnlyWrites()
    cap = Capture(ledger, observers=[obs])
    cap.bind(cap.on_write("t", TEXT, "mem", "user"), "rec-1")
    cap.on_delete("mem", ["rec-1"], "forget")
    assert obs.n == 1


# OpenTelemetry observer.


@pytest.fixture
def otel(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    # test_crewai disables the OTel SDK process-wide to silence crewai's own
    # telemetry; these tests need a working SDK.
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from dod.export.otel import OTelObserver

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter, OTelObserver(tracer=tracer)


def _all_attrs(spans):
    for span in spans:
        yield dict(span.attributes or {})
        for event in span.events:
            yield dict(event.attributes or {})


def _run(ledger, observer):
    cap = Capture(ledger, observers=[observer])
    cap.on_retrieve("t", ITEMS)
    w = cap.on_write("t", TEXT, "mem", "agent_inference", "a1", "s1")
    cap.bind(w, "rec-9")
    cap.on_delete("mem", ["rec-9"], "forget", w, "a1", "s1")
    return w


@pytest.mark.otel
def test_events_on_active_span(ledger, otel):
    tracer, exporter, observer = otel
    with tracer.start_as_current_span("agent-step"):
        w = _run(ledger, observer)
    (span,) = exporter.get_finished_spans()
    assert span.name == "agent-step"
    write, bind, delete = span.events
    assert write.name == "dod.memory_write"
    assert dict(write.attributes) == {
        "dod.write.id": w,
        "dod.write.store": "mem",
        "dod.write.source": "agent_inference",
        "dod.write.content_hash": hashlib.sha256(TEXT.encode()).hexdigest(),
        "dod.agent.id": "a1",
        "dod.session.id": "s1",
        "dod.parents.count": 2,
        "dod.parents.store": ("mem", "mem"),
        "dod.parents.record_id": ("rec-1", "rec-2"),
        "dod.parents.tag": ("retrieved_in_scope", "retrieved_in_scope"),
    }
    assert bind.name == "dod.memory_bind"
    assert dict(bind.attributes) == {
        "dod.write.id": w, "dod.write.store": "mem", "dod.write.record_id": "rec-9",
    }
    observer.on_bind(BindEvent(w, "mem", "rec-9", "abc"))  # no active span: its own
    assert exporter.get_finished_spans()[-1].attributes["dod.bind.content_hash"] == "abc"
    assert delete.name == "dod.memory_delete"
    assert dict(delete.attributes) == {
        "dod.delete.store": "mem", "dod.delete.record_id": "rec-9",
        "dod.delete.reason": "forget", "dod.write.id": w,
        "dod.agent.id": "a1", "dod.session.id": "s1",
    }


@pytest.mark.otel
def test_standalone_span_without_active_span(ledger, otel):
    _, exporter, observer = otel
    w = _run(ledger, observer)
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["dod.memory_write", "dod.memory_bind", "dod.memory_delete"]
    assert spans[0].attributes["dod.write.id"] == w
    assert spans[0].attributes["dod.parents.count"] == 2
    assert all(s.parent is None for s in spans)


@pytest.mark.otel
def test_batch_write_carries_record_id(ledger, otel):
    tracer, exporter, observer = otel
    cap = Capture(ledger, observers=[observer])
    with tracer.start_as_current_span("step"):
        cap.on_batch("t", "mem", writes=[((), TEXT, "user", "rec-a")])
    (event,) = exporter.get_finished_spans()[0].events
    assert event.attributes["dod.write.record_id"] == "rec-a"
    assert "dod.agent.id" not in event.attributes


@pytest.mark.otel
def test_no_content_and_dod_namespace_only(ledger, otel):
    tracer, exporter, observer = otel
    with tracer.start_as_current_span("step"):
        _run(ledger, observer)
    _run(ledger, observer)
    attrs = list(_all_attrs(exporter.get_finished_spans()))
    keys = {k for a in attrs for k in a if a is not None}
    dod_keys = {k for k in keys if k.startswith("dod.")}
    assert len(dod_keys) > 10
    # Only the test's own span may carry non-dod attributes, and it carries none.
    assert keys == dod_keys
    assert not any(k.startswith("gen_ai.") for k in keys)
    for a in attrs:
        for value in a.values():
            values = value if isinstance(value, tuple) else (value,)
            assert not any(TEXT in str(v) or "window seats" in str(v) for v in values)


@pytest.mark.otel
def test_failing_otel_does_not_break_capture(ledger, otel, monkeypatch):
    tracer, exporter, observer = otel

    def boom(*a, **k):
        raise RuntimeError("exporter down")

    monkeypatch.setattr(observer, "_emit", boom)
    cap = Capture(ledger, observers=[observer])
    w = cap.on_write("t", TEXT, "mem", "user")
    cap.bind(w, "rec-1")
    assert ledger._conn.execute("SELECT count(*) FROM bindings").fetchone()[0] == 1


@pytest.mark.otel
def test_default_tracer_is_noop_without_sdk_config(ledger):
    pytest.importorskip("opentelemetry.trace")
    from dod.export.otel import OTelObserver

    # Global provider unset: the API is a no-op, and capture still records.
    cap = Capture(ledger, observers=[OTelObserver()])
    cap.on_write("t", TEXT, "mem", "user")
    assert ledger._conn.execute("SELECT count(*) FROM writes").fetchone()[0] == 1
