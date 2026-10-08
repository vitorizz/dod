"""OpenTelemetry export: each captured write as a ``dod.*`` span event.

``OTelObserver`` is a ``Capture`` observer (``pip install "dod[otel]"``)::

    capture = Capture(ledger, observers=[OTelObserver()])

Every recorded write adds a ``dod.memory_write`` event to the current span,
so lineage edges show up inside the traces you already collect. With no
span active, it emits a short standalone span of the same name instead.
Binds and tombstones add ``dod.memory_bind`` / ``dod.memory_delete``.

Attributes (all in the ``dod.`` namespace, never ``gen_ai.``):

- ``dod.memory_write``: ``dod.write.id``, ``dod.write.store``,
  ``dod.write.record_id`` (when bound in the same call), ``dod.write.source``,
  ``dod.write.content_hash``, ``dod.agent.id``, ``dod.session.id``,
  ``dod.parents.count``, and the parallel arrays ``dod.parents.store``,
  ``dod.parents.record_id``, ``dod.parents.tag`` (one entry per edge).
- ``dod.memory_bind``: ``dod.write.id``, ``dod.write.store``,
  ``dod.write.record_id``, and ``dod.bind.content_hash`` when the integration
  reported the stored record's hash.
- ``dod.memory_delete``: ``dod.delete.store``, ``dod.delete.record_id``,
  ``dod.delete.reason``, ``dod.write.id`` (the write that caused it, if any),
  ``dod.agent.id``, ``dod.session.id``.

Optional IDs that are None are left out (OTel attributes can't be null).
No memory content is ever attached: only IDs, hashes and tags.

No network: this module only calls the OpenTelemetry API. Whether and where
the data goes is decided entirely by your own OTel SDK and exporter
configuration; with no SDK configured, the API is a no-op.
"""

from typing import Optional

from opentelemetry import trace

from dod.capture import BindEvent, TombstoneEvent, WriteEvent

WRITE_EVENT = "dod.memory_write"
BIND_EVENT = "dod.memory_bind"
DELETE_EVENT = "dod.memory_delete"


def _put(attrs: dict, key: str, value: Optional[str]) -> None:
    if value is not None:
        attrs[key] = str(value)


class OTelObserver:
    """Report capture events to OpenTelemetry. Use as a ``Capture`` observer.

    ``tracer`` is used only for the standalone span when no span is active;
    by default it is the global provider's ``dod`` tracer.
    """

    def __init__(self, tracer: Optional[trace.Tracer] = None):
        self._tracer = tracer if tracer is not None else trace.get_tracer("dod")

    def on_write(self, event: WriteEvent) -> None:
        attrs = {
            "dod.write.id": event.write_id,
            "dod.write.store": str(event.store),
            "dod.write.source": event.source,
            "dod.write.content_hash": event.content_hash,
            "dod.parents.count": len(event.parents),
            "dod.parents.store": [str(s) for s, _, _ in event.parents],
            "dod.parents.record_id": [str(r) for _, r, _ in event.parents],
            "dod.parents.tag": [t for _, _, t in event.parents],
        }
        _put(attrs, "dod.write.record_id", event.record_id)
        _put(attrs, "dod.agent.id", event.agent_id)
        _put(attrs, "dod.session.id", event.session_id)
        self._emit(WRITE_EVENT, attrs)

    def on_bind(self, event: BindEvent) -> None:
        attrs = {
            "dod.write.id": event.write_id,
            "dod.write.store": str(event.store),
            "dod.write.record_id": str(event.record_id),
        }
        _put(attrs, "dod.bind.content_hash", event.content_hash)
        self._emit(BIND_EVENT, attrs)

    def on_tombstone(self, event: TombstoneEvent) -> None:
        attrs = {
            "dod.delete.store": str(event.store),
            "dod.delete.record_id": str(event.record_id),
            "dod.delete.reason": event.reason,
        }
        _put(attrs, "dod.write.id", event.write_id)
        _put(attrs, "dod.agent.id", event.agent_id)
        _put(attrs, "dod.session.id", event.session_id)
        self._emit(DELETE_EVENT, attrs)

    def _emit(self, name: str, attrs: dict) -> None:
        span = trace.get_current_span()
        if span.is_recording():
            span.add_event(name, attrs)
        else:
            # No active span (or one that isn't recorded): a short span of our
            # own. Under a sampled-out parent, the SDK's default sampler drops
            # it too, so sampling decisions are respected.
            self._tracer.start_span(name, attributes=attrs).end()
