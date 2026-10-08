"""LangGraph integration: capture lineage from a ``BaseStore`` (langgraph 1.2.14,
whose ``Store`` lives in langgraph-checkpoint 4.2.0).

``instrument(store, capture)`` wraps the instance's ``batch`` and ``abatch``.
Stores use ``__slots__``, so the instance is switched in place to a subclass
that overrides the two; ``isinstance`` checks and identity are unchanged.

The funnel
    In langgraph-checkpoint 4.2.0, ``BaseStore``'s ``get``, ``search``,
    ``put``, ``delete`` and ``list_namespaces`` build ops and call
    ``self.batch``; ``aget``, ``asearch``, ``aput``, ``adelete`` and
    ``alist_namespaces`` call ``self.abatch``. Both run in the caller's
    thread or coroutine, so the scope is visible. ``InMemoryStore``
    implements only ``batch``/``abatch`` and keeps those defaults, so wrapping
    the two covers every read and write path, including direct ``batch``
    calls.

    ``AsyncBatchedBaseStore`` overrides the public methods to queue ops for a
    background task, which calls ``abatch`` in that task's context rather
    than the caller's, so the scope there is empty or belongs to whoever
    created the store. ``instrument`` raises ``TypeError`` for any store that
    overrides a public method, rather than miss or cross-link parents.

Record IDs
    An item's record ID is ``json.dumps([*namespace, key])``: unambiguous
    (JSON quotes and escapes every part) and reversible
    (``*namespace, key = json.loads(record_id)``).

Scoping
    The context key is ``dod.scope``'s, shared with every integration, so a
    put gets parents retrieved from any instrumented store in the same
    ``agent_task_scope``. Outside any scope, reads are not tracked, and a
    write is recorded with zero parents and a one-time warning.

Reads
    Items returned by ``GetOp`` and ``SearchOp`` are added to the scope's
    context as ``retrieved_in_scope``: the store can't see the prompt.
    ``list_namespaces`` returns no items.

Writes
    Before the store call, the instrumented store reads the keys being put
    (one extra ``batch`` with ``refresh_ttl=False``) and records every put in
    one ledger transaction, tagged ``agent_inference`` since ``put`` takes no
    source. The content hashed is the value as sorted-key JSON.

    - A put to a new key is a write, bound to its record ID after the store
      call succeeds. If the call raises, the write stays unbound.
    - A put over an existing key is a compaction whose parents include the
      overwritten item, bound the same way.
    - A delete (a put with ``None``) of an existing item is an unbound
      compaction with the deleted item as parent. A delete of a missing key
      deletes nothing and isn't recorded.

    The read and the put aren't atomic: a key created by another writer in
    between is recorded as a plain write. When one batch puts the same key
    twice, only the last put is recorded, as the stores apply only the last.
"""

import functools
import json
import logging
from typing import Any, Iterable, List, Optional, Tuple

from langgraph.store.base import BaseStore, GetOp, PutOp, SearchOp

from dod import scope as _scope
from dod.capture import Capture

logger = logging.getLogger(__name__)

_MARK = "_dod_instrumented"
_SOURCE = "agent_inference"
# Public methods that must keep BaseStore's defaults, which route through
# batch/abatch in the caller's context.
_ROUTED = (
    "get", "search", "put", "delete", "list_namespaces",
    "aget", "asearch", "aput", "adelete", "alist_namespaces",
)


def record_id(namespace: Iterable[str], key: str) -> str:
    """Return the record ID for the item at ``namespace``/``key``."""
    return json.dumps([*namespace, key])


def instrument(store: Any, capture: Capture, store_name: str = "langgraph") -> Any:
    """Wrap ``store``'s ``batch`` and ``abatch`` in place.

    Signatures, return values and exceptions are unchanged. Instrumenting an
    already instrumented store is a no-op.
    """
    if getattr(getattr(store, "batch", None), _MARK, False):
        return store
    if not isinstance(store, BaseStore):
        raise TypeError(f"{type(store).__name__} is not a langgraph BaseStore")
    overridden = [n for n in _ROUTED if getattr(type(store), n) is not getattr(BaseStore, n)]
    if overridden:
        raise TypeError(
            f"{type(store).__name__} overrides {', '.join(overridden)}; only stores "
            "whose public methods route through batch/abatch can be captured"
        )
    _scope.register(capture)
    cls = type(store)
    warned = []

    def record(puts: List[PutOp], found: list) -> Tuple[Any, list]:
        """Record the batch's puts; return the context key and pending binds."""
        key = _scope.current_key()
        if key is not None:
            ctx_key, agent_id, session_id = key, key[0], key[1]
        else:
            if not warned:
                warned.append(True)
                logger.warning("dod: put outside agent_task_scope; recorded with no parents")
            # A fresh key has no context, and no other write can share it.
            ctx_key, agent_id, session_id = object(), None, None
        writes, rids = [], []
        for op, old in zip(puts, found):
            rid = record_id(op.namespace, op.key)
            replaced = [(store_name, rid)] if old is not None else []
            if op.value is not None:
                writes.append((replaced, _content(op.value), _SOURCE, None))
                rids.append(rid)
            elif old is not None:
                # Nothing survives a deletion, so the compaction stays unbound.
                writes.append((replaced, "", _SOURCE, None))
                rids.append(None)
        if not writes:
            return ctx_key, []
        ids = capture.on_batch(
            ctx_key, store_name, writes, agent_id=agent_id, session_id=session_id
        )
        return ctx_key, [(w, r) for w, r in zip(ids, rids) if r is not None]

    def settle(ctx_key: Any, binds: list, ops: list, results: list) -> None:
        if binds:
            capture.on_batch(ctx_key, store_name, binds=binds)
        key = _scope.current_key()
        if key is None:
            return
        items = []
        for op, result in zip(ops, results):
            if isinstance(op, GetOp) and result is not None:
                items.append((store_name, record_id(result.namespace, result.key)))
            elif isinstance(op, SearchOp):
                items += [(store_name, record_id(i.namespace, i.key)) for i in result or ()]
        if items:
            capture.on_retrieve(key, items)

    @functools.wraps(cls.batch)
    def batch(self, ops):
        ops, puts = _plan(ops)
        found = cls.batch(self, _reads(puts)) if puts else []
        ctx_key, binds = record(puts, found)
        results = cls.batch(self, ops)
        settle(ctx_key, binds, ops, results)
        return results

    @functools.wraps(cls.abatch)
    async def abatch(self, ops):
        ops, puts = _plan(ops)
        found = (await cls.abatch(self, _reads(puts))) if puts else []
        ctx_key, binds = record(puts, found)
        results = await cls.abatch(self, ops)
        settle(ctx_key, binds, ops, results)
        return results

    for fn in (batch, abatch):
        setattr(fn, _MARK, True)
    # Stores use __slots__, so their methods can't be replaced on the instance.
    # A subclass per instrumented store, with no slots of its own, keeps the
    # layout, so the instance can switch to it in place.
    store.__class__ = type(
        cls.__name__, (cls,),
        {"__slots__": (), "__module__": cls.__module__, "batch": batch, "abatch": abatch},
    )
    return store


def _plan(ops: Iterable[Any]) -> Tuple[list, List[PutOp]]:
    """Materialize ``ops`` and pick the put that takes effect for each key."""
    ops = list(ops)
    puts = {}
    for op in ops:
        if isinstance(op, PutOp):
            puts[(tuple(op.namespace), op.key)] = op
    return ops, list(puts.values())


def _reads(puts: List[PutOp]) -> List[GetOp]:
    # Not a retrieval: nothing read here reaches the model.
    return [GetOp(op.namespace, op.key, refresh_ttl=False) for op in puts]


def _content(value: Optional[dict]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
