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

Quarantine
    Items the ledger has quarantined are dropped from ``GetOp`` (returns
    ``None``) and ``SearchOp`` results before they reach the caller or the
    context, as in the CrewAI integration (``dod.quarantine``). The pre-read
    of a put still sees them, so overwriting one is a compaction.

Excision
    ``DeleteAdapter(store)`` lets ``dod.excise`` delete items through the
    instrumented ``delete``, so each deletion is tombstoned.

Writes
    Before the store call, the instrumented store reads the keys being put
    (one extra ``batch`` with ``refresh_ttl=False``) and records every put in
    one ledger transaction, tagged ``agent_inference`` since ``put`` takes no
    source. The content hashed is the value as sorted-key JSON.

    - A put to a new key is a write, bound to its record ID after the store
      call succeeds. If the call raises, the write stays unbound.
    - A put over an existing key is a compaction whose parents include the
      overwritten item, bound the same way.
    - A delete (a put with ``None``) of an existing item records a tombstone
      (reason ``forget``, or ``excision`` when called by ``dod.excise``; actor
      the current scope's agent and task, if any) after the store call
      succeeds. A delete of a missing key deletes nothing and isn't recorded.

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
from dod.excise import deletion_reason
from dod.quarantine import QuarantineCache, count_blocked

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
    quarantine = QuarantineCache(capture._ledger)

    def record(puts: List[PutOp], found: list) -> Tuple[Any, list]:
        """Record the batch's writes; return what ``settle`` records after
        the store call: the context key, actor, pending binds and deletes."""
        key = _scope.current_key()
        agent_id, session_id = key if key is not None else (None, None)
        # A fresh key has no context, and no other write can share it.
        ctx_key = key if key is not None else object()
        writes, rids, deletes = [], [], []
        for op, old in zip(puts, found):
            rid = record_id(op.namespace, op.key)
            if op.value is not None:
                replaced = [(store_name, rid)] if old is not None else []
                writes.append((replaced, _content(op.value), _SOURCE, None))
                rids.append(rid)
            elif old is not None:
                deletes.append((rid, deletion_reason(), None))
        ids = []
        if writes:
            if key is None and not warned:
                warned.append(True)
                logger.warning("dod: put outside agent_task_scope; recorded with no parents")
            ids = capture.on_batch(
                ctx_key, store_name, writes, agent_id=agent_id, session_id=session_id
            )
        binds = list(zip(ids, rids))
        return ctx_key, agent_id, session_id, binds, deletes

    def settle(pending: tuple, ops: list, results: list) -> list:
        """Record binds and tombstones, then filter and track the reads."""
        ctx_key, agent_id, session_id, binds, deletes = pending
        if binds or deletes:
            capture.on_batch(ctx_key, store_name, binds=binds, agent_id=agent_id,
                             session_id=session_id, deletes=deletes)
        results = _drop_quarantined(quarantine.current(), store_name, ops, results)
        key = _scope.current_key()
        if key is None:
            return results
        items = []
        for op, result in zip(ops, results):
            if isinstance(op, GetOp) and result is not None:
                items.append((store_name, record_id(result.namespace, result.key)))
            elif isinstance(op, SearchOp):
                items += [(store_name, record_id(i.namespace, i.key)) for i in result or ()]
        if items:
            capture.on_retrieve(key, items)
        return results

    @functools.wraps(cls.batch)
    def batch(self, ops):
        ops, puts = _plan(ops)
        found = cls.batch(self, _reads(puts)) if puts else []
        pending = record(puts, found)
        results = cls.batch(self, ops)
        return settle(pending, ops, results)

    @functools.wraps(cls.abatch)
    async def abatch(self, ops):
        ops, puts = _plan(ops)
        found = (await cls.abatch(self, _reads(puts))) if puts else []
        pending = record(puts, found)
        results = await cls.abatch(self, ops)
        return settle(pending, ops, results)

    for fn in (batch, abatch):
        setattr(fn, _MARK, True)
    # Stores use __slots__, so their methods can't be replaced on the instance.
    # A subclass per instrumented store, with no slots of its own, keeps the
    # layout, so the instance can switch to it in place.
    store.__class__ = type(
        cls.__name__, (cls,),
        {"__slots__": (), "__module__": cls.__module__, "batch": batch, "abatch": abatch,
         "_dod_base": cls},
    )
    return store


class DeleteAdapter:
    """Delete adapter for ``dod.excise``: deletes one item through the
    instrumented ``delete``, which records its tombstone."""

    def __init__(self, store: Any) -> None:
        if not getattr(getattr(store, "batch", None), _MARK, False):
            raise TypeError("store must be instrumented with dod's langgraph.instrument")
        self.store = store

    def delete(self, record_id: str) -> bool:
        """Delete the item ``record_id`` names; return False if it wasn't there."""
        *namespace, key = json.loads(record_id)
        # Check through the uninstrumented store: not a retrieval, and it must
        # see quarantined items too.
        base = type(self.store)._dod_base
        (found,) = base.batch(self.store, [GetOp(tuple(namespace), key, refresh_ttl=False)])
        if found is None:
            return False
        self.store.delete(tuple(namespace), key)
        return True


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


def _drop_quarantined(blocked: frozenset, store_name: str, ops: list, results: list) -> list:
    """``results`` with quarantined items removed: a ``GetOp`` finds nothing,
    a ``SearchOp`` omits them. Each dropped item is counted."""
    if not blocked:
        return results
    dropped, out = [], list(results)
    for i, (op, result) in enumerate(zip(ops, results)):
        if isinstance(op, GetOp) and result is not None:
            item = (store_name, record_id(result.namespace, result.key))
            if item in blocked:
                dropped.append(item)
                out[i] = None
        elif isinstance(op, SearchOp) and result:
            kept = []
            for r in result:
                item = (store_name, record_id(r.namespace, r.key))
                if item in blocked:
                    dropped.append(item)
                else:
                    kept.append(r)
            if len(kept) != len(result):
                out[i] = kept
    if not dropped:
        return results
    count_blocked(dropped)
    return out


def _content(value: Optional[dict]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
