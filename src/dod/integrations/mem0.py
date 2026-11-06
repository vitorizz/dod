"""Mem0 integration: capture lineage from a ``mem0.Memory`` (mem0ai 2.0.5).

``instrument(memory, capture)`` wraps the instance's methods in place. The
record ID of a memory is its Mem0 ID.

Reads
    ``search``, ``get`` and ``get_all`` results join the scope's context
    (``retrieved_in_scope``, with their text kept for ``dod.prompt.confirm``).
    Quarantined memories are dropped first (``get`` returns None).

Writes: the memories Mem0 extracts in ``add``
    With ``infer=True`` (the default), Mem0 searches the existing memories
    and puts the matches in its extraction prompt. Those matches are parents
    of every memory the call adds, tagged ``in_prompt``, along with whatever
    the scope already retrieved. Each returned ``ADD`` is a write
    (``agent_inference``) bound to its ID in one transaction after the call.
    With ``infer=False`` the messages are stored as they are: a message
    whose role is ``user`` is recorded as source ``user``, any other as
    ``agent_inference``. A call outside ``agent_task_scope`` gets its own
    scope, so Mem0's internal retrieval is still recorded.

    Every rewrite of an existing memory (``update(id, data)``, or an
    ``UPDATE`` Mem0 makes inside ``add``) goes through ``_update_memory``,
    which is wrapped: a compaction of the memory, with the new text as its
    content. Inside ``add``, its parents are the add's context.

Deletes
    ``delete`` and ``delete_all`` record a tombstone per memory removed
    (reason ``forget``, or ``excision`` under ``dod.excise``); both go
    through Mem0's ``_delete_memory``, which is wrapped. ``reset()`` is not
    recorded.

Only the synchronous ``Memory`` is supported.
"""

import contextvars
import functools
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple

from mem0 import Memory

from dod import scope as _scope
from dod.capture import Capture
from dod.capture import _content_hash as _hash
from dod.excise import deletion_reason
from dod.quarantine import QuarantineCache, count_blocked

logger = logging.getLogger(__name__)

_MARK = "_dod_instrumented"
# The internal search results of the add() in progress (a list), else None.
_internal: contextvars.ContextVar = contextvars.ContextVar("dod_mem0_internal", default=None)
# The scope key of the add() in progress, for rewrites Mem0 makes inside it.
_add_key: contextvars.ContextVar = contextvars.ContextVar("dod_mem0_add_key", default=None)


def _results(value: Any) -> List[dict]:
    if isinstance(value, dict):
        value = value.get("results", [])
    return [r for r in (value or []) if isinstance(r, dict)]


def instrument(memory: Any, capture: Capture, store: str = "mem0") -> Any:
    """Wrap ``memory``'s methods in place; return it. Return values and
    exceptions are unchanged. Instrumenting twice is a no-op."""
    if getattr(memory, _MARK, False):
        return memory
    if not isinstance(memory, Memory):
        raise TypeError(f"{type(memory).__name__} is not a mem0 Memory")
    _scope.register(capture)
    quarantine = QuarantineCache(capture._ledger)
    base = {n: getattr(memory, n) for n in ("add", "search", "get", "get_all",
                                             "_update_memory", "_delete_memory")}
    vs_search = memory.vector_store.search

    def retrieved(key, rows: Iterable[Tuple[str, str]]) -> None:
        rows = [(i, t) for i, t in rows if i is not None]
        if key is not None and rows:
            capture.on_retrieve(key, [(store, i) for i, _ in rows],
                                texts=[t or "" for _, t in rows])

    def drop_blocked(rows: List[dict]) -> List[dict]:
        blocked = quarantine.current()
        if not blocked:
            return rows
        kept = [r for r in rows if (store, r.get("id")) not in blocked]
        if len(kept) != len(rows):
            count_blocked([(store, r.get("id")) for r in rows if (store, r.get("id")) in blocked])
        return kept

    # --- reads ---------------------------------------------------------------

    def listing(name: str):
        fn = base[name]

        @functools.wraps(fn)
        def read(*args, **kwargs):
            out = fn(*args, **kwargs)
            if isinstance(out, dict) and isinstance(out.get("results"), list):
                out = dict(out, results=drop_blocked(_results(out)))
                retrieved(_scope.current_key(),
                          [(r.get("id"), r.get("memory")) for r in out["results"]])
            return out

        return read

    @functools.wraps(base["get"])
    def get(memory_id, *args, **kwargs):
        out = base["get"](memory_id, *args, **kwargs)
        if out is None:
            return None
        if (store, out.get("id", memory_id)) in quarantine.current():
            count_blocked([(store, out.get("id", memory_id))])
            return None
        retrieved(_scope.current_key(), [(out.get("id", memory_id), out.get("memory"))])
        return out

    # Mem0's own retrieval inside add(): collected, not a read by the caller.
    @functools.wraps(vs_search)
    def vector_search(*args, **kwargs):
        out = vs_search(*args, **kwargs)
        found = _internal.get()
        if found is not None:
            for r in out or ():
                payload = getattr(r, "payload", None) or {}
                found.append((getattr(r, "id", None), payload.get("data", "")))
        return out

    # --- writes --------------------------------------------------------------

    @functools.wraps(base["add"])
    def add(messages, *args, **kwargs):
        key = _scope.current_key()
        own = key is None
        if own:
            key = ("mem0", f"add-{uuid.uuid4().hex}")
        found: list = []
        token, ktoken = _internal.set(found), _add_key.set((key, own, found))
        try:
            out = base["add"](messages, *args, **kwargs)
        finally:
            _internal.reset(token)
            _add_key.reset(ktoken)
        try:
            added = [r for r in _results(out) if r.get("event", "ADD") == "ADD" and r.get("id")]
            if found:
                # Everything Mem0 retrieved went into its extraction prompt.
                retrieved(key, found)
                capture.confirm(key, "\n".join(t for _, t in found if t))
            if added:
                infer = kwargs.get("infer", True)
                writes = [([], r.get("memory") or "",
                           ("user" if r.get("role") == "user" else "agent_inference")
                           if not infer else "agent_inference", r["id"]) for r in added]
                agent_id, session_id = key if not own else (None, None)
                capture.on_batch(key, store, writes, agent_id=agent_id, session_id=session_id)
        except Exception:
            logger.exception("dod: recording mem0 add failed after the store call succeeded;"
                             " lineage for these memories is incomplete")
        finally:
            if own:
                capture.reset(key)
        return out

    # Every rewrite of an existing memory goes through _update_memory: the
    # public update(), and any UPDATE Mem0 decides on inside add().
    @functools.wraps(base["_update_memory"])
    def update_memory(memory_id, data, *args, **kwargs):
        out = base["_update_memory"](memory_id, data, *args, **kwargs)
        in_add = _add_key.get()
        if in_add is not None:
            key, own, found = in_add
            if found:  # the matches Mem0 retrieved for this add's prompt
                retrieved(key, found)
                capture.confirm(key, "\n".join(t for _, t in found if t))
            agent_id, session_id = (None, None) if own else key
        else:
            key = _scope.current_key()
            if key is None:
                logger.warning("dod: mem0 update outside agent_task_scope;"
                               " recorded with no parents")
            agent_id, session_id = key if key is not None else (None, None)
            key = key if key is not None else object()
        try:
            capture.on_batch(key, store,
                             [([(store, memory_id)], data, "agent_inference", memory_id)],
                             agent_id=agent_id, session_id=session_id)
        except Exception:
            logger.exception("dod: recording mem0 update of %r failed", memory_id)
        return out

    @functools.wraps(base["_delete_memory"])
    def delete_memory(memory_id, *args, **kwargs):
        out = base["_delete_memory"](memory_id, *args, **kwargs)
        key = _scope.current_key()
        agent_id, session_id = key if key is not None else (None, None)
        try:
            capture.on_delete(store, [memory_id], deletion_reason(), agent_id=agent_id,
                              session_id=session_id)
        except Exception:
            logger.exception("dod: recording the deletion of mem0 memory %r failed", memory_id)
        return out

    memory.add, memory.get, memory._update_memory = add, get, update_memory
    memory.search, memory.get_all = listing("search"), listing("get_all")
    memory._delete_memory = delete_memory
    memory.vector_store.search = vector_search
    memory._dod_base = base
    setattr(memory, _MARK, True)
    return memory


def _text(memory: Any, memory_id: str) -> Optional[str]:
    found = memory.vector_store.get(vector_id=memory_id)
    return None if found is None else (found.payload or {}).get("data", "")


class DeleteAdapter:
    """Delete adapter for ``dod.excise``: deletes through the instrumented
    ``delete``, which records the tombstone."""

    def __init__(self, memory: Any) -> None:
        if not getattr(memory, _MARK, False):
            raise TypeError("memory must be instrumented with dod's mem0.instrument")
        self.memory = memory

    def delete(self, record_id: str) -> bool:
        if _text(self.memory, record_id) is None:
            return False
        self.memory.delete(record_id)
        return True


class ReconcileAdapter:
    """Store adapter for ``dod.reconcile``: reads Mem0's vector store
    directly (nothing recorded, quarantined memories included). ``filters``
    narrows the scan (e.g. ``{"user_id": "u1"}``); ``limit`` caps it."""

    def __init__(self, memory: Any, filters: Optional[dict] = None, limit: int = 100_000) -> None:
        if not isinstance(memory, Memory):
            raise TypeError("memory must be a mem0 Memory")
        self.memory, self.filters, self.limit = memory, filters, limit

    def scan(self) -> List[Tuple[str, str]]:
        rows = self.memory.vector_store.list(filters=self.filters, top_k=self.limit)
        if rows and isinstance(rows[0], list):  # some backends wrap the list
            rows = rows[0]
        return [(str(r.id), _hash((r.payload or {}).get("data", ""))) for r in rows]

    def get(self, record_id: str) -> Tuple[bool, Optional[str]]:
        text = _text(self.memory, record_id)
        return (False, None) if text is None else (True, _hash(text))
