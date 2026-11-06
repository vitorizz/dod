"""LangChain integration: capture lineage from any ``VectorStore``
(langchain-core 1.7.5).

``instrument(vectorstore, capture)`` wraps the instance's public read, write
and delete methods in place (instance attributes, so the store's own
internal ``self.`` calls go through them too). Only the outermost call is
recorded: ``add_texts`` that calls ``add_documents``, or a
``similarity_search`` that calls ``similarity_search_with_score``, records
once. Async variants are wrapped the same way; LangChain's defaults run the
sync method in an executor with the caller's context, so the scope is kept.

Record IDs
    A document's record ID is its ``Document.id`` (the IDs ``add_*``
    returns). Documents returned without an ID can't be tracked: they're
    skipped, with a one-time warning.

Reads
    Documents returned by ``similarity_search*``, ``max_marginal_relevance_search*``,
    ``search``, ``get_by_ids`` (and their ``a*`` versions, and so any
    ``as_retriever()`` retriever) join the scope's context as
    ``retrieved_in_scope``, with their ``page_content`` kept for
    ``dod.prompt.confirm``. Quarantined documents are dropped from results
    first, so a search can return fewer than ``k``.

Writes
    ``add_documents``/``add_texts`` record one write per document before
    the store call (content: ``page_content`` and ``metadata`` as sorted-key
    JSON; source ``agent_inference`` unless ``instrument(source=...)``),
    bound to the returned IDs after it. A document whose ID is already in
    the store is a compaction of it.

Deletes
    ``delete(ids)`` tombstones the IDs that existed (reason ``forget``, or
    ``excision`` under ``dod.excise``) after the store call. ``delete()``
    with no IDs can't be attributed and isn't recorded (warned once).

Excision and reconciliation
    ``DeleteAdapter(vs)`` deletes through the instrumented ``delete``.
    ``ReconcileAdapter(vs, list_ids=None)`` needs a way to list IDs: pass
    ``list_ids``, or use a store with a dict ``store`` attribute
    (``InMemoryVectorStore``).
"""

import contextvars
import functools
import inspect
import json
import logging
from typing import Any, Callable, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from dod import scope as _scope
from dod.capture import Capture
from dod.capture import _content_hash as _hash
from dod.excise import deletion_reason
from dod.quarantine import QuarantineCache, count_blocked

logger = logging.getLogger(__name__)

_MARK = "_dod_instrumented"
_READS = (
    "similarity_search", "similarity_search_with_score", "similarity_search_by_vector",
    "similarity_search_with_relevance_scores", "max_marginal_relevance_search",
    "max_marginal_relevance_search_by_vector", "search", "get_by_ids",
)
_WRITES = ("add_documents", "add_texts")
# Set while an instrumented call runs, so nested calls aren't recorded again.
_active: contextvars.ContextVar = contextvars.ContextVar("dod_langchain_active", default=None)


def _content(doc: Document) -> str:
    return json.dumps({"page_content": doc.page_content, "metadata": doc.metadata or {}},
                      sort_keys=True, separators=(",", ":"), default=str)


def _doc(entry: Any) -> Document:
    return entry[0] if isinstance(entry, tuple) else entry


def instrument(vectorstore: Any, capture: Capture, store: str = "langchain",
               source: str = "agent_inference") -> Any:
    """Wrap ``vectorstore``'s methods in place; return it. Signatures,
    return values and exceptions are unchanged. Instrumenting twice is a
    no-op."""
    if getattr(vectorstore, _MARK, False):
        return vectorstore
    if not isinstance(vectorstore, VectorStore):
        raise TypeError(f"{type(vectorstore).__name__} is not a langchain VectorStore")
    _scope.register(capture)
    quarantine = QuarantineCache(capture._ledger)
    base = {n: getattr(vectorstore, n) for n in (*_READS, *_WRITES, "delete")}
    base.update({"a" + n: getattr(vectorstore, "a" + n)
                 for n in (*_READS, *_WRITES, "delete")})
    warned = set()

    def warn(what: str, msg: str) -> None:
        if what not in warned:
            warned.add(what)
            logger.warning("dod: %s", msg)

    def existing(ids: Iterable[str]) -> set:
        ids = [i for i in ids if i is not None]
        if not ids:
            return set()
        token = _active.set(True)  # not a retrieval
        try:
            return {d.id for d in base["get_by_ids"](ids)}
        finally:
            _active.reset(token)

    # --- reads ---------------------------------------------------------------

    def settle_read(result: Any) -> Any:
        blocked = quarantine.current()
        entries = list(result)
        if blocked:
            kept = [e for e in entries if (store, _doc(e).id) not in blocked]
            if len(kept) != len(entries):
                count_blocked([(store, _doc(e).id) for e in entries
                               if (store, _doc(e).id) in blocked])
                entries = kept
                result = kept
        key = _scope.current_key()
        if key is not None and entries:
            items, texts = [], []
            for e in entries:
                d = _doc(e)
                if d.id is None:
                    warn("noid", "a vector store returned a document without an id;"
                                 " it isn't tracked as a parent")
                    continue
                items.append((store, d.id))
                texts.append(d.page_content)
            if items:
                capture.on_retrieve(key, items, texts=texts)
        return result

    def wrap_read(name: str) -> None:
        sync, asyn = base[name], base["a" + name]

        @functools.wraps(sync)
        def read(*args, **kwargs):
            if _active.get():
                return sync(*args, **kwargs)
            token = _active.set(True)
            try:
                result = sync(*args, **kwargs)
            finally:
                _active.reset(token)
            return settle_read(result)

        @functools.wraps(asyn)
        async def aread(*args, **kwargs):
            if _active.get():
                return await asyn(*args, **kwargs)
            token = _active.set(True)
            try:
                result = await asyn(*args, **kwargs)
            finally:
                _active.reset(token)
            return settle_read(result)

        setattr(vectorstore, name, read)
        setattr(vectorstore, "a" + name, aread)

    # --- writes --------------------------------------------------------------

    def plan_write(name: str, args: tuple, kwargs: dict) -> Optional[tuple]:
        bound = inspect.signature(base[name]).bind_partial(*args, **kwargs)
        a = bound.arguments
        extra = a.get("kwargs", {})
        ids = a.get("ids", extra.get("ids"))
        if name == "add_documents":
            docs = list(a.get("documents", ()))
            ids = list(ids) if ids is not None else [d.id for d in docs]
        else:
            texts = list(a.get("texts", ()))
            metas = a.get("metadatas", extra.get("metadatas")) or [{}] * len(texts)
            docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metas)]
            ids = list(ids) if ids is not None else [None] * len(docs)
        if not docs:
            return None
        key = _scope.current_key()
        if key is None:
            warn("noscope", "vector store write outside agent_task_scope; recorded with no parents")
        ctx_key = key if key is not None else object()
        agent_id, session_id = key if key is not None else (None, None)
        old = existing(ids)
        writes = [([(store, i)] if i in old else [], _content(d), source, None)
                  for d, i in zip(docs, ids)]
        wids = capture.on_batch(ctx_key, store, writes, agent_id=agent_id,
                                session_id=session_id)
        return ctx_key, agent_id, session_id, wids

    def settle_write(pending: Optional[tuple], returned: List[str]) -> None:
        if pending is None:
            return
        ctx_key, agent_id, session_id, wids = pending
        try:
            capture.on_batch(ctx_key, store, binds=list(zip(wids, returned)),
                             agent_id=agent_id, session_id=session_id)
        except Exception:
            logger.exception("dod: binding %d write(s) to store %r failed after the store"
                             " call succeeded; lineage for these items is incomplete",
                             len(wids), store)

    def wrap_write(name: str) -> None:
        sync, asyn = base[name], base["a" + name]

        @functools.wraps(sync)
        def write(*args, **kwargs):
            if _active.get():
                return sync(*args, **kwargs)
            pending = plan_write(name, args, kwargs)
            token = _active.set(True)
            try:
                returned = sync(*args, **kwargs)
            finally:
                _active.reset(token)
            settle_write(pending, returned)
            return returned

        @functools.wraps(asyn)
        async def awrite(*args, **kwargs):
            if _active.get():
                return await asyn(*args, **kwargs)
            pending = plan_write(name, args, kwargs)
            token = _active.set(True)
            try:
                returned = await asyn(*args, **kwargs)
            finally:
                _active.reset(token)
            settle_write(pending, returned)
            return returned

        setattr(vectorstore, name, write)
        setattr(vectorstore, "a" + name, awrite)

    # --- deletes -------------------------------------------------------------

    def plan_delete(args: tuple, kwargs: dict) -> Optional[list]:
        ids = kwargs.get("ids", args[0] if args else None)
        if ids is None:
            warn("deleteall", "delete() without ids isn't recorded")
            return None
        found = existing(list(ids))
        return [i for i in ids if i in found]

    def settle_delete(gone: Optional[list]) -> None:
        if not gone:
            return
        key = _scope.current_key()
        agent_id, session_id = key if key is not None else (None, None)
        try:
            capture.on_delete(store, gone, deletion_reason(), agent_id=agent_id,
                              session_id=session_id)
        except Exception:
            logger.exception("dod: recording %d tombstone(s) for store %r failed after the"
                             " store call succeeded", len(gone), store)

    sync_delete, async_delete = base["delete"], base["adelete"]

    @functools.wraps(sync_delete)
    def delete(*args, **kwargs):
        if _active.get():
            return sync_delete(*args, **kwargs)
        gone = plan_delete(args, kwargs)
        token = _active.set(True)
        try:
            result = sync_delete(*args, **kwargs)
        finally:
            _active.reset(token)
        if result is not False:
            settle_delete(gone)
        return result

    @functools.wraps(async_delete)
    async def adelete(*args, **kwargs):
        if _active.get():
            return await async_delete(*args, **kwargs)
        gone = plan_delete(args, kwargs)
        token = _active.set(True)
        try:
            result = await async_delete(*args, **kwargs)
        finally:
            _active.reset(token)
        if result is not False:
            settle_delete(gone)
        return result

    for n in _READS:
        wrap_read(n)
    for n in _WRITES:
        wrap_write(n)
    vectorstore.delete, vectorstore.adelete = delete, adelete
    vectorstore._dod_base = base
    setattr(vectorstore, _MARK, True)
    return vectorstore


class DeleteAdapter:
    """Delete adapter for ``dod.excise``: deletes through the instrumented
    ``delete``, which records the tombstone."""

    def __init__(self, vectorstore: Any) -> None:
        if not getattr(vectorstore, _MARK, False):
            raise TypeError("vectorstore must be instrumented with dod's langchain.instrument")
        self.vectorstore = vectorstore

    def delete(self, record_id: str) -> bool:
        if not self.vectorstore._dod_base["get_by_ids"]([record_id]):
            return False
        self.vectorstore.delete([record_id])
        return True


class ReconcileAdapter:
    """Store adapter for ``dod.reconcile``. Reads through the uninstrumented
    methods, so nothing is recorded as a retrieval and quarantined documents
    are seen. ``list_ids`` lists every ID in the store; without it the store
    needs a dict ``store`` attribute (``InMemoryVectorStore``)."""

    def __init__(self, vectorstore: Any,
                 list_ids: Optional[Callable[[], Iterable[str]]] = None) -> None:
        if not isinstance(vectorstore, VectorStore):
            raise TypeError("vectorstore must be a langchain VectorStore")
        if list_ids is None:
            if not isinstance(getattr(vectorstore, "store", None), dict):
                raise TypeError(f"{type(vectorstore).__name__} can't list its ids;"
                                " pass list_ids=")
            list_ids = lambda: list(vectorstore.store)  # noqa: E731
        self.vectorstore = vectorstore
        self.list_ids = list_ids

    def _get(self, ids: List[str]) -> List[Document]:
        fn = getattr(self.vectorstore, "_dod_base", {}).get("get_by_ids") \
            or self.vectorstore.get_by_ids
        return fn(ids)

    def scan(self) -> List[Tuple[str, str]]:
        ids = list(self.list_ids())
        out = []
        for i in range(0, len(ids), 500):
            out += [(d.id, _hash(_content(d))) for d in self._get(ids[i:i + 500])]
        return out

    def get(self, record_id: str) -> Tuple[bool, Optional[str]]:
        found = self._get([record_id])
        return (True, _hash(_content(found[0]))) if found else (False, None)
