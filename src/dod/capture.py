"""Framework-agnostic capture: bind in-context retrievals to each memory write.

A context key names one unit of work (for example an agent's task). The
context for a key is every item retrieved under it since the last
``reset(key)``. At write time, all of those items become the write's parents.
Nothing is filtered by relevance or recency: a missed parent breaks trace and
excision silently, while an extra edge only costs precision.

Retrievals are held in memory and never touch the ledger; only writes,
compactions, bindings, and deletions (tombstones) do. A write's context goes
to the ledger as its context set, stored once however many writes share it;
a compaction's replaced items are its direct parents.

In-memory state and its lifecycle:

- Pending writes. ``on_write``, ``on_compaction`` and unbound ``on_batch``
  writes remember each write's store until ``bind(write_id, record_id)`` (or
  an ``on_batch`` bind) succeeds; a failed ``bind`` keeps
  the entry so the caller can retry. At most ``max_pending`` writes are held;
  past that, the oldest unbound write is evicted and can no longer be bound
  through this ``Capture``. Evicted writes, and writes whose store call failed
  and are never bound, stay in the ledger with no binding.
- Context. A key's context lives until ``reset(key)``. It is deliberately not
  capped or expired: dropping items would silently drop parents. Callers must
  ``reset`` keys when a unit of work ends, or memory grows with the number of
  distinct keys and the items retrieved under them.

Observers. ``Capture(ledger, observers=[...])`` calls each observer's
``on_write(WriteEvent)``, ``on_bind(BindEvent)`` and
``on_tombstone(TombstoneEvent)`` after the ledger commit succeeds, in the
calling thread; a method an observer lacks is skipped. Events carry IDs,
hashes and tags, never content. An observer that raises is logged and
ignored: it never affects capture or the caller. With no observers, nothing
extra runs.
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import (
    Dict, Hashable, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union,
)

from dod.ledger import TAGS, Ledger

logger = logging.getLogger(__name__)

Item = Tuple[str, str]
Edge = Tuple[str, str, str]


class WriteEvent(NamedTuple):
    """A recorded write. ``parents`` are its ``(store, record_id, tag)`` edges."""

    write_id: str
    store: str
    record_id: Optional[str]
    source: str
    content_hash: str
    agent_id: Optional[str]
    session_id: Optional[str]
    parents: Tuple[Edge, ...]


class BindEvent(NamedTuple):
    """A write bound to the record ID its store assigned. ``content_hash``
    is the stored record's hash when the integration reported one."""

    write_id: str
    store: str
    record_id: str
    content_hash: Optional[str] = None


class TombstoneEvent(NamedTuple):
    """A recorded deletion of ``record_id`` from ``store``."""

    store: str
    record_id: str
    reason: str
    write_id: Optional[str]
    agent_id: Optional[str]
    session_id: Optional[str]


def _content_hash(content: Union[str, bytes]) -> str:
    """The ledger's content hash: sha256 of the UTF-8 (or raw) bytes."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


# Prompt confirmation: each retrieved item's text is kept, only in memory and
# only while its key's context lives, as a few short normalized snippets
# ("probes"). A prompt containing any probe confirms the item ``in_prompt``.
# The ledger never sees them.
# One probe per string, its first _PROBE_LEN chars: long enough to be
# specific, short enough to survive a prompt that truncates the text (any
# longer prefix that matched would imply this one does).
_PROBE_LEN = 64
_PROBE_MIN = 12   # shorter texts ("yes", an ID) would match by accident
_PROBES_MAX = 8   # per item


def _normalize(text: str) -> str:
    return " ".join(text.split())


def probes(value) -> Tuple[str, ...]:
    """Snippets of ``value`` to look for in a prompt: the start of each
    string in it (its first 64 characters) (a str, or the string leaves of a dict/list, as a store
    value), whitespace-normalized, at least ``_PROBE_MIN`` chars."""
    out: List[str] = []

    def walk(v) -> None:
        if len(out) >= _PROBES_MAX:
            return
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        if isinstance(v, str):
            # Only the start matters; normalize just enough of it.
            t = _normalize(v[:4 * _PROBE_LEN])
            if len(t) >= _PROBE_MIN:
                out.append(t[:_PROBE_LEN])
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(value)
    return tuple(dict.fromkeys(out))


def _edges(direct, context) -> Tuple[Edge, ...]:
    """A write's parent edges as the ledger's ``parents`` view has them.

    One edge per item; where context and direct parents disagree on the
    tag, ``in_prompt`` wins.
    """
    edges = {}
    for s, r, t in list(context) + list(direct):
        if edges.get((s, r)) != "in_prompt":
            edges[(s, r)] = t
    return tuple((s, r, t) for (s, r), t in edges.items())


class Capture:
    """Track per-key context and record each write with its parents.

    Safe to share across threads. Keys are isolated: a write under one key
    never gets parents retrieved under another.
    """

    def __init__(
        self, ledger: Ledger, max_pending: int = 100_000, observers: Sequence = ()
    ):
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self._ledger = ledger
        self._observers = tuple(observers)
        self._max_pending = max_pending
        self._lock = threading.Lock()
        # key -> {(store, record_id): tag}; in_prompt wins over retrieved_in_scope.
        self._context: Dict[Hashable, Dict[Item, str]] = {}
        # key -> {item: probes} for retrieved_in_scope items whose text the
        # integration passed: what confirm() looks for. Memory only.
        self._probes: Dict[Hashable, Dict[Item, Tuple[str, ...]]] = {}
        # write_id -> store, oldest first, until bind() succeeds.
        self._pending: "OrderedDict[str, str]" = OrderedDict()
        # write_ids with a bind() call in flight; a second concurrent bind raises.
        self._binding: Set[str] = set()

    def on_retrieve(
        self, key: Hashable, items: Iterable[Item], tag: str = "retrieved_in_scope",
        texts: Optional[Sequence] = None,
    ) -> None:
        """Add retrieved ``(store, record_id)`` items to ``key``'s context.

        ``texts``, parallel to ``items``, is each item's content as the
        integration read it (a str, or a store value whose string leaves
        count). It is kept in memory only, so ``confirm`` can later upgrade
        an item to ``in_prompt`` when a prompt contains it.
        """
        if tag not in TAGS:
            raise ValueError(f"unknown tag {tag!r}; expected one of {TAGS}")
        items = [(store, record_id) for store, record_id in items]
        found = None
        if texts is not None and tag != "in_prompt":
            found = [probes(t) for t in texts]
        with self._lock:
            ctx = self._context.setdefault(key, {})
            for item in items:
                if ctx.get(item) != "in_prompt":
                    ctx[item] = tag
            if found:
                pr = self._probes.setdefault(key, {})
                for item, ps in zip(items, found):
                    if ps and ctx.get(item) != "in_prompt":
                        pr[item] = ps

    def confirm(self, key: Hashable, prompt: str) -> List[Item]:
        """Upgrade ``key``'s retrieved items whose text appears in ``prompt``
        to ``in_prompt``; return them.

        Call it with the text actually sent to the model (the wrappers in
        ``dod.integrations.llm`` do). Writes after the call carry the
        upgraded tags; earlier writes keep theirs. Items never seen in a
        prompt stay ``retrieved_in_scope``: still parents, never dropped.
        """
        with self._lock:
            pr = self._probes.get(key)
            if not pr:
                return []
            candidates = list(pr.items())
        text = _normalize(prompt)
        hit = [item for item, ps in candidates if any(p in text for p in ps)]
        if hit:
            with self._lock:
                ctx = self._context.get(key)
                pr = self._probes.get(key, {})
                for item in hit:
                    pr.pop(item, None)
                    if ctx is not None and item in ctx:
                        ctx[item] = "in_prompt"
        return hit

    def on_write(
        self,
        key: Hashable,
        content: Union[str, bytes],
        store: str,
        source: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Record a write whose parents are ``key``'s current context."""
        return self._append(key, (), content, store, source, agent_id, session_id)

    def on_compaction(
        self,
        key: Hashable,
        replaced: Iterable[Item],
        content: Union[str, bytes],
        store: str,
        source: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Record a compaction as a write.

        Parents are everything in ``replaced`` (tagged ``in_prompt``) plus
        ``key``'s current context.
        """
        replaced = [(s, r, "in_prompt") for s, r in replaced]
        return self._append(key, replaced, content, store, source, agent_id, session_id)

    def on_batch(
        self,
        key: Hashable,
        store: str,
        writes: Iterable[Tuple[Iterable[Item], Union[str, bytes], str, Optional[str]]] = (),
        binds: Iterable[Tuple[str, ...]] = (),
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        deletes: Iterable[Tuple[str, str, Optional[str]]] = (),
    ) -> List[str]:
        """Record several writes, binds and deletions in one ledger transaction.

        Each of ``writes`` is ``(replaced, content, source, record_id)``: a
        compaction of ``replaced`` as with ``on_compaction`` (a plain write if
        ``replaced`` is empty), bound to ``record_id`` in the same transaction
        unless it is None. ``binds`` are ``(write_id, record_id)``, or
        ``(write_id, record_id, content_hash)``, for writes recorded earlier,
        with ``bind``'s rules. ``deletes`` are
        ``(record_id, reason, write_id)`` tombstones in ``store``, as with
        ``on_delete``. All or nothing: if the ledger call raises, nothing is
        recorded and every write in ``binds`` stays pending. Returns the new
        write IDs in order.
        """
        writes = [(list(r), c, s, rid) for r, c, s, rid in writes]
        binds = [(b[0], b[1], b[2] if len(b) > 2 else None) for b in binds]
        tombstones = [
            dict(store=store, record_id=r, reason=reason, write_id=w,
                 agent_id=agent_id, session_id=session_id)
            for r, reason, w in deletes
        ]
        with self._lock:
            ctx = self._context.get(key)
            parents = [(s, r, t) for (s, r), t in ctx.items()] if ctx else []
            ids = [w for w, _, _ in binds]
            if len(set(ids)) != len(ids) or any(w in self._binding for w in ids):
                raise KeyError(f"bind already in progress for one of {ids!r}")
            bindings = [(w, self._pending[w], r, h) for w, r, h in binds]
            self._binding.update(ids)
        rows = [
            dict(
                content=content,
                store=store,
                source=source,
                agent_id=agent_id,
                session_id=session_id,
                parents=[(s, r, "in_prompt") for s, r in replaced],
                context=parents,
                record_id=record_id,
            )
            for replaced, content, source, record_id in writes
        ]
        write_ids = None
        try:
            write_ids = self._ledger.append_batch(rows, bindings, tombstones)
        finally:
            with self._lock:
                self._binding.difference_update(ids)
                if write_ids is not None:
                    for w in ids:
                        self._pending.pop(w, None)
                    for w, (*_, record_id) in zip(write_ids, writes):
                        if record_id is None:
                            self._pending[w] = store
                    while len(self._pending) > self._max_pending:
                        self._pending.popitem(last=False)
        if self._observers:
            for w, (replaced, content, source, record_id) in zip(write_ids, writes):
                self._notify("on_write", WriteEvent(
                    w, store, record_id, source, _content_hash(content), agent_id,
                    session_id, _edges([(s, r, "in_prompt") for s, r in replaced], parents),
                ))
            for w, s, r, h in bindings:
                self._notify("on_bind", BindEvent(w, s, r, h))
            for t in tombstones:
                self._notify("on_tombstone", TombstoneEvent(**t))
        return write_ids

    def on_delete(
        self,
        store: str,
        record_ids: Iterable[str],
        reason: str,
        write_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Record a tombstone for each of ``record_ids``, deleted from ``store``.

        Call after the store deleted them. ``reason`` is one of the ledger's
        ``REASONS``; ``write_id`` is the recorded write whose save deleted
        them, if any, and ``agent_id``/``session_id`` the actor. A deletion
        has no parents, so no context key is involved.
        """
        tombstones = [
            dict(store=store, record_id=r, reason=reason, write_id=write_id,
                 agent_id=agent_id, session_id=session_id)
            for r in record_ids
        ]
        self._ledger.append_tombstones(tombstones)
        if self._observers:
            for t in tombstones:
                self._notify("on_tombstone", TombstoneEvent(**t))

    def snapshot(self, key: Hashable) -> Hashable:
        """Return a new key whose context is a copy of ``key``'s.

        For work that runs after ``key``'s scope may have ended. ``reset`` the
        returned key when done.
        """
        snap = object()
        with self._lock:
            ctx = self._context.get(key)
            if ctx:
                self._context[snap] = dict(ctx)
            pr = self._probes.get(key)
            if pr:
                self._probes[snap] = dict(pr)
        return snap

    def bind(self, write_id: str, record_id: str, content_hash: Optional[str] = None) -> None:
        """Bind a write to the record ID its store assigned. ``content_hash``
        is the stored record's hash, for integrations whose write hash isn't
        the stored content (see ``Ledger.bind``).

        Raises ``KeyError`` if the write is unknown, already bound, evicted, or
        has another ``bind`` in flight. If the ledger call raises, the write
        stays pending and can be retried.
        """
        with self._lock:
            if write_id in self._binding:
                raise KeyError(f"bind already in progress for {write_id!r}")
            store = self._pending[write_id]
            self._binding.add(write_id)
        bound = False
        try:
            self._ledger.bind(write_id, store, record_id, content_hash)
            bound = True
        finally:
            with self._lock:
                self._binding.discard(write_id)
                if bound:
                    self._pending.pop(write_id, None)
        if self._observers:
            self._notify("on_bind", BindEvent(write_id, store, record_id, content_hash))

    def reset(self, key: Hashable) -> None:
        """Clear ``key``'s context, for example when a new task starts."""
        with self._lock:
            self._context.pop(key, None)
            self._probes.pop(key, None)

    def _append(self, key, extra, content, store, source, agent_id, session_id):
        with self._lock:
            ctx = self._context.get(key)
            parents = [(s, r, t) for (s, r), t in ctx.items()] if ctx else []
        write_id = self._ledger.append_write(
            content=content,
            store=store,
            source=source,
            agent_id=agent_id,
            session_id=session_id,
            parents=extra,
            context=parents,
        )
        with self._lock:
            self._pending[write_id] = store
            while len(self._pending) > self._max_pending:
                self._pending.popitem(last=False)
        if self._observers:
            self._notify("on_write", WriteEvent(
                write_id, store, None, source, _content_hash(content), agent_id,
                session_id, _edges(extra, parents),
            ))
        return write_id

    def _notify(self, method: str, event) -> None:
        """Call ``method`` on every observer that has it; log and swallow errors."""
        for observer in self._observers:
            fn = getattr(observer, method, None)
            if fn is None:
                continue
            try:
                fn(event)
            except Exception:
                logger.exception("dod observer %r failed in %s; capture unaffected",
                                 observer, method)
