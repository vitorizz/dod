"""Framework-agnostic capture: bind in-context retrievals to each memory write.

A context key names one unit of work (for example an agent's task). The
context for a key is every item retrieved under it since the last
``reset(key)``. At write time, all of those items become the write's parents.
Nothing is filtered by relevance or recency: a missed parent breaks trace and
excision silently, while an extra edge only costs precision.

Retrievals are held in memory and never touch the ledger; only writes,
compactions, and bindings do.

In-memory state and its lifecycle:

- Pending writes. ``on_write`` and ``on_compaction`` remember each write's
  store until ``bind(write_id, record_id)`` succeeds; a failed ``bind`` keeps
  the entry so the caller can retry. At most ``max_pending`` writes are held;
  past that, the oldest unbound write is evicted and can no longer be bound
  through this ``Capture``. Evicted writes, and writes whose store call failed
  and are never bound, stay in the ledger with no binding.
- Context. A key's context lives until ``reset(key)``. It is deliberately not
  capped or expired: dropping items would silently drop parents. Callers must
  ``reset`` keys when a unit of work ends, or memory grows with the number of
  distinct keys and the items retrieved under them.
"""

import threading
from collections import OrderedDict
from typing import Dict, Hashable, Iterable, Optional, Set, Tuple, Union

from dod.ledger import TAGS, Ledger

Item = Tuple[str, str]


class Capture:
    """Track per-key context and record each write with its parents.

    Safe to share across threads. Keys are isolated: a write under one key
    never gets parents retrieved under another.
    """

    def __init__(self, ledger: Ledger, max_pending: int = 100_000):
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self._ledger = ledger
        self._max_pending = max_pending
        self._lock = threading.Lock()
        # key -> {(store, record_id): tag}; in_prompt wins over retrieved_in_scope.
        self._context: Dict[Hashable, Dict[Item, str]] = {}
        # write_id -> store, oldest first, until bind() succeeds.
        self._pending: "OrderedDict[str, str]" = OrderedDict()
        # write_ids with a bind() call in flight; a second concurrent bind raises.
        self._binding: Set[str] = set()

    def on_retrieve(
        self, key: Hashable, items: Iterable[Item], tag: str = "retrieved_in_scope"
    ) -> None:
        """Add retrieved ``(store, record_id)`` items to ``key``'s context."""
        if tag not in TAGS:
            raise ValueError(f"unknown tag {tag!r}; expected one of {TAGS}")
        items = [(store, record_id) for store, record_id in items]
        with self._lock:
            ctx = self._context.setdefault(key, {})
            for item in items:
                if ctx.get(item) != "in_prompt":
                    ctx[item] = tag

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

    def bind(self, write_id: str, record_id: str) -> None:
        """Bind a write to the record ID its store assigned.

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
            self._ledger.bind(write_id, store, record_id)
            bound = True
        finally:
            with self._lock:
                self._binding.discard(write_id)
                if bound:
                    self._pending.pop(write_id, None)

    def reset(self, key: Hashable) -> None:
        """Clear ``key``'s context, for example when a new task starts."""
        with self._lock:
            self._context.pop(key, None)

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
            parents=list(extra) + parents,
        )
        with self._lock:
            self._pending[write_id] = store
            while len(self._pending) > self._max_pending:
                self._pending.popitem(last=False)
        return write_id
