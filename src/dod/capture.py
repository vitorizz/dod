"""Framework-agnostic capture: bind in-context retrievals to each memory write.

A context key names one unit of work (for example an agent's task). The
context for a key is every item retrieved under it since the last
``reset(key)``. At write time, all of those items become the write's parents.
Nothing is filtered by relevance or recency: a missed parent breaks trace and
excision silently, while an extra edge only costs precision.

Retrievals are held in memory and never touch the ledger; only writes,
compactions, and bindings do.
"""

import threading
from typing import Dict, Hashable, Iterable, Optional, Tuple, Union

from dod.ledger import TAGS, Ledger

Item = Tuple[str, str]


class Capture:
    """Track per-key context and record each write with its parents.

    Safe to share across threads. Keys are isolated: a write under one key
    never gets parents retrieved under another.
    """

    def __init__(self, ledger: Ledger):
        self._ledger = ledger
        self._lock = threading.Lock()
        # key -> {(store, record_id): tag}; in_prompt wins over retrieved_in_scope.
        self._context: Dict[Hashable, Dict[Item, str]] = {}
        # write_id -> store, until the store assigns the record ID.
        self._pending: Dict[str, str] = {}

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
        """Bind a write to the record ID its store assigned."""
        with self._lock:
            store = self._pending.pop(write_id)
        self._ledger.bind(write_id, store, record_id)

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
        return write_id
