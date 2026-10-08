"""Read-side quarantine filtering shared by every integration.

An instrumented store drops results whose ``(store, record_id)`` the ledger
has quarantined, before they reach the caller or the context, so they never
become parents. Each dropped result is counted per item in memory; read the
counts with ``blocked_retrievals()``.
"""

import threading
from collections import Counter
from typing import Dict, Iterable

from dod.ledger import Item, Ledger

# (store, record_id) -> retrieval results dropped because it was quarantined.
_blocked: "Counter[Item]" = Counter()
_blocked_lock = threading.Lock()


class QuarantineCache:
    """A ledger's quarantined set, updated when its quarantine ``seq`` moves.

    Each call reads only the actions after the last ``seq`` applied (an
    index seek; usually none), so a change costs its own rows, not a re-read
    of every quarantined item. It sees quarantines from any connection.
    """

    __slots__ = ("ledger", "seq", "items", "lock")

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.seq = 0
        self.items: frozenset = frozenset()
        self.lock = threading.Lock()

    def current(self) -> frozenset:
        with self.lock:
            rows = self.ledger.quarantine_log(self.seq)
            if rows:
                items = set(self.items)
                for _seq, store, record_id, action in rows:
                    if action == "quarantine":
                        items.add((store, record_id))
                    else:
                        items.discard((store, record_id))
                self.items = frozenset(items)
                self.seq = rows[-1][0]
            return self.items


def count_blocked(items: Iterable[Item]) -> None:
    """Count one dropped retrieval result per item."""
    with _blocked_lock:
        _blocked.update(items)


def blocked_retrievals() -> Dict[Item, int]:
    """Retrieval results dropped by quarantine since process start, per item,
    across every instrumented store."""
    with _blocked_lock:
        return dict(_blocked)
