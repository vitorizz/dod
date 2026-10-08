"""Reconcile store contents against the ledger: find writes that bypassed capture.

Capture records only the writes that go through an instrumented call. A
direct database insert, an uninstrumented tool, a TTL sweep, or a person
editing a memory file changes a store without a ledger row, and the lineage
graph silently has a hole there: ``trace`` stops at the item, and
``blast_radius`` and ``excise`` miss what came from it. ``reconcile`` reads
each store and the ledger and reports every item on which they disagree.

Finding kinds, per item (``Finding.kind``):

- ``unrecorded``: in the store, never bound to a recorded write. It was
  written outside capture.
- ``resurrected``: in the store, but the ledger's last event for it is a
  tombstone. It was written again outside capture, or a recorded deletion
  didn't happen.
- ``missing``: live in the ledger (bound, and not tombstoned since), absent
  from the store. It was deleted outside capture: the ledger would still
  trace through it, and has no record of the deletion.
- ``modified``: in both, but the store's content hash differs from the hash
  of the last write bound to it. It was changed outside capture. Checked
  only for stores whose adapter returns content hashes comparable with the
  recorded ones (``None`` means existence only).

For ``unrecorded`` and ``resurrected`` items, ``Finding.read_into`` lists the
recorded writes that took the item as a parent while the ledger had no live
write for it (for a resurrected item, writes after its last tombstone):
the exact places where ``trace`` breaks today.

Consistency. Capture binds a write just after the store call succeeds and
tombstones a deletion just after it, so an item read mid-write can look
unrecorded or missing for a moment. Each store is read first and the ledger
after, then every finding is checked again after ``grace`` seconds, item by
item, against the store and a fresh ledger snapshot; only findings that
still hold are reported. Raise ``grace`` for writes that bind late (crewai's
``remember_many`` saves in the background).

Reconcile is read-only: it never writes to the ledger or a store, and store
adapters read around capture, so reconciling records no retrievals. It reads
each store once and the ledger's source tables once, so cost is linear in
both. Run it on a schedule (a cron job, a worker) to keep capture's
guarantee checked; nothing runs in the background on its own.

Store adapters implement ``ReconcileAdapter``; the integrations provide one
each (``dod.integrations.crewai.ReconcileAdapter`` and so on).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Protocol, Set, Tuple

from dod.ledger import Ledger

logger = logging.getLogger(__name__)

KINDS = ("unrecorded", "resurrected", "missing", "modified")

# Rows per executemany when loading a store's items.
_CHUNK = 10_000


class ReconcileAdapter(Protocol):
    def scan(self) -> Iterable[Tuple[str, Optional[str]]]:
        """Every item in the store, as ``(record_id, content_hash)``, read
        without recording retrievals. ``content_hash`` is what capture would
        record for the item's current content, or ``None`` if the adapter
        can't compute that (existence only)."""

    def get(self, record_id: str) -> Tuple[bool, Optional[str]]:
        """``(present, content_hash)`` for one item, read the same way."""

    # Optional: ``compare_bound_hash = True`` compares content only with the
    # hash recorded on the item's last binding (``Ledger.bind(content_hash=)``)
    # and skips items whose binding has none. By default the binding's hash
    # is used when present, else the bound write's content hash.


class Finding(NamedTuple):
    store: str
    record_id: str
    kind: str
    """One of ``KINDS``."""
    ledger_hash: Optional[str]
    """The recorded hash the store was compared with: the last binding's
    ``content_hash`` if it has one, else its write's content hash."""
    store_hash: Optional[str]
    """The store's current content hash, if present and known."""
    read_into: Tuple[str, ...]
    """For ``unrecorded`` and ``resurrected``: recorded writes that took the
    item as a parent while it had no live write, oldest first."""


@dataclass
class ReconcileReport:
    findings: List[Finding]
    """Every inconsistency that held after the grace period, by store then
    record ID."""
    scanned: Dict[str, int]
    """Items read from each store."""
    live: Dict[str, int]
    """Items each store should hold according to the ledger."""
    content_checked: Dict[str, bool]
    """Per store, whether content hashes were compared (``modified``)."""
    unchecked: List[str] = field(default_factory=list)
    """Stores the ledger has bindings for that weren't passed in, so weren't
    checked."""
    cleared: int = 0
    """First-pass findings that resolved within the grace period (writes or
    deletions in flight)."""
    ts: int = field(default_factory=time.time_ns)
    """When the stores were read (ns since the epoch)."""

    @property
    def ok(self) -> bool:
        """True if every store matches the ledger."""
        return not self.findings

    def by_kind(self) -> Dict[str, List[Finding]]:
        out: Dict[str, List[Finding]] = {}
        for f in self.findings:
            out.setdefault(f.kind, []).append(f)
        return out


def reconcile(
    ledger: Ledger,
    stores: Mapping[str, ReconcileAdapter],
    grace: float = 1.0,
) -> ReconcileReport:
    """Compare each store in ``stores`` (store name, as given to
    ``instrument``, to its adapter) with the ledger; see the module docs.

    Exceptions from an adapter propagate: a store that can't be read is never
    reported as matching.
    """
    if grace < 0:
        raise ValueError("grace must be >= 0")
    ts = time.time_ns()
    seen: Dict[str, Dict[str, Optional[str]]] = {}
    hashed: Dict[str, bool] = {}
    for name, adapter in stores.items():
        items: Dict[str, Optional[str]] = {}
        for record_id, content_hash in adapter.scan():
            items[record_id] = content_hash
        seen[name] = items
        hashed[name] = any(h is not None for h in items.values())

    bound_only = {n: bool(getattr(a, "compare_bound_hash", False)) for n, a in stores.items()}
    first, live, known = _check(ledger, seen, None, False, bound_only)
    report = ReconcileReport([], {n: len(i) for n, i in seen.items()}, live, hashed,
                             unchecked=sorted(known - set(stores)), ts=ts)
    if not first:
        return report

    if grace:
        time.sleep(grace)
    again: Dict[str, Dict[str, Optional[str]]] = {}
    flagged: Dict[str, List[str]] = {}
    for f in first:
        present, content_hash = stores[f.store].get(f.record_id)
        flagged.setdefault(f.store, []).append(f.record_id)
        if present:
            again.setdefault(f.store, {})[f.record_id] = content_hash
    findings, _, _ = _check(ledger, again, flagged, True, bound_only)
    report.findings = findings
    report.cleared = len(first) - len(findings)
    if findings:
        logger.warning(
            "dod: reconcile found %d item(s) changed outside capture: %s",
            len(findings),
            ", ".join(f"{k} {len(v)}" for k, v in report.by_kind().items()),
        )
    return report


# Store items, as read by an adapter.
_SEEN = (
    "CREATE TEMP TABLE IF NOT EXISTS rec_seen (store TEXT NOT NULL, record_id TEXT NOT NULL,"
    " hash TEXT, PRIMARY KEY (store, record_id)) WITHOUT ROWID"
)
# Items to classify on a re-check (all of a store's items on the first pass).
_SCOPE = (
    "CREATE TEMP TABLE IF NOT EXISTS rec_scope (store TEXT NOT NULL, record_id TEXT NOT NULL,"
    " PRIMARY KEY (store, record_id)) WITHOUT ROWID"
)
# Items found in the store with no live write, and since when a write that
# took one as a parent counts (0, or its last tombstone's ts).
_FLAG = (
    "CREATE TEMP TABLE IF NOT EXISTS rec_flag (store TEXT NOT NULL, record_id TEXT NOT NULL,"
    " since INTEGER NOT NULL, PRIMARY KEY (store, record_id)) WITHOUT ROWID"
)

# The ledger's state for one store's items: the last binding (by append
# order) with the hash to compare against (``{expected}``: the binding's own
# hash, else its write's, or for ``compare_bound_hash`` adapters the
# binding's only), and the last tombstone's ts. ``{only_b}``/``{only_t}``
# limit it to the items in rec_scope on a re-check.
_STATE = """
WITH bound AS (
    SELECT b.record_id, b.ts, {expected} AS content_hash FROM bindings b
    CROSS JOIN writes w ON w.write_id = b.write_id
    WHERE b.store = :s {only_b} AND b.rowid = (
        SELECT max(b2.rowid) FROM bindings b2
        WHERE b2.store = b.store AND b2.record_id = b.record_id
    )
),
gone AS (
    SELECT t.record_id, max(t.ts) AS ts FROM tombstones t
    WHERE t.store = :s {only_t} GROUP BY t.record_id
)"""

# Items in the store whose state disagrees: no binding, tombstoned since the
# last binding, or (where the store gave a hash) different content.
_IN_STORE = _STATE + """
SELECT r.record_id, r.hash, bound.ts, bound.content_hash, gone.ts FROM rec_seen r
LEFT JOIN bound ON bound.record_id = r.record_id
LEFT JOIN gone ON gone.record_id = r.record_id
WHERE r.store = :s AND (
    bound.ts IS NULL
    OR gone.ts >= bound.ts
    OR (r.hash IS NOT NULL AND bound.content_hash IS NOT NULL
        AND r.hash IS NOT bound.content_hash)
)"""

# Items live in the ledger: bound, and not tombstoned since.
_LIVE = _STATE + """
SELECT bound.record_id, bound.content_hash,
    NOT EXISTS (SELECT 1 FROM rec_seen r WHERE r.store = :s AND r.record_id = bound.record_id)
FROM bound LEFT JOIN gone ON gone.record_id = bound.record_id
WHERE gone.ts IS NULL OR bound.ts > gone.ts"""

# The context-set parts (their own member rows) holding a flagged item.
_PARTS = (
    "CREATE TEMP TABLE IF NOT EXISTS rec_parts (part TEXT NOT NULL, store TEXT NOT NULL,"
    " record_id TEXT NOT NULL, since INTEGER NOT NULL,"
    " PRIMARY KEY (part, store, record_id)) WITHOUT ROWID"
)
_TEMP = ("rec_seen", "rec_scope", "rec_flag", "rec_parts")
_FILL_PARTS = """
INSERT OR IGNORE INTO temp.rec_parts
SELECT m.set_hash, f.store, f.record_id, f.since FROM context_members m
CROSS JOIN rec_flag f ON f.store = m.store AND f.record_id = m.record_id"""

# Recorded writes that took a flagged item as a parent after ``since``: as a
# direct parent, or through a set whose chain includes a part holding it.
# Each source table is scanned once (CROSS JOIN fixes the order), probing
# the small temp tables by key.
_READ_INTO = """
SELECT f.store, f.record_id, w.write_id, w.seq FROM direct_parents d
CROSS JOIN rec_flag f ON f.store = d.parent_store AND f.record_id = d.parent_record_id
CROSS JOIN writes w ON w.write_id = d.write_id
WHERE w.ts > f.since
UNION
SELECT p.store, p.record_id, w.write_id, w.seq FROM context_chain c
CROSS JOIN rec_parts p ON p.part = c.part
CROSS JOIN writes w ON w.context_set = c.set_hash
WHERE w.ts > p.since
ORDER BY 1, 2, 4"""


def _check(
    ledger: Ledger,
    seen: Mapping[str, Mapping[str, Optional[str]]],
    scope: Optional[Mapping[str, List[str]]],
    reads: bool,
    bound_only: Mapping[str, bool],
) -> Tuple[List[Finding], Dict[str, int], Set[str]]:
    """Findings for the items in ``seen`` (per store, record ID to content
    hash), against one ledger snapshot, the live count per store, and every
    store the ledger has bindings for (on the first pass only).

    With ``scope`` (per store, the record IDs to re-check), only those items
    are classified: one in ``scope`` but not in ``seen`` is absent from the
    store. Without it, every item of each store in ``seen`` is. ``reads``
    fills ``Finding.read_into``, which costs a pass over the parent tables.
    ``bound_only`` names the stores compared with binding hashes only.
    """
    from dod.query import _walk_connection  # a read snapshot on its own connection

    names = list(scope) if scope is not None else list(seen)
    conn, release = _walk_connection(ledger)
    try:
        x = conn.execute
        for stmt in (_SEEN, _SCOPE, _FLAG, _PARTS):
            x(stmt)
        for t in _TEMP:
            x(f"DELETE FROM temp.{t}")
        for name in names:
            rows = [(name, r, h) for r, h in seen.get(name, {}).items()]
            for i in range(0, len(rows), _CHUNK):
                conn.executemany("INSERT INTO temp.rec_seen VALUES (?, ?, ?)",
                                 rows[i:i + _CHUNK])
        if scope is not None:
            conn.executemany("INSERT OR IGNORE INTO temp.rec_scope VALUES (?, ?)",
                             [(s, r) for s, ids in scope.items() for r in ids])
        only_b = only_t = ""
        if scope is not None:
            only_b = ("AND EXISTS (SELECT 1 FROM temp.rec_scope q"
                      " WHERE q.store = b.store AND q.record_id = b.record_id)")
            only_t = ("AND EXISTS (SELECT 1 FROM temp.rec_scope q"
                      " WHERE q.store = t.store AND q.record_id = t.record_id)")

        found: List[Tuple[str, str, str, Optional[str], Optional[str]]] = []
        flags: List[Tuple[str, str, int]] = []
        live: Dict[str, int] = {}
        for name in names:
            expected = ("b.content_hash" if bound_only.get(name)
                        else "coalesce(b.content_hash, w.content_hash)")
            fmt = dict(only_b=only_b, only_t=only_t, expected=expected)
            in_store, live_sql = _IN_STORE.format(**fmt), _LIVE.format(**fmt)
            for record_id, store_hash, bound_ts, ledger_hash, gone_ts in x(
                in_store, {"s": name}
            ):
                if bound_ts is None:
                    found.append((name, record_id, "unrecorded", None, store_hash))
                    flags.append((name, record_id, 0))
                elif gone_ts is not None and gone_ts >= bound_ts:
                    found.append((name, record_id, "resurrected", ledger_hash, store_hash))
                    flags.append((name, record_id, gone_ts))
                else:
                    found.append((name, record_id, "modified", ledger_hash, store_hash))
            n = 0
            for record_id, ledger_hash, absent in x(live_sql, {"s": name}):
                n += 1
                if absent:
                    found.append((name, record_id, "missing", ledger_hash, None))
            live[name] = n

        known: Set[str] = set()
        if scope is None:
            # Distinct stores by skipping through bindings_by_record.
            row = x("SELECT min(store) FROM bindings").fetchone()
            while row[0] is not None:
                known.add(row[0])
                row = x("SELECT min(store) FROM bindings WHERE store > ?", row).fetchone()
        read_into: Dict[Tuple[str, str], List[str]] = {}
        if flags and reads:
            conn.executemany("INSERT INTO temp.rec_flag VALUES (?, ?, ?)", flags)
            x(_FILL_PARTS)
            for s, r, write_id, _ in x(_READ_INTO):
                read_into.setdefault((s, r), []).append(write_id)
    finally:
        try:
            for t in _TEMP:
                conn.execute(f"DROP TABLE IF EXISTS temp.{t}")
        finally:
            release()
    findings = [
        Finding(s, r, kind, lh, sh, tuple(read_into.get((s, r), ())))
        for s, r, kind, lh, sh in sorted(found, key=lambda f: (f[0], f[1]))
    ]
    return findings, live, known
