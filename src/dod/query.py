"""Read-only lineage queries over the ledger.

A parent edge names a memory item ``(store, record_id)``; ``bindings`` map it
to the write(s) that produced it, so walks cross stores. An item bound to
several writes resolves to all of them (over-include rather than miss a
parent). ``trace`` reads edges from the ledger's ``parents`` view, which
expands each write's context set and direct parents into one row per edge, so
queries never see how they are stored. ``blast_radius`` walks forward over the
ledger's integer lineage index plus the not-yet-indexed tail.
"""

import gc
import json
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from .ledger import TAGS, Ledger

# Stay under SQLITE_MAX_VARIABLE_NUMBER on old builds (999).
_CHUNK = 900

_WRITE_COLS = "seq, write_id, content_hash, ts, store, agent_id, session_id, source"


# TracedWrite and Edge are NamedTuples, not frozen dataclasses: a blast radius
# can build tens of thousands of them, and frozen-dataclass construction was
# ~8x slower and dominated the 10k-write benchmark.
class TracedWrite(NamedTuple):
    """A write reached by a walk. ``depth`` 0 is the queried item itself."""

    write_id: str
    depth: int
    seq: int
    content_hash: str
    ts: int
    store: str
    agent_id: Optional[str]
    session_id: Optional[str]
    source: str


class Edge(NamedTuple):
    """A parent edge walked from ``child_write_id``.

    ``parent_write_id`` is ``None`` when the parent item has no binding.
    """

    child_write_id: str
    parent_store: str
    parent_record_id: str
    tag: str
    parent_write_id: Optional[str]


class Tombstone(NamedTuple):
    """A recorded deletion of ``(store, record_id)`` from its store."""

    seq: int
    store: str
    record_id: str
    ts: int
    reason: str
    write_id: Optional[str]
    """The write whose save deleted the item, if any."""
    agent_id: Optional[str]
    session_id: Optional[str]


# Build a NamedTuple from a row in C, skipping its generated Python __new__.
_traced_write = partial(tuple.__new__, TracedWrite)
_edge = partial(tuple.__new__, Edge)

Item = Tuple[str, str]


@dataclass(frozen=True)
class UnrecordedAncestor:
    """A parent item that was in context but has no binding in the ledger."""

    store: str
    record_id: str
    depth: int


@dataclass
class TraceResult:
    store: str
    record_id: str
    items: List[TracedWrite]
    """Writes bound to the traced item (depth 0)."""
    ancestors: List[TracedWrite]
    """Ancestor writes, each once at its minimum depth, ordered by (depth, seq)."""
    edges: List[Edge]
    roots: List[TracedWrite]
    """Reached writes (including depth 0) with no recorded parents."""
    unrecorded: List[UnrecordedAncestor] = field(default_factory=list)
    truncated: bool = False
    """True if ``max_depth`` stopped the walk before every ancestor was reached."""
    deleted: Dict[Item, List[Tombstone]] = field(default_factory=dict)
    """Tombstones, ordered by seq, of the traced item and every parent item
    walked, keyed by ``(store, record_id)``. Items never deleted are absent."""

    def edges_tagged(self, tag: str) -> List[Edge]:
        return _edges_tagged(self.edges, tag)

    def is_deleted(self, store: str, record_id: str) -> bool:
        return (store, record_id) in self.deleted


class BlastResult:
    """Everything derived from an item: what ``blast_radius`` reached.

    ``edges`` is built on first access, since a large blast radius has more
    edges than writes and most callers only need the writes and their items.
    It is exactly the edges the walk followed: the ledger is append-only, so
    the walk records how far it read and the edges are read up to there,
    whatever has been appended since.
    """

    def __init__(
        self,
        store: str,
        record_id: str,
        items: List[TracedWrite],
        descendants: List[TracedWrite],
        edges: Optional[List[Edge]] = None,
        bindings: Optional[Dict[str, List[Tuple[str, str]]]] = None,
        store_counts: Optional[Dict[str, int]] = None,
        truncated: bool = False,
        deleted: Optional[Dict[Item, List[Tombstone]]] = None,
        *,
        load_edges: Optional[Callable[[], List[Edge]]] = None,
    ):
        if (edges is None) == (load_edges is None):
            raise ValueError("pass exactly one of edges and load_edges")
        self.store = store
        self.record_id = record_id
        self.items = items
        """Writes bound to the queried item (depth 0)."""
        self.descendants = descendants
        """Descendant writes, each once at its minimum depth, ordered by (depth, seq)."""
        self.bindings = bindings if bindings is not None else {}
        """``(store, record_id)`` items bound to each reached write, depth 0 included."""
        self.store_counts = store_counts if store_counts is not None else {}
        """Number of descendant writes per store."""
        self.truncated = truncated
        """True if ``max_depth`` stopped the walk before every descendant was reached."""
        self.deleted = deleted if deleted is not None else {}
        """Tombstones, ordered by seq, of the queried item and every item bound
        to a reached write, keyed by ``(store, record_id)``. Items never deleted
        are absent."""
        self._edges = edges
        self._load_edges = load_edges
        self._edges_lock = threading.Lock()

    @property
    def edges(self) -> List[Edge]:
        """Parent edges walked forward; ``parent_write_id`` is the reached write."""
        if self._edges is None:
            with self._edges_lock:
                if self._edges is None:
                    self._edges = self._load_edges()
                    self._load_edges = None
        return self._edges

    def edges_tagged(self, tag: str) -> List[Edge]:
        return _edges_tagged(self.edges, tag)

    def is_deleted(self, store: str, record_id: str) -> bool:
        return (store, record_id) in self.deleted

    def __repr__(self) -> str:
        return (
            f"BlastResult(store={self.store!r}, record_id={self.record_id!r},"
            f" descendants={len(self.descendants)}, store_counts={self.store_counts!r},"
            f" truncated={self.truncated!r}, deleted={len(self.deleted)})"
        )


def _edges_tagged(edges: List[Edge], tag: str) -> List[Edge]:
    if tag not in TAGS:
        raise ValueError(f"unknown parent tag {tag!r}; expected one of {TAGS}")
    return [e for e in edges if e.tag == tag]


def _chunks(seq: List[str]) -> Iterable[List[str]]:
    for i in range(0, len(seq), _CHUNK):
        yield seq[i : i + _CHUNK]


def _marks(n: int) -> str:
    return ",".join("?" * n)


def _run(ledger: Ledger, sql: str, args) -> list:
    # Lock per statement, not per walk, so appends aren't held up.
    with ledger._lock:
        return ledger._conn.execute(sql, args).fetchall()


def _load(ledger: Ledger, write_ids: List[str], depth: int) -> List[TracedWrite]:
    out = []
    for chunk in _chunks(write_ids):
        rows = _run(
            ledger,
            f"SELECT {_WRITE_COLS} FROM writes WHERE write_id IN ({_marks(len(chunk))})",
            chunk,
        )
        out += [TracedWrite(r[1], depth, r[0], *r[2:]) for r in rows]
    out.sort(key=lambda w: w.seq)
    return out


_TOMBSTONE_COLS = "seq, store, record_id, ts, reason, write_id, agent_id, session_id"


def _group(rows: Iterable[tuple]) -> Dict[Item, List[Tombstone]]:
    out: Dict[Item, List[Tombstone]] = {}
    for r in rows:
        t = Tombstone(*r)
        out.setdefault((t.store, t.record_id), []).append(t)
    for ts in out.values():
        ts.sort(key=lambda t: t.seq)
    return out


def _tombstones(ledger: Ledger, items: Iterable[Item]) -> Dict[Item, List[Tombstone]]:
    """Tombstones of ``items``, one indexed query per store and chunk."""
    by_store: Dict[str, Set[str]] = {}
    for s, r in items:
        by_store.setdefault(s, set()).add(r)
    rows = []
    for s, ids in sorted(by_store.items()):
        for chunk in _chunks(sorted(ids)):
            rows += _run(
                ledger,
                f"SELECT {_TOMBSTONE_COLS} FROM tombstones"
                f" WHERE store = ? AND record_id IN ({_marks(len(chunk))})",
                [s, *chunk],
            )
    return dict(sorted(_group(rows).items()))


def trace(
    ledger: Ledger, store: str, record_id: str, max_depth: Optional[int] = None
) -> TraceResult:
    """Walk backward from ``(store, record_id)`` to every ancestor write.

    Breadth-first, one indexed query per level over the current frontier, so
    only the reached subgraph is read. A visited set gives each write its
    minimum depth and makes cycles terminate. ``max_depth`` (``None`` for no
    limit) bounds ancestor depth; 1 is direct parents. Deleted items are
    walked like any other; their tombstones are reported in ``deleted``.

    Raises ``KeyError`` if the item has no binding in the ledger.
    """
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    start_ids = sorted(
        {r[0] for r in _run(
            ledger,
            "SELECT write_id FROM bindings WHERE store = ? AND record_id = ?",
            (store, record_id),
        )}
    )
    if not start_ids:
        raise KeyError((store, record_id))

    items = _load(ledger, start_ids, 0)
    seen: Set[str] = set(start_ids)
    seen_unrecorded: Dict[tuple, int] = {}
    ancestors: List[TracedWrite] = []
    edges: List[Edge] = []
    has_parents: Set[str] = set()
    frontier = start_ids
    depth = 0

    while frontier and (max_depth is None or depth < max_depth):
        depth += 1
        nxt: Set[str] = set()
        for chunk in _chunks(frontier):
            rows = _run(
                ledger,
                "SELECT DISTINCT p.write_id, p.parent_store, p.parent_record_id,"
                " p.tag, b.write_id"
                " FROM parents p LEFT JOIN bindings b"
                " ON b.store = p.parent_store AND b.record_id = p.parent_record_id"
                f" WHERE p.write_id IN ({_marks(len(chunk))})",
                chunk,
            )
            for child, pstore, prec, tag, pwid in rows:
                has_parents.add(child)
                edges.append(Edge(child, pstore, prec, tag, pwid))
                if pwid is None:
                    seen_unrecorded.setdefault((pstore, prec), depth)
                elif pwid not in seen:
                    seen.add(pwid)
                    nxt.add(pwid)
        frontier = sorted(nxt)
        ancestors += _load(ledger, frontier, depth)

    truncated = False
    if frontier:
        # Stopped by max_depth: the last level wasn't expanded, so check
        # which of it still has parents (those aren't roots).
        for chunk in _chunks(frontier):
            rows = _run(
                ledger,
                f"SELECT DISTINCT write_id FROM parents WHERE write_id IN ({_marks(len(chunk))})",
                chunk,
            )
            has_parents.update(r[0] for r in rows)
        truncated = any(w in has_parents for w in frontier)

    reached = items + ancestors
    return TraceResult(
        store=store,
        record_id=record_id,
        items=items,
        ancestors=ancestors,
        edges=edges,
        roots=[w for w in reached if w.write_id not in has_parents],
        unrecorded=[
            UnrecordedAncestor(s, r, d)
            for (s, r), d in sorted(seen_unrecorded.items(), key=lambda kv: (kv[1], kv[0]))
        ],
        truncated=truncated,
        deleted=_tombstones(
            ledger, [(store, record_id)] + [(e.parent_store, e.parent_record_id) for e in edges]
        ),
    )


# The forward walk runs inside SQLite, over integers: the lineage index plus
# temp tables for its tail (what the index hasn't reached yet), all private to
# the walk's connection. It is breadth-first over writes, and also over the
# items, context-set parts and sets between them, so each item, part and set
# is expanded once however many reached writes lead to it:
#   write -> its items (write_items) -> writes naming them (direct_children)
#   item -> parts holding it (part_items) -> sets using a part (set_parts)
#        -> writes using a set (set_writes)
# walk_reach holds each reached write's seq at its minimum depth; walk_item,
# walk_part and walk_set the expanded nodes, in insertion order (rowid), so a
# level's new nodes are a rowid range. Tail items not in the index get
# negative IDs (-rowid of walk_tail_names), so the two ID spaces never collide.
_WALK_TABLES = (
    "CREATE TEMP TABLE IF NOT EXISTS walk_reach ("
    " seq INTEGER PRIMARY KEY, depth INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS temp.walk_reach_by_depth ON walk_reach (depth)",
    "CREATE TEMP TABLE IF NOT EXISTS walk_item (item INTEGER NOT NULL UNIQUE)",
    "CREATE TEMP TABLE IF NOT EXISTS walk_part (part INTEGER NOT NULL UNIQUE)",
    "CREATE TEMP TABLE IF NOT EXISTS walk_set (sid INTEGER NOT NULL UNIQUE)",
    # kind 0: a direct parent of write `id`; 1: a member row of set `id`;
    # 2: a binding of write `id`.
    "CREATE TEMP TABLE IF NOT EXISTS walk_tail_raw ("
    " store TEXT NOT NULL, record_id TEXT NOT NULL, id INTEGER NOT NULL,"
    " tag INTEGER, kind INTEGER NOT NULL)",
    "CREATE TEMP TABLE IF NOT EXISTS walk_tail_names ("
    " store TEXT NOT NULL, record_id TEXT NOT NULL, item INTEGER,"
    " UNIQUE (store, record_id))",
    "CREATE INDEX IF NOT EXISTS temp.walk_tail_names_by_item ON walk_tail_names (item)",
    "CREATE TEMP TABLE IF NOT EXISTS tail_direct ("
    " item INTEGER NOT NULL, seq INTEGER NOT NULL, tag INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS temp.tail_direct_by_item ON tail_direct (item)",
    "CREATE TEMP TABLE IF NOT EXISTS tail_part_items ("
    " item INTEGER NOT NULL, part INTEGER NOT NULL, tag INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS temp.tail_part_items_by_item ON tail_part_items (item)",
    "CREATE TEMP TABLE IF NOT EXISTS tail_set_parts ("
    " part INTEGER NOT NULL, sid INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS temp.tail_set_parts_by_part ON tail_set_parts (part)",
    "CREATE TEMP TABLE IF NOT EXISTS tail_set_writes ("
    " sid INTEGER NOT NULL, seq INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS temp.tail_set_writes_by_sid ON tail_set_writes (sid)",
    "CREATE TEMP TABLE IF NOT EXISTS tail_write_items ("
    " seq INTEGER NOT NULL, item INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS temp.tail_write_items_by_seq ON tail_write_items (seq)",
    "CREATE INDEX IF NOT EXISTS temp.tail_write_items_by_item ON tail_write_items (item)",
)
_TEMP_TABLES = (
    "walk_reach", "walk_item", "walk_part", "walk_set", "walk_tail_raw", "walk_tail_names",
    "tail_direct", "tail_part_items", "tail_set_parts", "tail_set_writes", "tail_write_items",
)
_TAIL_TABLES = ("tail_direct", "tail_part_items", "tail_set_parts", "tail_set_writes",
                "tail_write_items")

# The tail: writes past :wm, context sets past :swm, bindings past :bwm, up to
# the snapshot's ends (:top, :stop, :btop), so a later read rebuilds it exactly.
_TAIL_RAW = (
    "INSERT INTO walk_tail_raw SELECT d.parent_store, d.parent_record_id, w.seq,"
    " d.tag = 'in_prompt', 0 FROM writes w CROSS JOIN direct_parents d"
    " ON d.write_id = w.write_id WHERE w.seq > :wm AND w.seq <= :top",
    "INSERT INTO walk_tail_raw SELECT m.store, m.record_id, cs.seq,"
    " m.tag = 'in_prompt', 1 FROM context_sets cs CROSS JOIN context_members m"
    " ON m.set_hash = cs.set_hash WHERE cs.seq > :swm AND cs.seq <= :stop",
    "INSERT INTO walk_tail_raw SELECT b.store, b.record_id, w.seq, NULL, 2"
    " FROM bindings b CROSS JOIN writes w ON w.write_id = b.write_id"
    " WHERE b.rowid > :bwm AND b.rowid <= :btop",
)
_TAIL_BUILD = (
    "INSERT INTO walk_tail_names (store, record_id)"
    " SELECT DISTINCT store, record_id FROM walk_tail_raw",
    "UPDATE walk_tail_names SET item = coalesce((SELECT i.item FROM items i"
    " WHERE i.store = walk_tail_names.store AND i.record_id = walk_tail_names.record_id),"
    " -rowid)",
    "INSERT INTO tail_direct SELECT n.item, x.id, x.tag FROM walk_tail_raw x"
    " CROSS JOIN walk_tail_names n ON n.store = x.store AND n.record_id = x.record_id"
    " WHERE x.kind = 0",
    "INSERT INTO tail_part_items SELECT n.item, x.id, x.tag FROM walk_tail_raw x"
    " CROSS JOIN walk_tail_names n ON n.store = x.store AND n.record_id = x.record_id"
    " WHERE x.kind = 1",
    # A tail binding that repeats an indexed one (or another tail one) is one row.
    "INSERT INTO tail_write_items SELECT DISTINCT x.id, n.item FROM walk_tail_raw x"
    " CROSS JOIN walk_tail_names n ON n.store = x.store AND n.record_id = x.record_id"
    " WHERE x.kind = 2 AND NOT EXISTS (SELECT 1 FROM write_items y"
    " WHERE y.seq = x.id AND y.item = n.item)",
)
# No strings: chains and set use come straight from the source tables.
_TAIL_SETS = (
    "INSERT INTO tail_set_parts SELECT p.seq, s.seq FROM context_sets s"
    " CROSS JOIN context_chain c ON c.set_hash = s.set_hash"
    " CROSS JOIN context_sets p ON p.set_hash = c.part"
    " WHERE s.seq > :swm AND s.seq <= :stop",
    "INSERT INTO tail_set_writes SELECT cs.seq, w.seq FROM writes w"
    " CROSS JOIN context_sets cs ON cs.set_hash = w.context_set"
    " WHERE w.seq > :wm AND w.seq <= :top",
)


def _build_tail(conn: sqlite3.Connection, bounds: Dict[str, int]) -> Set[str]:
    """Fill the tail temp tables for ``bounds``; return the non-empty ones."""
    for t in _TEMP_TABLES:
        conn.execute(f"DELETE FROM temp.{t}")
    raw = 0
    for stmt in _TAIL_RAW:
        raw += conn.execute(stmt, bounds).rowcount
    if raw:
        for stmt in _TAIL_BUILD:
            conn.execute(stmt)
    for stmt in _TAIL_SETS:
        conn.execute(stmt, bounds)
    return {
        t for t in _TAIL_TABLES
        if conn.execute(f"SELECT EXISTS (SELECT 1 FROM temp.{t})").fetchone()[0]
    }


def _arms(select: str, tables: Iterable[Tuple[str, str]], tail: Set[str]) -> str:
    """``select`` once per (alias, table) pair whose table is in the index or
    a non-empty tail table, joined by UNION ALL. ``{t}`` names the table."""
    return " UNION ALL ".join(
        select.format(t=table) for _, table in tables
        if not table.startswith("tail_") or table in tail
    )


def _level(tail: Set[str]) -> List[str]:
    """One breadth-first level's statements, in order. Parameters: :p, the
    frontier's depth; :d = :p + 1; :i, :pt, :st, the rowids of walk_item,
    walk_part and walk_set before this level."""
    return [
        # The frontier's items, each expanded once.
        "INSERT OR IGNORE INTO walk_item (item) " + _arms(
            "SELECT wi.item FROM walk_reach cur INDEXED BY walk_reach_by_depth"
            " CROSS JOIN {t} wi ON wi.seq = cur.seq WHERE cur.depth = :p",
            [("wi", "write_items"), ("wi", "tail_write_items")], tail),
        # Writes naming a new item as a direct parent.
        "INSERT OR IGNORE INTO walk_reach " + _arms(
            "SELECT c.seq, :d FROM walk_item it CROSS JOIN {t} c ON c.item = it.item"
            " WHERE it.rowid > :i",
            [("c", "direct_children"), ("c", "tail_direct")], tail),
        # Parts holding a new item, sets using a new part, writes using a new set.
        "INSERT OR IGNORE INTO walk_part (part) " + _arms(
            "SELECT pi.part FROM walk_item it CROSS JOIN {t} pi ON pi.item = it.item"
            " WHERE it.rowid > :i",
            [("pi", "part_items"), ("pi", "tail_part_items")], tail),
        "INSERT OR IGNORE INTO walk_set (sid) " + _arms(
            "SELECT sp.sid FROM walk_part p CROSS JOIN {t} sp ON sp.part = p.part"
            " WHERE p.rowid > :pt",
            [("sp", "set_parts"), ("sp", "tail_set_parts")], tail),
        "INSERT OR IGNORE INTO walk_reach " + _arms(
            "SELECT sw.seq, :d FROM walk_set ws CROSS JOIN {t} sw ON sw.sid = ws.sid"
            " WHERE ws.rowid > :st",
            [("sw", "set_writes"), ("sw", "tail_set_writes")], tail),
    ]


# Each reached write's items, as (seq, item, store, record_id).
_BOUND = (
    "SELECT r.seq, wi.item, i.store, i.record_id FROM walk_reach r"
    " CROSS JOIN write_items wi ON wi.seq = r.seq CROSS JOIN items i ON i.item = wi.item"
)
_BOUND_TAIL = (
    "SELECT r.seq, wi.item, n.store, n.record_id FROM walk_reach r"
    " CROSS JOIN tail_write_items wi ON wi.seq = r.seq"
    " CROSS JOIN walk_tail_names n ON n.item = wi.item"
)
_REACHED = (
    "SELECT w.write_id, r.depth, w.seq, w.content_hash, w.ts, w.store,"
    " w.agent_id, w.session_id, w.source"
    " FROM walk_reach r CROSS JOIN writes w ON w.seq = r.seq ORDER BY r.depth, r.seq"
)
_TAG = ("retrieved_in_scope", "in_prompt")
_TOMBSTONE_OF = f"SELECT {_TOMBSTONE_COLS} FROM tombstones WHERE store = ? AND record_id = ?"


# Idle walk connections a ledger keeps (see ``Ledger._walk_pool``).
_POOL_MAX = 4
# Page cache per walk connection, in KiB (SQLite's convention for negative
# values): large walks read the same index pages level after level.
_WALK_CACHE_KIB = 32768


def _walk_connection(ledger: Ledger) -> Tuple[sqlite3.Connection, Callable[[], None]]:
    """A connection to walk on and the function that releases it.

    A file-backed ledger lends a connection of its own (from a small pool),
    so the walk runs in one read transaction (a consistent snapshot) and never
    holds the ledger's lock: appends aren't held up by a long walk. An
    in-memory ledger has no file to open twice, so the walk holds the
    ledger's lock and uses its connection, dropping the temp tables afterwards.
    """
    if ledger._file:
        with ledger._walk_pool_lock:
            conn = ledger._walk_pool.pop() if ledger._walk_pool else None
        if conn is None:
            conn = sqlite3.connect(ledger._file, isolation_level=None, check_same_thread=False)
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("ATTACH DATABASE ? AS lineage", (ledger._index_path,))
                for db in ("main", "lineage"):
                    conn.execute(f"PRAGMA {db}.cache_size=-{_WALK_CACHE_KIB}")
                for stmt in _WALK_TABLES:
                    conn.execute(stmt)
            except BaseException:
                conn.close()
                raise
        try:
            conn.execute("BEGIN")
        except BaseException:
            conn.close()
            raise

        def release() -> None:
            try:
                conn.execute("COMMIT")
            except sqlite3.Error:
                conn.close()
                return
            with ledger._walk_pool_lock:
                if len(ledger._walk_pool) < _POOL_MAX and not ledger._stop.is_set():
                    ledger._walk_pool.append(conn)
                    return
            conn.close()

        return conn, release

    ledger._lock.acquire()
    conn = ledger._conn

    def release_shared() -> None:
        try:
            for t in _TEMP_TABLES + ("edge_items", "edge_parts", "edge_sets"):
                conn.execute(f"DROP TABLE IF EXISTS temp.{t}")
        finally:
            ledger._lock.release()

    return conn, release_shared


def blast_radius(
    ledger: Ledger, store: str, record_id: str, max_depth: Optional[int] = None
) -> BlastResult:
    """Walk forward from ``(store, record_id)`` to every descendant write.

    Write C descends from write W when one of C's parent items is bound to W:
    the inverse of the step ``trace`` takes, so every descendant's trace
    reaches the queried item. Breadth-first with a visited set, so each write
    appears once at its minimum depth and cycles terminate. ``max_depth``
    (``None`` for no limit) bounds descendant depth; 1 is direct children.

    The walk runs inside SQLite over the integer lineage index, a few indexed
    statements per level, so only the reached subgraph is read and no strings
    move until the result is built. Items, context-set parts and sets are
    expanded once each, so a context shared by many writes costs one step, not
    one per write. What the index hasn't reached yet (past its watermarks) is
    read from the source tables and given integer IDs for this walk, so
    results are always current. A deleted item is still walked: its tombstone
    is reported in ``deleted``, and what was derived from it stays in the
    result. ``edges`` is built when first read (see ``BlastResult``).

    Raises ``KeyError`` if the item has no binding in the ledger.
    """
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    conn, release = _walk_connection(ledger)
    try:
        x = conn.execute
        for stmt in _WALK_TABLES:
            x(stmt)
        # The snapshot: index rows at or below the watermarks, the rest from
        # the source tables, all in this connection's one read transaction.
        bounds = _bounds(ledger, conn)
        tail = _build_tail(conn, bounds)

        row = x(
            "SELECT item FROM items WHERE store = ? AND record_id = ?"
            " UNION ALL SELECT item FROM walk_tail_names WHERE store = ? AND record_id = ?",
            (store, record_id, store, record_id),
        ).fetchone()
        seeded = row is not None and x(
            "INSERT OR IGNORE INTO walk_reach " + _arms(
                "SELECT seq, 0 FROM {t} WHERE item = :i",
                [("w", "write_items"), ("w", "tail_write_items")], tail),
            {"i": row[0]},
        ).rowcount
        if not seeded:
            raise KeyError((store, record_id))

        item_s, direct_s, part_s, set_s, writes_s = _level(tail)
        # walk_item, walk_part and walk_set rows before this level (rowids
        # run 1..n in insertion order, since nothing is ever deleted).
        marks = {"i": 0, "pt": 0, "st": 0}

        def expand(d: int) -> int:
            """Run level ``d``; return the writes it reached. Steps whose
            input gained nothing this level are skipped."""
            args = {"p": d - 1, "d": d, **marks}
            items = x(item_s, args).rowcount
            if not items:
                return 0
            n = x(direct_s, args).rowcount
            parts = x(part_s, args).rowcount
            sets = x(set_s, args).rowcount if parts else 0
            if sets:
                n += x(writes_s, args).rowcount
            marks["i"] += items
            marks["pt"] += parts
            marks["st"] += sets
            return n

        depth = 0
        exhausted = False
        while max_depth is None or depth < max_depth:
            depth += 1
            if not expand(depth):
                exhausted = True
                break
        truncated = False
        if not exhausted:
            # Stopped by max_depth: would one more level reach anything?
            x("SAVEPOINT probe")
            try:
                truncated = expand(depth + 1) > 0
            finally:
                x("ROLLBACK TO probe")
                x("RELEASE probe")

        reached = x(_REACHED).fetchall()
        bound = x(_BOUND).fetchall()
        if "tail_write_items" in tail:
            bound += x(_BOUND_TAIL).fetchall()
        deleted_rows = x(_TOMBSTONE_OF, (store, record_id)).fetchall()
        if x("SELECT EXISTS (SELECT 1 FROM tombstones)").fetchone()[0]:
            deleted_rows += _tombstones_of_bound(conn, tail)
    finally:
        release()

    # Building tens of thousands of tuples triggers repeated generational
    # collections that find nothing to free; pause the collector meanwhile.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        return _blast_result(
            ledger, store, record_id, max_depth, bounds, reached, bound, truncated,
            deleted_rows,
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def _bounds(ledger: Ledger, conn: sqlite3.Connection) -> Dict[str, int]:
    """The walk's snapshot: the index watermarks, and how far each source
    table reached when it was read. Everything is append-only, so these
    bounds reproduce exactly what the walk saw."""
    wm, swm, bwm = ledger._index_watermarks(conn)
    top, stop, btop = conn.execute(
        "SELECT (SELECT coalesce(max(seq), 0) FROM writes),"
        " (SELECT coalesce(max(seq), 0) FROM context_sets),"
        " (SELECT coalesce(max(rowid), 0) FROM bindings)"
    ).fetchone()
    return {"wm": wm, "swm": swm, "bwm": bwm, "top": top, "stop": stop, "btop": btop}


def _tombstones_of_bound(conn: sqlite3.Connection, tail: Set[str]) -> list:
    """Tombstones of every item bound to a reached write."""
    sql = (
        f"SELECT {_TOMBSTONE_COLS} FROM tombstones WHERE seq IN ("
        " SELECT t.seq FROM walk_reach r CROSS JOIN write_items wi ON wi.seq = r.seq"
        " CROSS JOIN items i ON i.item = wi.item"
        " CROSS JOIN tombstones t ON t.store = i.store AND t.record_id = i.record_id"
    )
    if "tail_write_items" in tail:
        sql += (
            " UNION ALL SELECT t.seq FROM walk_reach r"
            " CROSS JOIN tail_write_items wi ON wi.seq = r.seq"
            " CROSS JOIN walk_tail_names n ON n.item = wi.item"
            " CROSS JOIN tombstones t ON t.store = n.store AND t.record_id = n.record_id"
        )
    return conn.execute(sql + ")").fetchall()


def _blast_result(
    ledger, store, record_id, max_depth, bounds, reached, bound, truncated, deleted_rows
) -> BlastResult:
    reached = list(map(_traced_write, reached))
    wid = {w.seq: w.write_id for w in reached}
    bindings: Dict[str, List[Item]] = {w.write_id: [] for w in reached}
    for seq, _, s, r in bound:
        bindings[wid[seq]].append((s, r))
    for items_of in bindings.values():
        if len(items_of) > 1:
            items_of.sort()
    descendants = [w for w in reached if w.depth]
    # Writes at max_depth weren't expanded, so their edges weren't walked.
    if max_depth is None:
        expanded = bound
    else:
        depth_of = {w.seq: w.depth for w in reached}
        expanded = [b for b in bound if depth_of[b[0]] < max_depth]
    return BlastResult(
        store=store,
        record_id=record_id,
        items=[w for w in reached if not w.depth],
        descendants=descendants,
        bindings=bindings,
        store_counts=dict(sorted(Counter(w.store for w in descendants).items())),
        truncated=truncated,
        deleted=dict(sorted(_group(set(deleted_rows)).items())),
        load_edges=partial(_load_edges, ledger, bounds, wid, expanded),
    )


# Edges from the parent items in temp.edge_items: direct ones, then context
# ones through parts and sets, each as (item, child seq, tag). Index rows are
# bounded by the walk's watermarks (the index may have grown since).
_EDGE_TABLES = (
    "CREATE TEMP TABLE IF NOT EXISTS edge_items (item INTEGER PRIMARY KEY)",
    "CREATE TEMP TABLE IF NOT EXISTS edge_parts ("
    " item INTEGER NOT NULL, part INTEGER NOT NULL, tag INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS temp.edge_parts_by_part ON edge_parts (part)",
    "CREATE TEMP TABLE IF NOT EXISTS edge_sets ("
    " item INTEGER NOT NULL, sid INTEGER NOT NULL, tag INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS temp.edge_sets_by_sid ON edge_sets (sid)",
    "DELETE FROM edge_items",
    "DELETE FROM edge_parts",
    "DELETE FROM edge_sets",
)


def _edge_queries(tail: Set[str]) -> Tuple[str, List[str], str]:
    direct = _arms(
        "SELECT e.item, c.seq, c.tag FROM edge_items e CROSS JOIN {t} c ON c.item = e.item"
        " WHERE c.seq <= :wm",
        [("c", "direct_children")], tail,
    ) + (
        " UNION ALL SELECT e.item, c.seq, c.tag FROM edge_items e"
        " CROSS JOIN tail_direct c ON c.item = e.item" if "tail_direct" in tail else ""
    )
    steps = [
        "INSERT INTO edge_parts SELECT e.item, pi.part, pi.tag FROM edge_items e"
        " CROSS JOIN part_items pi ON pi.item = e.item WHERE pi.part <= :swm"
        + (" UNION ALL SELECT e.item, pi.part, pi.tag FROM edge_items e"
           " CROSS JOIN tail_part_items pi ON pi.item = e.item"
           if "tail_part_items" in tail else ""),
        "INSERT INTO edge_sets SELECT ep.item, sp.sid, max(ep.tag) FROM edge_parts ep"
        " CROSS JOIN (SELECT part, sid FROM set_parts WHERE sid <= :swm"
        + (" UNION ALL SELECT part, sid FROM tail_set_parts" if "tail_set_parts" in tail else "")
        + ") sp ON sp.part = ep.part GROUP BY ep.item, sp.sid",
    ]
    context = (
        "SELECT es.item, sw.seq, max(es.tag) FROM edge_sets es"
        " CROSS JOIN (SELECT sid, seq FROM set_writes WHERE seq <= :wm"
        + (" UNION ALL SELECT sid, seq FROM tail_set_writes" if "tail_set_writes" in tail else "")
        + ") sw ON sw.sid = es.sid GROUP BY es.item, sw.seq"
    )
    return direct, steps, context


def _load_edges(
    ledger: Ledger, bounds: Dict[str, int], wid: Dict[int, str], parents: List[tuple]
) -> List[Edge]:
    """The edges a walk followed, as of its snapshot ``bounds``.

    ``parents`` are ``(seq, item, store, record_id)`` for every item bound to
    an expanded write. Each child write gets one edge per such item, tagged as
    the ``parents`` view tags it: a direct parent's own (merged) tag, else
    ``in_prompt`` if any part of the child's context set holds the item as
    ``in_prompt``. Index rows are bounded by the walk's watermarks and the
    tail is rebuilt from the same ranges, so this reads exactly what the walk
    read, whatever has been appended since.
    """
    by_item: Dict[int, List[int]] = {}
    names: Dict[int, Item] = {}
    for seq, item, s, r in parents:
        by_item.setdefault(item, []).append(seq)
        names[item] = (s, r)
    if not by_item:
        return []
    conn, release = _walk_connection(ledger)
    try:
        x = conn.execute
        for stmt in _WALK_TABLES + _EDGE_TABLES:
            x(stmt)
        tail = _build_tail(conn, bounds)
        # Tail item IDs are this connection's: map the walk's by name.
        ids = {item: item for item in by_item if item > 0}
        if any(item < 0 for item in by_item):
            by_name = {
                (s, r): i for s, r, i in x("SELECT store, record_id, item FROM walk_tail_names")
            }
            for item in by_item:
                if item < 0:
                    ids[item] = by_name[names[item]]
        back = {v: k for k, v in ids.items()}
        x("INSERT INTO edge_items SELECT value FROM json_each(?)", (json.dumps(list(back)),))
        direct, steps, context = _edge_queries(tail)
        rows = {(back[i], seq): tag for i, seq, tag in x(direct, bounds)}
        for stmt in steps:
            x(stmt, bounds)
        for i, seq, tag in x(context, bounds):
            rows.setdefault((back[i], seq), tag)  # a direct parent's tag wins
    finally:
        release()
    edges = []
    append = edges.append
    gc_was_enabled = gc.isenabled()
    gc.disable()  # see blast_radius
    try:
        for (item, seq), tag in rows.items():
            s, r = names[item]
            child = wid[seq]
            t = _TAG[tag]
            for parent in by_item[item]:
                append(_edge((child, s, r, t, wid[parent])))
    finally:
        if gc_was_enabled:
            gc.enable()
    return edges
