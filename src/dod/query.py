"""Read-only lineage queries over the ledger.

A parent edge names a memory item ``(store, record_id)``; ``bindings`` map it
to the write(s) that produced it, so walks cross stores. An item bound to
several writes resolves to all of them (over-include rather than miss a
parent). ``trace`` reads edges from the ledger's ``parents`` view, which
expands each write's context set and direct parents into one row per edge, so
queries never see how they are stored. ``blast_radius`` reads the same edges
by parent from ``child_edges`` plus the not-yet-indexed tail.
"""

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from operator import itemgetter
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from .ledger import _EDGES_BETWEEN, TAGS, Ledger

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


@dataclass
class BlastResult:
    store: str
    record_id: str
    items: List[TracedWrite]
    """Writes bound to the queried item (depth 0)."""
    descendants: List[TracedWrite]
    """Descendant writes, each once at its minimum depth, ordered by (depth, seq)."""
    edges: List[Edge]
    """Parent edges walked forward; ``parent_write_id`` is the reached write."""
    bindings: Dict[str, List[Tuple[str, str]]]
    """``(store, record_id)`` items bound to each reached write, depth 0 included."""
    store_counts: Dict[str, int]
    """Number of descendant writes per store."""
    truncated: bool = False
    """True if ``max_depth`` stopped the walk before every descendant was reached."""
    deleted: Dict[Item, List[Tombstone]] = field(default_factory=dict)
    """Tombstones, ordered by seq, of the queried item and every item bound
    to a reached write, keyed by ``(store, record_id)``. Items never deleted
    are absent."""

    def edges_tagged(self, tag: str) -> List[Edge]:
        return _edges_tagged(self.edges, tag)

    def is_deleted(self, store: str, record_id: str) -> bool:
        return (store, record_id) in self.deleted


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


# The forward walk runs inside SQLite, over temp tables private to its
# connection, appended level by level so each level is a rowid range:
# walk_reach holds each reached write, keyed by seq, at its minimum depth;
# walk_items the distinct items bound to each reached write; walk_edges the
# edges walked from them; walk_tail the edges of writes past the child_edges
# watermark. Python reads only the result.
_WALK_TABLES = """
CREATE TEMP TABLE walk_reach (
    seq INTEGER PRIMARY KEY, write_id TEXT NOT NULL, depth INTEGER NOT NULL
);
CREATE INDEX temp.walk_reach_by_depth ON walk_reach (depth);
CREATE TEMP TABLE walk_items (seq INTEGER, write_id TEXT, store TEXT, record_id TEXT);
CREATE TEMP TABLE walk_edges (
    seq INTEGER, write_id TEXT, store TEXT, record_id TEXT, tag TEXT, parent TEXT
);
CREATE TEMP TABLE walk_tail (
    parent_store TEXT, parent_record_id TEXT, seq INTEGER, write_id TEXT, tag TEXT
);
CREATE INDEX temp.walk_tail_by_parent ON walk_tail (parent_store, parent_record_id);
"""

# The children of the walk_items rows in (:lo, :hi], from child_edges (at or
# below the watermark) and the tail, as walk_edges rows. CROSS JOIN fixes the
# join order: the items drive every lookup.
_CHILDREN = " UNION ALL ".join(
    "SELECT c.seq, c.write_id, i.store, i.record_id, c.tag, i.write_id"
    f" FROM walk_items i CROSS JOIN {table} c"
    " ON c.parent_store = i.store AND c.parent_record_id = i.record_id"
    f"{cond} WHERE i.rowid > :lo AND i.rowid <= :hi"
    for table, cond in (("child_edges", " AND c.seq <= :wm"), ("walk_tail", ""))
)
_ADD_EDGES = f"INSERT INTO walk_edges {_CHILDREN}"
# INSERT OR IGNORE keeps each write's first, so minimum, depth.
_ADD_REACHED = (
    "INSERT OR IGNORE INTO walk_reach SELECT seq, write_id, :d FROM walk_edges"
    " WHERE rowid > :lo"
)
# DISTINCT drops a binding appended twice; a write's bindings are all read
# at its one depth.
_ADD_ITEMS = (
    "INSERT INTO walk_items SELECT DISTINCT f.seq, b.write_id, b.store, b.record_id"
    " FROM walk_reach f CROSS JOIN bindings b ON b.write_id = f.write_id"
    " WHERE f.depth = :d"
)
_UNREACHED_CHILD = (
    f"SELECT EXISTS (SELECT 1 FROM ({_CHILDREN}) e"
    " WHERE e.seq NOT IN (SELECT seq FROM walk_reach))"
)
_REACHED = (
    "SELECT w.write_id, f.depth, w.seq, w.content_hash, w.ts, w.store,"
    " w.agent_id, w.session_id, w.source"
    " FROM walk_reach f CROSS JOIN writes w ON w.seq = f.seq"
)
_DELETED = (
    f"SELECT {_TOMBSTONE_COLS} FROM tombstones WHERE seq IN ("
    " SELECT t.seq FROM walk_items i CROSS JOIN tombstones t"
    " ON t.store = i.store AND t.record_id = i.record_id)"
)


def _walk_connection(ledger: Ledger) -> Tuple[sqlite3.Connection, Callable[[], None]]:
    """A connection to walk on and the function that releases it.

    A file-backed ledger gets a connection of its own, so the walk runs in
    one read transaction (a consistent snapshot) and never holds the
    ledger's lock: appends aren't held up by a long walk. An in-memory
    ledger has no file to open twice, so the walk holds the ledger's lock
    and uses its connection, dropping the temp tables afterwards.
    """
    path = next(
        (f for _, name, f in _run(ledger, "PRAGMA database_list", ()) if name == "main"), ""
    )
    if path:
        conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("BEGIN")
        except BaseException:
            conn.close()
            raise
        return conn, conn.close

    ledger._lock.acquire()
    conn = ledger._conn

    def release() -> None:
        try:
            for t in ("walk_reach", "walk_items", "walk_edges", "walk_tail"):
                conn.execute(f"DROP TABLE IF EXISTS temp.{t}")
        finally:
            ledger._lock.release()

    return conn, release


def blast_radius(
    ledger: Ledger, store: str, record_id: str, max_depth: Optional[int] = None
) -> BlastResult:
    """Walk forward from ``(store, record_id)`` to every descendant write.

    Write C descends from write W when one of C's parent items is bound to W:
    the inverse of the step ``trace`` takes, so every descendant's trace
    reaches the queried item. Breadth-first with a visited set, so each write
    appears once at its minimum depth and cycles terminate. ``max_depth``
    (``None`` for no limit) bounds descendant depth; 1 is direct children.

    Breadth-first over the ledger's ``child_edges`` index, a few indexed
    statements per level over the current frontier, so only the reached
    subgraph is read. The levels run inside SQLite, in temp tables on the
    walk's own connection, so Python builds only the result, not per-level
    state. The ledger's background thread fills that index; writes it hasn't
    reached yet (past its watermark) are read from the source tables once
    and merged, so results are always current. A deleted item is still walked:
    its tombstone is reported in ``deleted``, and what was derived from it
    stays in the result.

    Raises ``KeyError`` if the item has no binding in the ledger.
    """
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    conn, release = _walk_connection(ledger)
    try:
        def run(sql: str, **args) -> sqlite3.Cursor:
            return conn.execute(sql, args)

        for stmt in _WALK_TABLES.split(";"):
            if stmt.strip():
                conn.execute(stmt)
        if not run(
            "INSERT INTO walk_reach SELECT seq, write_id, 0 FROM writes WHERE write_id IN"
            " (SELECT write_id FROM bindings WHERE store = :s AND record_id = :r)",
            s=store, r=record_id,
        ).rowcount:
            raise KeyError((store, record_id))
        # Edges at or below the watermark come from child_edges, the rest
        # from the tail, so a fill committing mid-walk changes nothing.
        ((wm,),) = run("SELECT coalesce(max(upto), 0) FROM child_edges_filled").fetchall()
        run(
            "INSERT INTO walk_tail SELECT parent_store, parent_record_id, seq, write_id, tag"
            f" FROM ({_EDGES_BETWEEN})",
            lo=wm, hi=2**63 - 1,
        )

        # The frontier's items are walk_items rows (lo, hi].
        lo, hi = 0, run(_ADD_ITEMS, d=0).rowcount
        n_edges = 0
        depth = 0
        exhausted = False
        while max_depth is None or depth < max_depth:
            depth += 1
            added = run(_ADD_EDGES, lo=lo, hi=hi, wm=wm).rowcount
            new = run(_ADD_REACHED, d=depth, lo=n_edges).rowcount
            n_edges += added
            if not new:
                exhausted = True
                break
            lo, hi = hi, hi + run(_ADD_ITEMS, d=depth).rowcount
        # Stopped by max_depth, the last level wasn't expanded: check whether
        # it has children the walk hasn't reached.
        truncated = not exhausted and bool(
            run(_UNREACHED_CHILD, lo=lo, hi=hi, wm=wm).fetchone()[0]
        )

        edges = list(map(_edge, run(
            "SELECT write_id, store, record_id, tag, parent FROM walk_edges"
        )))
        reached = list(map(_traced_write, run(_REACHED)))
        bound = run("SELECT write_id, store, record_id FROM walk_items").fetchall()
        # walk_items holds the queried item, bound to every depth-0 write.
        deleted = _group(run(_DELETED))
    finally:
        release()

    reached.sort(key=itemgetter(1, 2))  # (depth, seq)
    bindings: Dict[str, List[Item]] = {w.write_id: [] for w in reached}
    for wid, s, r in bound:
        bindings[wid].append((s, r))
    for items_of in bindings.values():
        if len(items_of) > 1:
            items_of.sort()
    descendants = [w for w in reached if w.depth]
    return BlastResult(
        store=store,
        record_id=record_id,
        items=[w for w in reached if not w.depth],
        descendants=descendants,
        edges=edges,
        bindings=bindings,
        store_counts=dict(sorted(Counter(w.store for w in descendants).items())),
        truncated=truncated,
        deleted=dict(sorted(deleted.items())),
    )
