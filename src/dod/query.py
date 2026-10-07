"""Read-only lineage queries over the ledger.

A parent edge names a memory item ``(store, record_id)``; ``bindings`` map it
to the write(s) that produced it, so walks cross stores. An item bound to
several writes resolves to all of them (over-include rather than miss a
parent).
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

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

    def edges_tagged(self, tag: str) -> List[Edge]:
        return _edges_tagged(self.edges, tag)


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

    def edges_tagged(self, tag: str) -> List[Edge]:
        return _edges_tagged(self.edges, tag)


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


def trace(
    ledger: Ledger, store: str, record_id: str, max_depth: Optional[int] = None
) -> TraceResult:
    """Walk backward from ``(store, record_id)`` to every ancestor write.

    Breadth-first, one indexed query per level over the current frontier, so
    only the reached subgraph is read. A visited set gives each write its
    minimum depth and makes cycles terminate. ``max_depth`` (``None`` for no
    limit) bounds ancestor depth; 1 is direct parents.

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
    )


def blast_radius(
    ledger: Ledger, store: str, record_id: str, max_depth: Optional[int] = None
) -> BlastResult:
    """Walk forward from ``(store, record_id)`` to every descendant write.

    Write C descends from write W when one of C's parent items is bound to W:
    the inverse of the step ``trace`` takes, so every descendant's trace
    reaches the queried item. Breadth-first with a visited set, so each write
    appears once at its minimum depth and cycles terminate. ``max_depth``
    (``None`` for no limit) bounds descendant depth; 1 is direct children.

    There is no parent->child index (kept off the append path), so
    ``bindings``, ``parents`` and ``writes`` are each read in one scan and the
    walk runs in memory, rather than rescanning ``parents`` at every level.

    Raises ``KeyError`` if the item has no binding in the ledger.
    """
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    start_ids: List[str] = []
    items_of: Dict[str, List[Tuple[str, str]]] = {}
    for wid, s, r in _run(ledger, "SELECT write_id, store, record_id FROM bindings", ()):
        bound = items_of.setdefault(wid, [])
        if (s, r) in bound:
            continue  # the same binding appended twice
        bound.append((s, r))
        if s == store and r == record_id:
            start_ids.append(wid)
    if not start_ids:
        raise KeyError((store, record_id))
    start_ids.sort()

    children: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for child, pstore, prec, tag in _run(
        ledger, "SELECT write_id, parent_store, parent_record_id, tag FROM parents", ()
    ):
        children.setdefault((pstore, prec), []).append((child, tag))

    seen: Dict[str, int] = dict.fromkeys(start_ids, 0)
    edges: List[Edge] = []
    frontier = start_ids
    depth = 0

    while frontier and (max_depth is None or depth < max_depth):
        depth += 1
        nxt: List[str] = []
        for wid in frontier:
            for item in items_of.get(wid, ()):
                for child, tag in children.get(item, ()):
                    edges.append(Edge(child, item[0], item[1], tag, wid))
                    if child not in seen:
                        seen[child] = depth
                        nxt.append(child)
        frontier = nxt

    # Stopped by max_depth: the last level wasn't expanded, so check whether
    # it has children the walk hasn't reached.
    truncated = any(
        child not in seen
        for wid in frontier
        for item in items_of.get(wid, ())
        for child, _ in children.get(item, ())
    )
    reached = [
        TracedWrite(r[1], seen[r[1]], r[0], *r[2:])
        for r in _run(ledger, f"SELECT {_WRITE_COLS} FROM writes ORDER BY seq", ())
        if r[1] in seen
    ]
    descendants = sorted((w for w in reached if w.depth), key=lambda w: w.depth)
    return BlastResult(
        store=store,
        record_id=record_id,
        items=[w for w in reached if not w.depth],
        descendants=descendants,
        edges=edges,
        bindings={w: sorted(items_of.get(w, ())) for w in seen},
        store_counts=dict(sorted(Counter(w.store for w in descendants).items())),
        truncated=truncated,
    )
