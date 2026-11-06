"""Propagation metrics, computed from the ledger.

Lineage comes only from the ledger: ``trace``, ``blast_radius``, a dry-run
``excise``, tombstones, and edge tags. The one outside input is ``wrong``,
the set of live items the scenario's ground-truth checker flagged; it labels
items, it never adds or removes edges. Results hold IDs and counts only.

Definitions (also in ``README.md`` and ``docs/decisions.md``):

- derived writes: writes in the planted item's blast radius (depth >= 1),
  over all edges. ``derived_writes_in_prompt`` follows ``in_prompt`` edges
  only.
- derived over time: cumulative derived writes after each step; a write's
  step is its ``session_id`` (``step-NNNN``).
- contaminating edges: the parent edges ``blast_radius`` walked forward from
  the planted item, split by tag.
- outputs: items in the ``tickets`` namespace of the LangGraph store. An
  output traces to the planted item if any of the planted item's writes is
  among its ``trace`` ancestors.
- excise vs wipe: live items (bound, no tombstone) a quarantine ``excise``
  of the planted item would remove, against every live item.
  ``excise_in_prompt`` is the planted item plus items bound to writes in
  ``derived_writes_in_prompt``: what excision over ``in_prompt`` edges only
  would remove (``excise`` itself follows every edge).
"""

import json
import sqlite3
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from dod.excise import excise
from dod.ledger import Ledger
from dod.query import blast_radius, trace

Item = Tuple[str, str]


def is_output(item: Item) -> bool:
    store, record_id = item
    if store != "langgraph":
        return False
    try:
        return json.loads(record_id)[0] == "tickets"
    except (ValueError, IndexError, TypeError):
        return False


def step_of(session_id: Optional[str]) -> Optional[int]:
    if session_id and session_id.startswith("step-"):
        return int(session_id[5:])
    return None


def _ledger_items(path: str) -> Tuple[List[Item], Set[Item], int]:
    """Bound items, tombstoned items and write count, on a read-only connection."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        items = [tuple(r) for r in conn.execute(
            "SELECT DISTINCT store, record_id FROM bindings ORDER BY store, record_id")]
        dead = {tuple(r) for r in conn.execute("SELECT store, record_id FROM tombstones")}
        writes = conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0]
    finally:
        conn.close()
    return items, dead, writes


def _reach(start: Iterable[str], edges, forward: bool, tag: Optional[str]) -> Set[str]:
    """Writes reachable from ``start`` over ``edges`` (optionally one tag only)."""
    adj: Dict[str, List[str]] = {}
    for e in edges:
        if e.parent_write_id is None or (tag and e.tag != tag):
            continue
        a, b = (e.parent_write_id, e.child_write_id) if forward else (e.child_write_id, e.parent_write_id)
        adj.setdefault(a, []).append(b)
    seen = set(start)
    frontier = list(seen)
    while frontier:
        nxt = []
        for w in frontier:
            for x in adj.get(w, ()):
                if x not in seen:
                    seen.add(x)
                    nxt.append(x)
        frontier = nxt
    return seen


def _split(items: Iterable[Item], wrong: Set[Item],
           wrong_planted: Optional[Set[Item]] = None) -> Dict[str, int]:
    items = list(items)
    n_wrong = sum(1 for i in items if i in wrong)
    out = {"items": len(items), "wrong": n_wrong, "correct": len(items) - n_wrong}
    if wrong_planted is not None:
        out["wrong_planted"] = sum(1 for i in items if i in wrong_planted)
    return out


def _frac(n: int, d: int) -> Optional[float]:
    return round(n / d, 4) if d else None


def planted_metrics(ledger: Ledger, planted: Item, live: Set[Item], wrong: Set[Item],
                    steps: int, wrong_planted: Optional[Set[Item]] = None) -> dict:
    b = blast_radius(ledger, *planted)
    starts = [w.write_id for w in b.items]
    in_prompt = _reach(starts, b.edges, forward=True, tag="in_prompt") - set(starts)

    per_step = [0] * (steps + 1)
    for w in b.descendants:
        s = step_of(w.session_id)
        if s is not None and 1 <= s <= steps:
            per_step[s] += 1
    over_time, total = [], 0
    for n in per_step[1:]:
        total += n
        over_time.append(total)

    tags = {"in_prompt": len(b.edges_tagged("in_prompt")),
            "retrieved_in_scope": len(b.edges_tagged("retrieved_in_scope"))}
    def plan(**options) -> List[Item]:
        targets = excise(ledger, *planted, dry_run=True, **options).targets
        return [(s, r) for s, rs in targets.items() for r in rs]

    target_items = plan()
    derived_items = {i for w in b.descendants for i in b.bindings.get(w.write_id, ())}
    ip_items = {planted}.union(*(b.bindings.get(w, ()) for w in in_prompt))
    return {
        "store": planted[0],
        "record_id": planted[1],
        "derived_writes": len(b.descendants),
        "derived_writes_in_prompt": len(in_prompt),
        "derived_over_time": over_time,
        "max_depth": max((w.depth for w in b.descendants), default=0),
        "stores_reached": b.store_counts,
        "contaminating_edges": dict(tags, in_prompt_share=_frac(
            tags["in_prompt"], tags["in_prompt"] + tags["retrieved_in_scope"])),
        "deleted_derived_items": sum(1 for i in derived_items if i in b.deleted),
        "excise": _split((i for i in target_items if i in live), wrong, wrong_planted),
        "excise_in_prompt": _split(sorted(i for i in ip_items if i in live), wrong, wrong_planted),
        # The library's own options (excise(in_prompt_only=, precise=)):
        # precise leaves out writes that only read an earlier, clean version
        # of an item (the scenario rewrites knowledge-base entries).
        "excise_lib_in_prompt": _split((i for i in plan(in_prompt_only=True) if i in live),
                                       wrong, wrong_planted),
        "excise_precise": _split((i for i in plan(precise=True) if i in live), wrong,
                                 wrong_planted),
        "excise_precise_in_prompt": _split(
            (i for i in plan(in_prompt_only=True, precise=True) if i in live), wrong,
            wrong_planted),
    }


def output_metrics(ledger: Ledger, outputs: List[Item], planted: List[Item],
                   wrong: Set[Item]) -> dict:
    planted_writes = set()
    for p in planted:
        planted_writes.update(w.write_id for w in trace(ledger, *p).items)
    wrong_out = [o for o in outputs if o in wrong]
    traced = traced_ip = correct_traced = 0
    for o in outputs:
        t = trace(ledger, *o)
        hit = bool(planted_writes & {w.write_id for w in t.ancestors})
        if o in wrong:
            traced += hit
            ip = _reach([w.write_id for w in t.items], t.edges, forward=False, tag="in_prompt")
            traced_ip += bool(planted_writes & ip)
        else:
            correct_traced += hit
    return {
        "total": len(outputs),
        "wrong": len(wrong_out),
        "wrong_tracing_to_planted": traced,
        "wrong_tracing_to_planted_in_prompt": traced_ip,
        "fraction_wrong_tracing": _frac(traced, len(wrong_out)),
        "fraction_wrong_tracing_in_prompt": _frac(traced_ip, len(wrong_out)),
        "correct_tracing_to_planted": correct_traced,
    }


def compute(ledger: Ledger, path: str, planted: List[Item], wrong: Set[Item], steps: int,
            output: Callable[[Item], bool] = is_output,
            wrong_planted: Optional[Set[Item]] = None) -> dict:
    """All metrics for one run. ``path`` is the ledger file ``ledger`` has open."""
    items, dead, writes = _ledger_items(path)
    live = {i for i in items if i not in dead}
    wrong = {i for i in wrong if i in live}
    outputs = [i for i in items if output(i) and i in live]
    return {
        "ledger": {"writes": writes, "items": len(items), "tombstoned_items": len(dead)},
        "planted": [planted_metrics(ledger, p, live, wrong, steps,
                                    None if wrong_planted is None else wrong_planted & live)
                    for p in planted],
        "outputs": output_metrics(ledger, outputs, planted, wrong),
        "wipe": _split(sorted(live), wrong,
                       None if wrong_planted is None else wrong_planted & live),
    }
