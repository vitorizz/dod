"""Regenerate: replace what a bad item contaminated with clean rewrites.

``excise`` removes an item and everything derived from it. ``regenerate``
does the same, then gives each derived item a chance to be rebuilt from its
clean inputs, so the agent keeps what it learned without the bad parent.

The library never stores content or calls a model, so the rewrite is the
caller's: ``rewrite(task)`` is called once per derived item, oldest first,
inside its own ``agent_task_scope``. It writes the replacement through an
instrumented store as a **new record** and returns its ``(store,
record_id)`` (or ``None`` to drop the item without a replacement). Capture
records that write like any other, with whatever the callback retrieved as
its parents, so the replacement's lineage is real, not asserted.

Steps:

1. The targets are ``excise``'s: the named item plus every item bound to a
   write in its blast radius. All are quarantined first, so instrumented
   reads stop returning them while the rewrites run and no rewrite can
   take one as a parent by retrieving it.
2. Each derived item (not the named item, and not other items bound to its
   own writes: those carry the bad content itself) becomes a
   ``RegenerateTask``, ordered by its latest write's ``seq``, so an item's
   contaminated parents are rewritten before it. The task lists the
   original write's parents and what became of each: ``clean`` (untouched),
   ``bad`` (the named item or its writes' items), ``replaced`` (rewritten
   earlier in this call, with its replacement) or ``dropped``.
3. After each rewrite, the replacement's latest bound write must exist and
   have no contaminated direct parent. Its parents are then clean by
   induction: every other parent is outside the blast radius, or a
   replacement checked the same way. A replacement that fails the check
   (not captured, written in place over a target, or derived from a target)
   is quarantined too and reported in ``failed``; so is a callback that
   raises. Later tasks see that item as ``dropped``.
4. ``then="delete"`` deletes the old targets through their stores' delete
   adapters once every rewrite has run (tombstone reason ``excision``);
   the default ``"quarantine"`` leaves them quarantined, which is reversible.

Rewriting in place (the same record ID) is refused: the record would stay
bound to its old write, so ``trace`` from it, and ``blast_radius`` from the
bad item, would still connect them. The callback should not read the old
item either (it is quarantined): content derived from it would be
contaminated without an edge saying so.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple

from dod.excise import DeleteAdapter, Skipped, _delete, _quarantine, _targets
from dod.excise import ExciseResult
from dod.ledger import Item, Ledger
from dod.query import BlastResult, blast_radius
from dod.scope import agent_task_scope

THEN = ("quarantine", "delete")
STATUSES = ("clean", "bad", "replaced", "dropped")

# Stay under SQLITE_MAX_VARIABLE_NUMBER on old builds (999).
_CHUNK = 900


class Parent(NamedTuple):
    """One parent edge of a derived item's original write(s)."""

    store: str
    record_id: str
    tag: str
    status: str
    """One of ``STATUSES``."""
    replacement: Optional[Item]
    """For ``replaced``: the item that replaces it (``None`` in a dry run's
    plan, before it exists)."""


@dataclass
class RegenerateTask:
    """One derived item to rewrite."""

    store: str
    record_id: str
    write_ids: List[str]
    """The item's writes in the blast radius, oldest first."""
    depth: int
    """Its shallowest write's distance from the bad item."""
    parents: List[Parent]
    """Its writes' parents, with what became of each."""

    @property
    def item(self) -> Item:
        return (self.store, self.record_id)

    def clean_parents(self) -> List[Item]:
        """Inputs the rewrite may use: untouched parents and replacements."""
        out = []
        for p in self.parents:
            if p.status == "clean":
                out.append((p.store, p.record_id))
            elif p.status == "replaced" and p.replacement is not None:
                out.append(p.replacement)
        return list(dict.fromkeys(out))


class Failed(NamedTuple):
    store: str
    record_id: str
    reason: str


@dataclass
class RegenerateResult:
    store: str
    record_id: str
    then: str
    dry_run: bool
    tasks: List[RegenerateTask]
    """Derived items, in the order they were (or would be) rewritten."""
    targets: Dict[str, List[str]]
    """Record IDs per store: the bad item and everything derived from it."""
    replaced: Dict[Item, Item] = field(default_factory=dict)
    """Old item -> its checked replacement."""
    dropped: List[Item] = field(default_factory=list)
    """Items the callback chose not to replace."""
    failed: List[Failed] = field(default_factory=list)
    """Items whose rewrite raised or failed the lineage check."""
    done: Dict[str, List[str]] = field(default_factory=dict)
    """Old targets quarantined or deleted by this call, per store."""
    skipped: List[Skipped] = field(default_factory=list)
    """Old targets left as they were (already quarantined, not in store,
    delete failed)."""
    blast: Optional[BlastResult] = None


def regenerate(
    ledger: Ledger,
    store: str,
    record_id: str,
    rewrite: Optional[Callable[[RegenerateTask], Optional[Item]]] = None,
    then: str = "quarantine",
    stores: Optional[Mapping[str, DeleteAdapter]] = None,
    dry_run: bool = False,
    agent_id: str = "dod-regenerate",
) -> RegenerateResult:
    """Excise ``(store, record_id)`` and rewrite each item derived from it
    through ``rewrite``; see the module docs.

    ``rewrite`` is required unless ``dry_run``. ``then`` is what happens to
    the old targets: ``"quarantine"`` (default) or ``"delete"``, which needs
    ``stores`` (store name to delete adapter) covering every target store;
    a missing adapter raises ``ValueError`` before anything changes. Each
    rewrite runs in ``agent_task_scope(agent_id, "regenerate-<n>")``.

    Raises ``KeyError`` if the item has no binding in the ledger.
    """
    if then not in THEN:
        raise ValueError(f"unknown then {then!r}; expected one of {THEN}")
    if rewrite is None and not dry_run:
        raise ValueError("rewrite is required unless dry_run=True")
    if then == "delete" and stores is None:
        raise ValueError('then="delete" requires stores: {store name: delete adapter}')

    blast = blast_radius(ledger, store, record_id)
    root = (store, record_id)
    grouped = _targets(blast, root)
    if then == "delete":
        missing = sorted(s for s in grouped if s not in stores)
        if missing:
            raise ValueError(f"no delete adapter for store(s) {missing}; nothing was changed")
    old = [i for its in grouped.values() for i in its]
    contaminated = set(old)
    tasks, bad = _plan(ledger, blast, root)

    result = RegenerateResult(
        store, record_id, then, dry_run, tasks,
        {s: [r for _, r in items] for s, items in grouped.items()}, blast=blast,
    )
    # The plan's statuses: tasks before this one count as replaced (by an
    # item not yet known), later ones as pending.
    planned: Dict[Item, Optional[Item]] = {}
    for t in tasks:
        _status(t, bad, planned, set(), contaminated - bad - set(planned))
        planned[t.item] = None
    if dry_run:
        return result

    # Quarantine every old target first: no rewrite can retrieve one.
    shim = ExciseResult(store, record_id, "quarantine", False, result.targets)
    _quarantine(ledger, old, shim)
    lost: set = set()  # items with no replacement (dropped or failed)
    for n, task in enumerate(tasks):
        _status(task, bad, result.replaced, lost,
                contaminated - bad - set(result.replaced) - lost)
        try:
            with agent_task_scope(agent_id, f"regenerate-{n}"):
                new = rewrite(task)
        except Exception as exc:
            result.failed.append(Failed(*task.item, f"rewrite raised {type(exc).__name__}: {exc}"))
            lost.add(task.item)
            continue
        if new is None:
            result.dropped.append(task.item)
            lost.add(task.item)
            continue
        new = (str(new[0]), str(new[1]))
        problem = _check(ledger, new, contaminated)
        if problem:
            result.failed.append(Failed(*task.item, problem))
            lost.add(task.item)
            if new not in contaminated:
                ledger.quarantine([new], reason="failed regeneration of " + repr(task.item))
            continue
        result.replaced[task.item] = new

    if then == "quarantine":
        result.done, result.skipped = shim.done, shim.skipped
    else:
        dele = ExciseResult(store, record_id, "delete", False, result.targets)
        _delete(stores, old, dele)
        result.done, result.skipped = dele.done, dele.skipped
    return result


def _plan(ledger: Ledger, blast: BlastResult, root: Item) -> Tuple[List[RegenerateTask], set]:
    """Tasks for the derived items, and the items that carry the bad content
    itself (bound to the root's own writes)."""
    bad = {root}
    for w in blast.items:
        bad.update(blast.bindings.get(w.write_id, ()))
    writes: Dict[Item, list] = {}
    for w in blast.descendants:
        for item in blast.bindings.get(w.write_id, ()):
            if item not in bad:
                writes.setdefault(item, []).append(w)
    parents = _parents(ledger, [w.write_id for ws in writes.values() for w in ws])
    tasks = []
    for item, ws in writes.items():
        ws.sort(key=lambda w: w.seq)
        edges: Dict[Item, str] = {}
        for w in ws:
            for s, r, tag in parents.get(w.write_id, ()):
                if (s, r) != item and edges.get((s, r)) != "in_prompt":
                    edges[(s, r)] = tag
        tasks.append((ws[-1].seq, RegenerateTask(
            item[0], item[1], [w.write_id for w in ws], min(w.depth for w in ws),
            [Parent(s, r, t, "clean", None) for (s, r), t in sorted(edges.items())],
        )))
    tasks.sort(key=lambda x: x[0])
    return _topological([t for _, t in tasks]), bad


def _topological(tasks: List[RegenerateTask]) -> List[RegenerateTask]:
    """``tasks`` (in ``seq`` order) reordered so each comes after the tasks
    for its parents. A cycle (an item later compacted with its own
    descendant as a parent) is broken at the edge back to the task already
    being visited; that parent is then ``dropped`` for the task."""
    by_item = {t.item: t for t in tasks}
    out: List[RegenerateTask] = []
    state: Dict[Item, int] = {}  # 1 visiting, 2 done

    def visit(t: RegenerateTask) -> None:
        stack = [(t, iter(t.parents))]
        state[t.item] = 1
        while stack:
            cur, it = stack[-1]
            for p in it:
                dep = by_item.get((p.store, p.record_id))
                if dep is not None and dep.item not in state:
                    state[dep.item] = 1
                    stack.append((dep, iter(dep.parents)))
                    break
            else:
                stack.pop()
                state[cur.item] = 2
                out.append(cur)

    for t in tasks:
        if t.item not in state:
            visit(t)
    return out


def _status(task: RegenerateTask, bad: set, replaced: Mapping[Item, Optional[Item]], lost: set,
            pending: set) -> None:
    """Set each parent's status from what has happened so far. ``pending``
    are targets with no replacement yet (not rewritten so far, or in a
    cycle): a parent among them is ``dropped`` for this task."""
    out = []
    for p in task.parents:
        item = (p.store, p.record_id)
        if item in bad:
            out.append(p._replace(status="bad", replacement=None))
        elif item in replaced:
            out.append(p._replace(status="replaced", replacement=replaced[item]))
        elif item in lost or item in pending:
            out.append(p._replace(status="dropped", replacement=None))
        else:
            out.append(p._replace(status="clean", replacement=None))
    task.parents = out


def _parents(ledger: Ledger, write_ids: List[str]) -> Dict[str, List[Tuple[str, str, str]]]:
    out: Dict[str, List[Tuple[str, str, str]]] = {}
    ids = sorted(set(write_ids))
    for i in range(0, len(ids), _CHUNK):
        chunk = ids[i:i + _CHUNK]
        with ledger._lock:
            rows = ledger._conn.execute(
                "SELECT write_id, parent_store, parent_record_id, tag FROM parents"
                f" WHERE write_id IN ({','.join('?' * len(chunk))})",
                chunk,
            ).fetchall()
        for w, s, r, t in rows:
            out.setdefault(w, []).append((s, r, t))
    return out


def _check(ledger: Ledger, new: Item, contaminated: set) -> Optional[str]:
    """Why ``new`` isn't a clean replacement, or None if it is."""
    if new in contaminated:
        return f"replacement {new!r} is a target (rewrite as a new record, not in place)"
    with ledger._lock:
        row = ledger._conn.execute(
            "SELECT write_id FROM bindings WHERE store = ? AND record_id = ?"
            " ORDER BY rowid DESC LIMIT 1",
            new,
        ).fetchone()
    if row is None:
        return (f"replacement {new!r} has no recorded write"
                " (write it through an instrumented store)")
    for s, r, _ in _parents(ledger, [row[0]]).get(row[0], ()):
        if (s, r) in contaminated:
            return f"replacement {new!r} was derived from target {(s, r)!r}"
    return None
