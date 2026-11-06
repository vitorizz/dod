"""Excise an item and everything derived from it, across stores.

The caller names the item to excise; the library never decides which items
are bad. Targets are the item itself plus every item bound to a write in its
blast radius (``dod.query.blast_radius``), in whichever store it lives.

- ``mode="quarantine"`` (default, reversible): ``Ledger.quarantine`` for each
  target. No store is touched; instrumented stores stop returning the items
  until they are released.
- ``mode="delete"``: each target is deleted through its store's adapter, an
  object with ``delete(record_id) -> bool`` (True if something was deleted).
  The adapters in ``dod.integrations`` delete through the instrumented store,
  which records the tombstone with reason ``excision``.

Unrelated items are never touched. Re-running on the same item is
idempotent: already quarantined items and items no longer in their store are
reported in ``skipped``, and nothing new is recorded for them.
"""

import contextvars
import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, NamedTuple, Optional, Protocol

from dod.ledger import Item, Ledger
from dod.query import BlastResult, blast_radius

MODES = ("quarantine", "delete")

_excising: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "dod_excising", default=False
)


def deletion_reason() -> str:
    """Tombstone reason for a delete made now: ``excision`` inside
    ``excise``, ``forget`` otherwise. Read by the integrations."""
    return "excision" if _excising.get() else "forget"


class DeleteAdapter(Protocol):
    def delete(self, record_id: str) -> bool:
        """Delete ``record_id`` from the store; False if it wasn't there."""


class Skipped(NamedTuple):
    store: str
    record_id: str
    reason: str


@dataclass
class ExciseResult:
    store: str
    record_id: str
    mode: str
    dry_run: bool
    targets: Dict[str, List[str]]
    """Record IDs per store: the excised item and every item derived from it."""
    done: Dict[str, List[str]] = field(default_factory=dict)
    """Record IDs per store that were quarantined or deleted by this call.
    Empty on a dry run."""
    skipped: List[Skipped] = field(default_factory=list)
    """Targets left as they were, with the reason."""
    blast: Optional[BlastResult] = None
    """The blast radius the targets were taken from."""


def excise(
    ledger: Ledger,
    store: str,
    record_id: str,
    mode: str = "quarantine",
    stores: Optional[Mapping[str, DeleteAdapter]] = None,
    dry_run: bool = False,
    in_prompt_only: bool = False,
    precise: bool = False,
) -> ExciseResult:
    """Quarantine (default) or delete ``(store, record_id)`` and every item
    derived from it.

    ``stores`` maps store name to a delete adapter and is required for
    ``mode="delete"``; if any target's store has no adapter, ``ValueError``
    is raised before anything is deleted. ``dry_run=True`` returns the plan
    (``targets``) and changes nothing. ``in_prompt_only`` takes only what
    was derived through the model's prompt (``blast_radius``'s option of the
    same name): smaller, and precise when prompts are confirmed
    (``dod.prompt``), but it misses what an unconfirmed retrieval influenced.
    ``precise`` leaves out writes that only read an earlier, clean version of
    a contaminated item (``blast_radius``'s option of the same name).

    Raises ``KeyError`` if the item has no binding in the ledger.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if mode == "delete" and stores is None:
        raise ValueError('mode="delete" requires stores: {store name: delete adapter}')

    blast = blast_radius(ledger, store, record_id, in_prompt_only=in_prompt_only,
                         precise=precise)
    targets = _targets(blast, (store, record_id))
    if mode == "delete":
        missing = sorted(s for s in targets if s not in stores)
        if missing:
            raise ValueError(f"no delete adapter for store(s) {missing}; nothing was deleted")

    result = ExciseResult(store, record_id, mode, dry_run,
                          {s: [r for _, r in items] for s, items in targets.items()},
                          blast=blast)
    if dry_run:
        return result
    items = [i for its in targets.values() for i in its]
    if mode == "quarantine":
        _quarantine(ledger, items, result)
    else:
        _delete(stores, items, result)
    return result


def _targets(blast: BlastResult, root: Item) -> Dict[str, List[Item]]:
    """Target items grouped by store, root first, then by descendant depth."""
    order = [root]
    for w in blast.items + blast.descendants:
        order += blast.bindings.get(w.write_id, ())
    out: Dict[str, List[Item]] = {}
    for item in dict.fromkeys(order):
        out.setdefault(item[0], []).append(item)
    return dict(sorted(out.items()))


def _quarantine(ledger: Ledger, items: List[Item], result: ExciseResult) -> None:
    already = ledger.quarantined()
    todo = []
    for item in items:
        if item in already:
            result.skipped.append(Skipped(*item, "already quarantined"))
        else:
            todo.append(item)
    reason = "excision of " + json.dumps([result.store, result.record_id])
    ledger.quarantine(todo, reason=reason)
    _mark_done(result, todo)


def _delete(stores: Mapping[str, DeleteAdapter], items: List[Item],
            result: ExciseResult) -> None:
    done = []
    token = _excising.set(True)
    try:
        for s, r in items:
            try:
                deleted = stores[s].delete(r)
            except Exception as exc:  # report it and keep going: others may succeed
                result.skipped.append(Skipped(s, r, f"delete failed: {type(exc).__name__}: {exc}"))
                continue
            if deleted:
                done.append((s, r))
            else:
                result.skipped.append(Skipped(s, r, "not in store"))
    finally:
        _excising.reset(token)
    _mark_done(result, done)


def _mark_done(result: ExciseResult, items: List[Item]) -> None:
    for s, r in items:
        result.done.setdefault(s, []).append(r)
