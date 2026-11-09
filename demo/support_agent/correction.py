"""The console's side of ``dod.admin``: its answers, drawn as boxes.

The support lead's chat runs ``dod.admin`` (``World.admin``): every
``@command`` goes to ``Admin.handle`` except ``@excise``, which the console
draws richer: the records the model matched (ticked where they contradict
the statement, which the lead can change), the plan for what's ticked, then
apply. This module only maps items to the console's boxes and labels; the
finding, planning and fixing are the library's.
"""

from typing import List, Optional

from dod.admin import GROUPS

from .lineage import Lineage
from .world import World

Item = tuple


def _box(lin: Lineage, item) -> Optional[str]:
    try:
        return lin.box_for(*item)
    except KeyError:
        return None


def _boxes(lin: Lineage, item) -> List[str]:
    """Every box drawing ``item``, oldest turn first."""
    return sorted((bid for bid, b in lin.boxes.items()
                   if (b["item"] and tuple(b["item"]) == tuple(item))
                   or (b["kind"] == "handbook" and tuple(item) in {tuple(i) for i in b["items"]})),
                  key=lambda bid: lin.boxes[bid]["turn"])


def _entry(lin: Lineage, item, **extra) -> dict:
    b = _box(lin, item)
    return {"item": list(item), "box": b, "label": lin.label(*item),
            "turn": lin.boxes[b]["turn"] if b else None, **extra}


def candidates(w: World, text: str) -> dict:
    """What ``Admin.find`` matched: contradicting records first."""
    match, recs = w.admin.find(text)
    if not match.about:
        return {"candidates": []}
    lin = Lineage(w)
    lin.view()
    content = {r.item: r.text for r in recs}
    out = [_entry(lin, i, content=content.get(i, ""), conflicts=i in match.contradicts)
           for i in match.about]
    out.sort(key=lambda c: (not c["conflicts"], c["label"]))
    return {"candidates": out}


def plan(w: World, text: str, items: List[list]) -> dict:
    """``Admin.plan`` for the ticked items, as boxes."""
    items = [tuple(i) for i in items]
    if not items:
        return {"error": "nothing ticked"}
    p = w.admin.plan(text, items)
    w.admin.pending.pop(p.id, None)  # the console re-plans on every tick
    lin = Lineage(w)
    view = lin.view()
    groups = {g: [] for g in GROUPS}
    seen = set()
    for g in GROUPS:
        for item in p.groups[g]:
            # A record can be drawn in several turns (a thread's state);
            # records the console doesn't draw (tool results) are only counted.
            for bid in _boxes(lin, item):
                if bid in seen:
                    continue
                seen.add(bid)
                b = lin.boxes[bid]
                groups[g].append({"item": list(item), "box": bid, "label": b["label"],
                                  "turn": b["turn"], "preview": b["preview"]})
    touched = {b for b in seen}
    untouched = [b["label"] for b in view["boxes"]
                 if b["status"] == "live" and b["id"] not in touched
                 and not any(tuple(i) in set(p.roots) | {x for g in p.groups.values() for x in g}
                             for i in lin._items_of(b["id"]))]
    customers = sorted({e["label"].split(" (")[0].replace("Reply to ", "").lower()
                        for e in groups["notify"]})
    return {"roots": [dict(_entry(lin, r), preview=lin.boxes[_box(lin, r)]["preview"]
                           if _box(lin, r) else "") for r in p.roots],
            "groups": groups, "customers": customers,
            "untouched": p.untouched, "untouched_labels": untouched,
            "affected": p.affected, "wipe": p.affected + p.untouched,
            "code": [f"plan = admin.plan({text!r}, wrong)", "admin.apply(plan)"]
            + [f"# regenerates from {r[0]} {r[1]}" for r in p.roots]}


async def apply(w: World, text: str, items: List[list]) -> dict:
    """``Admin.apply``, through ``World.correct`` (which records the
    correction's turn); returns the corrected replies."""
    out = await w.correct(text, [tuple(i) for i in items])
    a = out["applied"]
    pairs = [w.reply_pair(old[1], new[1]) for old, new in a.replaced.items()
             if old[0] == "mcp:crm"]
    return {"correction": a.correction[1], "cancelled": [i[1] for i in a.cancelled],
            "pairs": pairs, "failed": [[list(i), why] for i, why in a.failed],
            "quarantined": len(a.quarantined),
            "roots": [_entry(Lineage(w), r) for r in a.plan.roots]}


def command(w: World, text: str) -> dict:
    """Any other ``@command``: ``Admin.handle``, with the items it names
    mapped to boxes so the console can highlight them."""
    r = w.admin.handle(text, user="support-lead")
    lin = Lineage(w)
    d = r.data
    focus = None
    if "origins" in d:
        focus = {"kind": "trace", "box": _box(lin, d["item"])}
    elif "sources" in d:
        focus = {"kind": "blast", "box": _box(lin, d["sources"][0])}
    return {"text": r.text, "code": r.code, "focus": focus,
            "choices": [list(c) for c in r.choices]}
