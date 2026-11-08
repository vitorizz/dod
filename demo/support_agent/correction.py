"""Correcting a wrong fact: how the demo's cleanup is meant to be used.

Cleanup starts when a person finds a mistake, not from a button on every
record. A support lead states the correction ("The refund window is 30 days,
not 90."), and the console goes from there:

1. ``candidates``: the agent's model reads the correction and its beliefs
   (policy notes, customer notes, the handbook) and picks those about the
   same fact, marking the ones that contradict it (the scripted model
   matches keywords and numbers). The person confirms which are wrong:
   neither dod nor the model decides that.
2. ``plan``: trace each confirmed belief to its origin (a customer message
   that states the wrong number too, or else the belief itself), then
   group everything built on it by what to do:

   - fix: wrong notes stop being read, and the correction is written as a
     policy note in their place;
   - cancel: open tickets, so fulfilment doesn't act on them;
   - notify: replies sent to customers stay on record, and each customer
     gets a correction;
   - on record: conversation state, kept but hidden from the agent;
   - untouched: everything else.

3. ``apply``: record the correction, write its note, cancel the tickets,
   then ``dod.regenerate`` from each origin, which quarantines what it
   reached (reversible: nothing is deleted) and sends each corrected reply.
   Everything the correction writes has the correction as its parent.
"""

import re
from typing import Dict, List, Optional

from .lineage import Lineage
from .world import POLICIES, TOPICS, World

BELIEFS = ("policy_note", "customer_note", "handbook")
GROUPS = {"policy_note": "fix", "customer_note": "fix", "handbook": "fix",
          "ticket": "cancel", "reply": "notify", "message": "record"}


def _days(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*days?", text or "")
    return int(m.group(1)) if m else None


def topic_of(text: str) -> Optional[str]:
    low = text.lower()
    return next((t for t, words in TOPICS.items() if any(w in low for w in words)), None)


def _conflicts(text: str, correction: str) -> bool:
    """States a number of days other than the correction's (used to find
    the origin message; the beliefs themselves are matched by the model)."""
    want, got = _days(correction), _days(text)
    return want is not None and got is not None and got != want


def _live(b: dict) -> bool:
    return not (b.get("quarantined") or b.get("deleted"))


def records(lin: Lineage) -> List[dict]:
    """The agent's live beliefs, one per record: each handbook policy,
    policy note and customer note, with the box that draws it."""
    out = []
    for b in lin.boxes.values():
        if b["kind"] not in BELIEFS or not _live(b):
            continue
        if b["kind"] == "handbook":
            out += [{"id": f"handbook:{k}", "box": b["id"], "label": f"Company policy: {k}",
                     "kind": "handbook", "store": b["store"], "topic": k, "content": v}
                    for k, v in POLICIES.items()]
        else:
            topic = b["item"][1].split("/")[-2] if b["kind"] == "policy_note" else None
            out.append({"id": b["id"], "box": b["id"], "label": b["label"], "kind": b["kind"],
                        "store": b["store"], "topic": topic, "content": b["preview"]})
    return out


def candidates(w: World, text: str) -> dict:
    """The records about the correction's fact, as the agent's model reads
    them (the scripted model: by keyword and number); the ones it says
    contradict the correction first. The person confirms."""
    lin = Lineage(w)
    lin.view()
    recs = records(lin)
    m = w.llm.match(text, recs, TOPICS)
    if m["topic"] is None:
        return {"topic": None, "candidates": [],
                "error": "Couldn't tell what this is about; name the policy (refunds,"
                         " shipping or warranty)."}
    out = [dict(r, conflicts=r["id"] in m["contradicts"]) for r in recs if r["id"] in m["relevant"]]
    out.sort(key=lambda c: (not c["conflicts"], c["label"]))
    return {"topic": m["topic"], "candidates": out}


def plan(w: World, text: str, boxes: List[str], topic: Optional[str] = None) -> dict:
    """What applying the correction to the confirmed ``boxes`` would do."""
    lin = Lineage(w)
    view = lin.view()
    by_id = {b["id"]: b for b in view["boxes"]}
    boxes = list(dict.fromkeys(boxes))  # handbook policies share one box
    unknown = [b for b in boxes if b not in by_id]
    if unknown or not boxes:
        return {"error": f"unknown or no records: {unknown}"}
    roots: List[str] = []
    sources: Dict[str, List[str]] = {}
    for bid in boxes:
        t = lin.trace(bid)
        bad = [o for o in t["origins"]
               if by_id[o]["kind"] == "message" and _conflicts(by_id[o]["preview"], text)]
        sources[bid] = bad or [bid]
        for r in sources[bid]:
            if r not in roots:
                roots.append(r)
    reached = set()
    for r in roots:
        reached |= set(lin.blast(r)["boxes"])
    groups: Dict[str, List[dict]] = {g: [] for g in ("fix", "cancel", "notify", "record")}
    for bid in sorted(reached, key=lambda b: (by_id[b]["turn"], b)):
        b = by_id[bid]
        if not _live(b) or b["kind"] not in GROUPS:
            continue
        if b["kind"] == "ticket" and b["status"] == "cancelled":
            continue
        groups[GROUPS[b["kind"]]].append({"box": bid, "label": b["label"],
                                          "preview": b["preview"], "turn": b["turn"]})
    customers = sorted({i["preview"].split(":")[0].replace("To ", "")
                        for i in groups["notify"]})
    c = lin.compare(roots)
    return {"topic": topic or topic_of(text), "roots": [{"box": r, "label": by_id[r]["label"],
                                                 "preview": by_id[r]["preview"]} for r in roots],
            "sources": sources, "groups": groups, "customers": customers,
            "untouched": c["kept"], "untouched_labels": c["kept_labels"],
            "wipe": c["live"], "affected": c["excise"]}


async def apply(w: World, text: str, boxes: List[str], topic: Optional[str] = None) -> dict:
    """Apply the plan for ``boxes``; returns the corrected replies."""
    p = plan(w, text, boxes, topic)
    if "error" in p:
        return p
    lin = Lineage(w)
    tickets = [lin.boxes[i["box"]]["item"][1] for i in p["groups"]["cancel"]]
    cid, note = w.record_correction(text, p["topic"] or "general")
    await w.cancel_tickets(cid, tickets)
    pairs, failed = [], []
    for r in p["roots"]:
        for item in lin._items_of(r["box"]):
            out = await w.regenerate(*item, correction=(cid, text))
            failed += out["failed"]
            pairs += [w.reply_pair(old[1], new[1]) for old, new in out["replaced"]
                      if old[0] == "mcp:crm"]
    return {"correction": cid, "note": note, "cancelled": tickets, "pairs": pairs,
            "failed": failed, "plan": p}
