"""What the console draws: the ledger, organized by turn.

A *turn* is one unit of work: the handbook seed, or one customer message
through the agent (``World.turns``, by ledger ``seq`` range). A *box* is one
version of an item, written in one turn:

- conversation: the customer's message (the turn's user-sourced state
  write). The agent's own update to the thread state is folded into it.
- the handbook: the six seeded policies, shown as one box.
- every other write the agent made: notes, policy files, tickets, replies.

A *read* edge says a turn had a box in its context: for each write in the
turn, each parent item recorded by dod is resolved to its latest version
written before the turn began. Writes a turn reads from itself (a reply that
saw the ticket written a moment earlier) aren't drawn: within a turn,
everything comes from the message and the reads.

``trace`` follows read edges backward (turn by turn); ``blast`` is dod's
``blast_radius`` (what ``excise`` would quarantine), mapped onto boxes.
"""

import json
import sqlite3
from typing import Dict, List, Optional, Set, Tuple

import dod
from dod.query import blast_radius

from .world import POLICIES, World

COLUMNS = ["langgraph_state", "memory", "notes", "mcp:crm"]
HANDBOOK = "handbook"

Item = Tuple[str, str]


class Lineage:
    def __init__(self, world: World) -> None:
        self.w = world
        self._build()

    # --- the model -------------------------------------------------------

    def _build(self) -> None:
        w = self.w
        conn = sqlite3.connect(f"file:{w.dir / 'ledger.db'}?mode=ro", uri=True)
        try:
            writes = conn.execute(
                "SELECT w.seq, w.write_id, w.store, w.source, b.store, b.record_id"
                " FROM writes w JOIN bindings b ON b.write_id = w.write_id ORDER BY w.seq"
            ).fetchall()
            parents: Dict[str, List[Tuple[str, str, str]]] = {}
            for wid, s, r, tag in conn.execute(
                    "SELECT write_id, parent_store, parent_record_id, tag FROM parents"):
                parents.setdefault(wid, []).append((s, r, tag))
            tombstoned = {(s, r) for s, r in conn.execute(
                "SELECT store, record_id FROM tombstones")}
        finally:
            conn.close()
        quarantined = w.ledger.quarantined()

        self.turns = [dict(t) for t in w.turns]
        bounds = [(t["start"], t["end"], t["id"]) for t in self.turns]

        def turn_of(seq: int) -> Optional[int]:
            for lo, hi, tid in bounds:
                if lo < seq <= hi:
                    return tid
            return None

        self.boxes: Dict[str, dict] = {}
        alias: Dict[str, Optional[str]] = {}  # write_id -> box id (None: not drawn)
        versions: Dict[Item, List[Tuple[int, str]]] = {}  # item -> [(seq, write_id)]
        turn_writes: Dict[int, List[str]] = {}
        message_box: Dict[int, str] = {}
        later: List[Tuple[str, int]] = []  # agent updates to state, folded afterwards

        for seq, wid, store, source, bstore, rid in writes:
            item = (bstore, rid)
            versions.setdefault(item, []).append((seq, wid))
            tid = turn_of(seq)
            if tid is None:
                alias[wid] = None
                continue
            turn_writes.setdefault(tid, []).append(wid)
            if rid.startswith("call:"):
                alias[wid] = None  # a read-only tool result: part of its turn
            elif bstore == "langgraph_state":
                if source == "user":
                    bid = f"msg:{tid}"
                    message_box[tid] = bid
                    self._box(bid, tid, item, "message", source, quarantined, tombstoned)
                    alias[wid] = bid
                else:
                    later.append((wid, tid))
            elif bstore == w.memory_store and json.loads(rid)[0] == "policies":
                self._box(HANDBOOK, tid, None, "handbook", source, quarantined, tombstoned)
                self.boxes[HANDBOOK].setdefault("items", []).append(list(item))
                alias[wid] = HANDBOOK
            else:
                bid = f"w:{wid}"
                self._box(bid, tid, item, self._kind(item), source, quarantined, tombstoned)
                alias[wid] = bid
        for wid, tid in later:
            alias[wid] = message_box.get(tid)

        # Reads: a turn's context, resolved to versions from before it began.
        self.reads: Dict[int, Dict[str, str]] = {}
        for t in self.turns:
            if t["kind"] == "seed":
                continue
            edges: Dict[str, str] = {}
            for wid in turn_writes.get(t["id"], ()):
                for s, r, tag in parents.get(wid, ()):
                    before = [v for v in versions.get((s, r), ()) if v[0] <= t["start"]]
                    if not before:
                        continue  # written earlier in this same turn
                    bid = alias.get(before[-1][1])
                    if bid is None:
                        continue
                    if edges.get(bid) != "in_prompt":
                        edges[bid] = tag
            self.reads[t["id"]] = edges
        for t in self.turns:
            t["message_box"] = message_box.get(t["id"])

    def _kind(self, item: Item) -> str:
        store, rid = item
        if store == "mcp:crm":
            return "reply" if rid.startswith("R") else "ticket"
        if store == "notes":
            return "policy_note"
        return "customer_note"

    def _box(self, bid, tid, item, kind, source, quarantined, tombstoned) -> None:
        if bid in self.boxes:
            return
        box = {"id": bid, "turn": tid, "kind": kind, "source": source,
               "store": item[0] if item else self.w.memory_store,
               "item": list(item) if item else None}
        if item:
            box["quarantined"] = item in quarantined
            box["deleted"] = item in tombstoned
        else:  # the handbook: quarantined if any policy is (set in view())
            box["quarantined"] = box["deleted"] = False
        self.boxes[bid] = box

    # --- what the UI gets --------------------------------------------------

    def view(self) -> dict:
        w = self.w
        q = w.ledger.quarantined()
        turns_by_id = {t["id"]: t for t in self.turns}
        out_boxes = []
        for b in self.boxes.values():
            t = turns_by_id[b["turn"]]
            if b["kind"] == "handbook":
                b["quarantined"] = any(tuple(i) in q for i in b.get("items", ()))
                b["label"] = f"Company policies ({len(POLICIES)})"
                b["preview"] = " · ".join(POLICIES.values())
            elif b["kind"] == "message":
                b["label"] = (f"{t['customer'].capitalize()} asks again" if t["kind"] == "regenerated"
                              else f"{t['customer'].capitalize()} says")
                b["preview"] = t["message"]
                b["reply"] = next((e["reply"] for e in w.log if e["turn"] == t["id"]), "")
            else:
                b["label"] = self.label(*b["item"])
                b["preview"] = self.preview(*b["item"])
            out_boxes.append(b)
        return {
            "columns": COLUMNS,
            "turns": [{k: t.get(k) for k in ("id", "kind", "customer", "message", "reply_id",
                                              "replaces", "message_box")} for t in self.turns],
            "boxes": out_boxes,
            "reads": [{"from": bid, "turn": tid, "tag": tag}
                      for tid, edges in self.reads.items() for bid, tag in edges.items()],
        }

    def label(self, store: str, rid: str) -> str:
        if store == "mcp:crm":
            return {"R": "Reply ", "T": "Ticket "}.get(rid[:1], "") + rid
        if store == "notes":
            return rid.rsplit("/", 1)[-1]
        parts = json.loads(rid)
        return f"Note on {parts[1]}"

    def preview(self, store: str, rid: str) -> str:
        w = self.w
        try:
            if store == w.memory_store:
                from langgraph.store.base import GetOp

                *ns, key = json.loads(rid)
                base = type(w.memory)._dod_base
                (item,) = base.batch(w.memory, [GetOp(tuple(ns), key, refresh_ttl=False)])
                return item.value.get("text", "") if item else "(deleted)"
            if store == "notes":
                p = w.dir / "notes" / rid.lstrip("/")
                return p.read_text().strip() if p.exists() else "(deleted)"
            if store == "mcp:crm":
                db = sqlite3.connect(w._crm_db)
                try:
                    if rid.startswith("R"):
                        row = db.execute("SELECT customer, text FROM replies WHERE id = ?",
                                         (rid,)).fetchone()
                        return f"To {row[0]}: {row[1]}" if row else "(deleted)"
                    row = db.execute("SELECT action, order_id, summary FROM tickets WHERE id = ?",
                                     (rid,)).fetchone()
                    return f"{row[0]} for order {row[1]}: {row[2]}" if row else "(deleted)"
                finally:
                    db.close()
        except Exception as exc:  # a preview never breaks the console
            return f"(unavailable: {type(exc).__name__})"
        return ""

    # --- trace and blast -----------------------------------------------------

    def trace(self, box_id: str) -> dict:
        """Everything ``box_id`` was derived from, turn by turn, back to the
        origins: what customers said, and the handbook."""
        box = self.boxes[box_id]
        if box["kind"] in ("message", "handbook"):
            return {"boxes": [box_id], "turns": [box["turn"]], "reads": [],
                    "origins": [box_id], "self_origin": True}
        boxes: Set[str] = {box_id}
        edges: Set[Tuple[str, int]] = set()
        seen: Set[int] = set()
        stack = [box["turn"]]
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            msg = next((t["message_box"] for t in self.turns if t["id"] == tid), None)
            if msg:
                boxes.add(msg)
            for bid in self.reads.get(tid, {}):
                boxes.add(bid)
                edges.add((bid, tid))
                stack.append(self.boxes[bid]["turn"])
        origins = [b for b in boxes if self.boxes[b]["kind"] in ("message", "handbook")]
        return {"boxes": sorted(boxes), "turns": sorted(seen),
                "reads": [list(e) for e in sorted(edges)], "origins": origins,
                "self_origin": False}

    def blast(self, box_id: str) -> dict:
        """What dod's ``blast_radius`` reaches from the box's item (what
        ``excise`` would quarantine), drawn as boxes."""
        items = self._items_of(box_id)
        reached: Set[Item] = set(items)
        for item in items:
            b = blast_radius(self.w.ledger, *item)
            reached |= {i for its in b.bindings.values() for i in its}
        boxes = {bid for bid, b in self.boxes.items()
                 if (b["item"] and tuple(b["item"]) in reached)
                 or (b["kind"] == "handbook" and any(tuple(i) in reached for i in b["items"]))}
        boxes.add(box_id)
        turns = {self.boxes[b]["turn"] for b in boxes if b != box_id}
        edges = [[bid, tid] for tid, e in self.reads.items() if tid in turns
                 for bid in e if bid in boxes]
        per: Dict[str, int] = {}
        for b in boxes - {box_id}:
            s = self.boxes[b]["store"]
            per[s] = per.get(s, 0) + 1
        return {"boxes": sorted(boxes), "turns": sorted(turns), "reads": edges,
                "per_store": per}

    def _items_of(self, box_id: str) -> List[Item]:
        b = self.boxes[box_id]
        if b["kind"] == "handbook":
            return [tuple(i) for i in b["items"]]
        return [tuple(b["item"])]

    def compare(self, box_id: str) -> dict:
        """Excising ``box_id``'s item vs. wiping every store, counted in live
        records (items), with the boxes each keeps and loses."""
        self.view()  # labels
        hit: Set[Item] = set()
        for item in self._items_of(box_id):
            r = dod.excise(self.w.ledger, *item, dry_run=True)
            hit |= {(s, rid) for s, rids in r.targets.items() for rid in rids}
        live: Set[Item] = set()
        kept, lost = [], []
        for b in self.boxes.values():
            if b["deleted"] or b["quarantined"]:
                continue
            items = set(self._items_of(b["id"]))
            live |= items
            (lost if items & hit else kept).append(b["label"])
        return {"live": len(live), "excise": len(live & hit), "kept": len(live - hit),
                "kept_labels": kept, "lost_labels": lost}

    def box_for(self, store: str, rid: str) -> Optional[str]:
        for bid, b in self.boxes.items():
            if b["item"] == [store, rid]:
                return bid
        return None
