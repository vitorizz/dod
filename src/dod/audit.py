"""Audit: verify the ledger hasn't been tampered with, and report deletions.

Every transaction that appends to the ledger also appends an entry to its
audit chain (``audit_chain``): the id ranges it added to each table, and the
sha256 of the previous entry's digest plus all of those rows. ``verify``
recomputes the chain from the rows on disk:

- a row edited or deleted (the schema's triggers forbid both, but anyone
  with the file can drop a trigger) changes its entry's digest;
- an entry edited, deleted or reordered breaks the ``prev`` links;
- a row inserted outside the library has no entry: it's reported as
  unchained.

A file created before the chain existed (schema 5) gets one sealing entry
covering all of its rows when it is migrated: edits made before that can't be
detected, edits after can.

Removing the newest entries together with their rows can't be detected from
the file alone. Anchor the chain outside it: ``head()`` is the newest digest;
log it, export it with your audit trail, or record it with a third party,
and a later ``verify(expected_head=...)`` proves nothing was cut off.

``report(ledger, store, record_id)`` is the evidence for one excision: what
was quarantined or deleted (the item and everything derived from it), when,
why and by whom, and the chain entries that recorded each action. It holds
IDs, hashes and timestamps only, like the ledger.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dod.ledger import _CHAINED, Ledger, _chain_digest, _context_set
from dod.query import _walk_connection, blast_radius


@dataclass
class VerifyReport:
    ok: bool
    entries: int
    head: Optional[str]
    """The newest entry's digest: anchor it outside the file."""
    problems: List[str] = field(default_factory=list)
    """Each broken link, digest mismatch, overlap or unchained row range."""


def head(ledger: Ledger) -> Optional[str]:
    """The newest audit chain digest, or None for an empty chain."""
    conn, release = _walk_connection(ledger)
    try:
        row = conn.execute("SELECT digest FROM audit_chain ORDER BY seq DESC LIMIT 1").fetchone()
    finally:
        release()
    return row[0] if row else None


def verify(ledger: Ledger, expected_head: Optional[str] = None) -> VerifyReport:
    """Recompute the audit chain from the rows on disk (one read snapshot).

    ``expected_head``, a digest from an earlier ``head()``, must still be in
    the chain: otherwise entries were removed or the file was replaced.
    """
    conn, release = _walk_connection(ledger)
    problems: List[str] = []
    try:
        entries = conn.execute(
            "SELECT seq, digest, prev, w_lo, w_hi, s_lo, s_hi, b_lo, b_hi, t_lo, t_hi,"
            " q_lo, q_hi FROM audit_chain ORDER BY seq").fetchall()
        prev = None
        covered: Dict[str, List[Tuple[int, int, int]]] = {k: [] for _, _, k in _CHAINED}
        for seq, digest, p, *bounds in entries:
            if p != prev:
                problems.append(f"entry {seq}: link broken (its prev is not entry {seq - 1}'s digest)")
            ranges = {}
            for i, (_, _, k) in enumerate(_CHAINED):
                lo, hi = bounds[2 * i], bounds[2 * i + 1]
                if lo is not None:
                    ranges[k] = (lo, hi)
                    covered[k].append((lo, hi, seq))
            if _chain_digest(conn, p, ranges) != digest:
                problems.append(f"entry {seq}: rows changed (digest mismatch)")
            prev = digest
        for table, col, k in _CHAINED:
            (top,) = conn.execute(f"SELECT coalesce(max({col}), 0) FROM {table}").fetchone()
            spans = sorted(covered[k])
            at = 0
            for lo, hi, seq in spans:
                if lo > at:
                    _unchained(conn, table, col, at, lo, problems)
                elif lo < at:
                    problems.append(f"entry {seq}: {table} ids ({lo}, {hi}] overlap an earlier entry")
                at = max(at, hi)
            if top > at:
                _unchained(conn, table, col, at, top, problems)
        _check_sets(conn, problems)
        if expected_head is not None and expected_head not in {e[1] for e in entries}:
            problems.append("expected head is not in the chain: entries were removed or the file replaced")
    finally:
        release()
    return VerifyReport(not problems, len(entries), prev, problems)


def _check_sets(conn, problems) -> None:
    """Each context set's hash must be the address of its members (merged
    over its chain's parts), and its size their count."""
    members: Dict[str, Dict[Tuple[str, str], str]] = {}
    for set_hash, size, store, record_id, tag in conn.execute(
            "SELECT s.set_hash, s.size, m.store, m.record_id, m.tag FROM context_sets s"
            " JOIN context_chain c ON c.set_hash = s.set_hash"
            " JOIN context_members m ON m.set_hash = c.part ORDER BY s.seq"):
        merged = members.setdefault(set_hash, {"": size})
        if merged.get((store, record_id)) != "in_prompt":
            merged[(store, record_id)] = tag
    for (set_hash,) in conn.execute("SELECT set_hash FROM context_sets"):
        merged = members.pop(set_hash, None)
        size = merged.pop("") if merged else None
        address = _context_set([(s, r, t) for (s, r), t in (merged or {}).items()])[1]
        if address != set_hash or (merged and size != len(merged)):
            problems.append(f"context set {set_hash[:12]}: members changed (address mismatch)")


def _unchained(conn, table, col, lo, hi, problems) -> None:
    (n,) = conn.execute(f"SELECT count(*) FROM {table} WHERE {col} > ? AND {col} <= ?",
                        (lo, hi)).fetchone()
    if n:
        problems.append(f"{table}: {n} row(s) with {col} in ({lo}, {hi}] are in no chain entry"
                        " (appended outside the library)")


def report(ledger: Ledger, store: str, record_id: str) -> dict:
    """Evidence for the excision of ``(store, record_id)``: for the item and
    every item derived from it, its quarantine actions and deletions
    (tombstones) with time, reason and actor, and the audit chain entry that
    recorded each; plus the chain's verification at report time.

    Raises ``KeyError`` if the item has no binding in the ledger.
    """
    blast = blast_radius(ledger, store, record_id)
    items = sorted({(store, record_id)} | {i for its in blast.bindings.values() for i in its})
    conn, release = _walk_connection(ledger)
    try:
        def entry(k: str, ident: int) -> Optional[dict]:
            row = conn.execute(
                f"SELECT seq, digest FROM audit_chain WHERE {k}_lo < ? AND {k}_hi >= ?",
                (ident, ident)).fetchone()
            return {"seq": row[0], "digest": row[1]} if row else None

        out = []
        for s, r in items:
            q = [{"action": a, "ts": ts, "reason": reason, "actor": actor,
                  "chain": entry("q", seq)}
                 for seq, ts, a, reason, actor in conn.execute(
                     "SELECT seq, ts, action, reason, actor FROM quarantine"
                     " WHERE store = ? AND record_id = ? ORDER BY seq", (s, r))]
            t = [{"reason": reason, "ts": ts, "write_id": w, "chain": entry("t", seq)}
                 for seq, ts, reason, w in conn.execute(
                     "SELECT seq, ts, reason, write_id FROM tombstones"
                     " WHERE store = ? AND record_id = ? ORDER BY seq", (s, r))]
            writes = [{"write_id": w, "content_hash": h, "ts": ts}
                      for w, h, ts in conn.execute(
                          "SELECT w.write_id, w.content_hash, w.ts FROM bindings b"
                          " JOIN writes w ON w.write_id = b.write_id"
                          " WHERE b.store = ? AND b.record_id = ? ORDER BY w.seq", (s, r))]
            last_q = q[-1]["action"] if q else None
            out.append({"store": s, "record_id": r, "writes": writes, "quarantine": q,
                        "deletions": t,
                        "status": "deleted" if t else "quarantined" if last_q == "quarantine"
                        else "released" if last_q == "release" else "live"})
    finally:
        release()
    v = verify(ledger)
    return {"item": [store, record_id], "items": out,
            "summary": {k: sum(1 for i in out if i["status"] == k)
                        for k in ("deleted", "quarantined", "released", "live")},
            "chain": {"ok": v.ok, "head": v.head, "entries": v.entries, "problems": v.problems}}


def to_markdown(rep: dict) -> str:
    """``report`` as a readable document."""
    s, r = rep["item"]
    sm = rep["summary"]
    lines = [f"# Excision report: {s} {r}", "",
             f"{len(rep['items'])} item(s): the item and everything derived from it.",
             f"Deleted {sm['deleted']}, quarantined {sm['quarantined']},"
             f" released {sm['released']}, still live {sm['live']}.", "",
             f"Audit chain: {'verified' if rep['chain']['ok'] else 'NOT VERIFIED'},"
             f" {rep['chain']['entries']} entries, head `{rep['chain']['head']}`."]
    lines += [f"- {p}" for p in rep["chain"]["problems"]]
    lines += ["", "| Store | Record | Status | Actions (ns since epoch) |", "|---|---|---|---|"]
    for i in rep["items"]:
        acts = [f"{a['action']} {a['ts']} ({a['reason'] or ''}{', ' + a['actor'] if a['actor'] else ''})"
                f" [entry {a['chain']['seq'] if a['chain'] else '?'}]" for a in i["quarantine"]]
        acts += [f"deleted {d['ts']} ({d['reason']}) [entry {d['chain']['seq'] if d['chain'] else '?'}]"
                 for d in i["deletions"]]
        lines.append(f"| {i['store']} | `{i['record_id']}` | {i['status']} | {'<br>'.join(acts) or '-'} |")
    return "\n".join(lines) + "\n"
