"""``dod``: query and act on a ledger from the command line.

    dod trace        STORE RECORD_ID     where an item came from
    dod blast-radius STORE RECORD_ID     everything derived from it
    dod excise       STORE RECORD_ID     quarantine it and everything derived from it
    dod release      STORE RECORD_ID...  undo a quarantine
    dod quarantined                      what is quarantined now
    dod stats                            what the ledger holds

The ledger is ``--ledger PATH``, else ``$DOD_LEDGER``. ``--json`` prints
machine-readable output. Deleting through stores needs their adapters, so
the CLI only quarantines (reversible); use ``dod.excise(mode="delete")`` in
code to delete.
"""

import argparse
import json
import os
import sqlite3
import sys
from typing import List, Optional

import dod
from dod.ledger import Ledger, LedgerSchemaError, _check_schema
from dod.query import blast_radius, trace


def _path(args) -> str:
    path = args.ledger or os.environ.get("DOD_LEDGER")
    if not path:
        raise SystemExit("dod: no ledger given: pass --ledger PATH or set DOD_LEDGER")
    if not os.path.exists(path):
        raise SystemExit(f"dod: no ledger at {path}")
    return path


def _ledger(args) -> Ledger:
    path = _path(args)
    try:
        return Ledger(path)
    except LedgerSchemaError as exc:
        raise SystemExit(f"dod: {exc}")


def _out(args, data, text: List[str]) -> None:
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print("\n".join(text))


def _w(w) -> dict:
    return {"write_id": w.write_id, "depth": w.depth, "store": w.store, "source": w.source,
            "agent_id": w.agent_id, "session_id": w.session_id}


def cmd_trace(args) -> int:
    with _ledger(args) as led:
        try:
            r = trace(led, args.store, args.record_id, max_depth=args.max_depth)
        except KeyError:
            raise SystemExit(f"dod: no recorded write for {args.store} {args.record_id}")
        roots = [_w(w) for w in r.roots]
        data = {"item": [args.store, args.record_id], "ancestors": [_w(w) for w in r.ancestors],
                "roots": roots, "unrecorded": [[u.store, u.record_id] for u in r.unrecorded],
                "truncated": r.truncated,
                "deleted": {f"{s} {rid}": [t.reason for t in ts] for (s, rid), ts in r.deleted.items()}}
        lines = [f"{args.store} {args.record_id}: {len(r.ancestors)} ancestor write(s)"]
        for w in r.ancestors:
            lines.append(f"  {'  ' * (w.depth - 1)}<- {w.store} write {w.write_id} ({w.source})")
        lines.append("roots (where it starts):")
        lines += [f"  {w['store']} write {w['write_id']} (source: {w['source']})" for w in roots]
        if r.unrecorded:
            lines.append("parents with no recorded write (trace stops here):")
            lines += [f"  {u.store} {u.record_id}" for u in r.unrecorded]
        if r.truncated:
            lines.append("(stopped at --max-depth)")
        _out(args, data, lines)
    return 0


def cmd_blast(args) -> int:
    with _ledger(args) as led:
        try:
            b = blast_radius(led, args.store, args.record_id, max_depth=args.max_depth,
                             in_prompt_only=args.in_prompt_only)
        except KeyError:
            raise SystemExit(f"dod: no recorded write for {args.store} {args.record_id}")
        items = sorted({i for its in b.bindings.values() for i in its} - {(args.store, args.record_id)})
        data = {"item": [args.store, args.record_id], "descendants": len(b.descendants),
                "per_store": b.store_counts, "items": [list(i) for i in items],
                "in_prompt_only": args.in_prompt_only, "truncated": b.truncated}
        lines = [f"{args.store} {args.record_id}: {len(b.descendants)} write(s) derived from it"
                 + (" (through prompts only)" if args.in_prompt_only else "")]
        lines += [f"  {s}: {n}" for s, n in sorted(b.store_counts.items())]
        lines.append("items:")
        lines += [f"  {s} {r}" for s, r in items]
        _out(args, data, lines)
    return 0


def cmd_excise(args) -> int:
    with _ledger(args) as led:
        try:
            r = dod.excise(led, args.store, args.record_id, dry_run=args.dry_run,
                           in_prompt_only=args.in_prompt_only)
        except KeyError:
            raise SystemExit(f"dod: no recorded write for {args.store} {args.record_id}")
        n = sum(len(v) for v in r.targets.values())
        data = {"targets": r.targets, "done": r.done, "dry_run": r.dry_run,
                "skipped": [list(s) for s in r.skipped]}
        verb = "would quarantine" if args.dry_run else "quarantined"
        lines = [f"{verb} {n if args.dry_run else sum(len(v) for v in r.done.values())}"
                 f" item(s) in {len(r.targets)} store(s):"]
        for s, rids in sorted(r.targets.items()):
            lines += [f"  {s} {rid}" for rid in rids]
        if r.skipped:
            lines.append("skipped:")
            lines += [f"  {s.store} {s.record_id}: {s.reason}" for s in r.skipped]
        if not args.dry_run:
            lines.append("undo with: dod release STORE RECORD_ID...")
        _out(args, data, lines)
    return 0


def cmd_release(args) -> int:
    with _ledger(args) as led:
        items = [(args.store, r) for r in args.record_ids]
        quarantined = led.quarantined()
        missing = [i for i in items if i not in quarantined]
        if missing:
            raise SystemExit("dod: not quarantined: " + ", ".join(f"{s} {r}" for s, r in missing)
                             + "; nothing was released")
        led.release(items, reason=args.reason, actor=args.actor)
        _out(args, {"released": [list(i) for i in items]},
             [f"released {len(items)} item(s)"])
    return 0


def cmd_quarantined(args) -> int:
    with _ledger(args) as led:
        q = sorted(led.quarantined())
        _out(args, {"quarantined": [list(i) for i in q]},
             [f"{len(q)} quarantined item(s)"] + [f"  {s} {r}" for s, r in q])
    return 0


def cmd_stats(args) -> int:
    path = _path(args)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        try:
            _check_schema(conn, path)
        except LedgerSchemaError as exc:
            raise SystemExit(f"dod: {exc}")
        q = lambda sql: conn.execute(sql).fetchall()
        writes = dict(q("SELECT store, count(*) FROM writes GROUP BY store"))
        items = dict(q("SELECT store, count(DISTINCT record_id) FROM bindings GROUP BY store"))
        sources = dict(q("SELECT source, count(*) FROM writes GROUP BY source"))
        tomb = q("SELECT count(*) FROM tombstones")[0][0]
        (version,) = conn.execute("PRAGMA user_version").fetchone()
    finally:
        conn.close()
    data = {"schema_version": version, "writes": writes, "items": items, "sources": sources,
            "tombstones": tomb}
    lines = [f"ledger {path} (schema {version})",
             f"writes: {sum(writes.values())}  items: {sum(items.values())}  tombstones: {tomb}"]
    lines += [f"  {s}: {writes.get(s, 0)} write(s), {items.get(s, 0)} item(s)"
              for s in sorted(set(writes) | set(items))]
    lines.append("sources: " + ", ".join(f"{k} {v}" for k, v in sorted(sources.items())))
    _out(args, data, lines)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dod", description="Lineage for agent memory.")
    p.add_argument("--version", action="version", version=f"dod {dod.__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ledger", help="ledger file (default: $DOD_LEDGER)")
    common.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)

    def item(sp):
        sp.add_argument("store")
        sp.add_argument("record_id")

    t = sub.add_parser("trace", parents=[common], help="where an item came from")
    item(t)
    t.add_argument("--max-depth", type=int)
    t.set_defaults(fn=cmd_trace)
    b = sub.add_parser("blast-radius", parents=[common], help="everything derived from an item")
    item(b)
    b.add_argument("--max-depth", type=int)
    b.add_argument("--in-prompt-only", action="store_true",
                   help="follow only edges confirmed in a prompt")
    b.set_defaults(fn=cmd_blast)
    e = sub.add_parser("excise", parents=[common],
                       help="quarantine an item and everything derived from it")
    item(e)
    e.add_argument("--dry-run", action="store_true", help="show the targets, change nothing")
    e.add_argument("--in-prompt-only", action="store_true")
    e.set_defaults(fn=cmd_excise)
    r = sub.add_parser("release", parents=[common], help="undo a quarantine")
    r.add_argument("store")
    r.add_argument("record_ids", nargs="+")
    r.add_argument("--reason")
    r.add_argument("--actor")
    r.set_defaults(fn=cmd_release)
    q = sub.add_parser("quarantined", parents=[common], help="list quarantined items")
    q.set_defaults(fn=cmd_quarantined)
    s = sub.add_parser("stats", parents=[common], help="what the ledger holds")
    s.set_defaults(fn=cmd_stats)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
