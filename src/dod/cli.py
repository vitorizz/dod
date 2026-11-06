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
from dod import audit
from dod.ledger import Ledger, LedgerSchemaError, _check_schema
from dod.query import blast_radius, trace


def _path(args) -> str:
    path = args.ledger or os.environ.get("DOD_LEDGER")
    if not path:
        raise SystemExit("dod: no ledger given: pass --ledger PATH or set DOD_LEDGER")
    if not os.path.exists(path):
        raise SystemExit(f"dod: no ledger at {path}")
    return path


def _audited(path: str) -> bool:
    """Whether the file's audit chain is in use, so writes must extend it."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return bool(conn.execute("SELECT 1 FROM audit_chain LIMIT 1").fetchone())
    except sqlite3.Error:
        return False  # no chain table (an older schema)
    finally:
        conn.close()


def _ledger(args) -> Ledger:
    path = _path(args)
    try:
        return Ledger(path, audit=_audited(path))
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
                             in_prompt_only=args.in_prompt_only, precise=args.precise)
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
                           in_prompt_only=args.in_prompt_only, precise=args.precise)
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


def cmd_verify(args) -> int:
    with _ledger(args) as led:
        v = audit.verify(led, expected_head=args.expect_head)
    lines = [f"{'verified' if v.ok else 'NOT VERIFIED'}: {v.entries} audit entr"
             f"{'y' if v.entries == 1 else 'ies'}, head {v.head}"]
    lines += [f"  {p}" for p in v.problems]
    _out(args, {"ok": v.ok, "entries": v.entries, "head": v.head, "problems": v.problems}, lines)
    return 0 if v.ok else 1


def cmd_report(args) -> int:
    with _ledger(args) as led:
        try:
            rep = audit.report(led, args.store, args.record_id)
        except KeyError:
            raise SystemExit(f"dod: no recorded write for {args.store} {args.record_id}")
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(audit.to_markdown(rep), end="")
    return 0


def cmd_ship(args) -> int:
    """Ship the ledger to Postgres until interrupted (SIGINT or SIGTERM)."""
    import signal
    import threading

    from dod.ship import Shipper

    dsn = args.dsn or os.environ.get("DOD_SHIP_DSN")
    if not dsn:
        raise SystemExit("dod: no Postgres DSN: pass --dsn or set DOD_SHIP_DSN")
    path = _path(args)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    with Ledger(path, audit=_audited(path), background_index=False) as led:
        shipper = Shipper(led, dsn, args.node, prefix=args.prefix,
                          interval=args.interval, pull=not args.no_pull)
        if args.once:
            n = shipper.sync()
            _out(args, {"shipped": n}, [f"shipped {n} row(s)"])
            return 0
        print(f"dod: shipping {path} as node {args.node!r} every {args.interval}s",
              file=sys.stderr, flush=True)
        try:
            while not done.wait(args.interval):
                try:
                    shipper.sync()
                except Exception as exc:  # retried on the next pass
                    print(f"dod: ship failed: {exc}", file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            pass
        try:
            shipper.sync()
        except Exception as exc:
            print(f"dod: final ship failed: {exc}; the next run ships it", file=sys.stderr,
                  flush=True)
            return 1
    return 0


def cmd_pg(args) -> int:
    """Lineage across every node shipped to Postgres (see ``dod ship``)."""
    from dod.ship import PgLineage

    dsn = args.dsn or os.environ.get("DOD_SHIP_DSN")
    if not dsn:
        raise SystemExit("dod: no Postgres DSN: pass --dsn or set DOD_SHIP_DSN")
    lin = PgLineage(dsn, args.prefix)
    try:
        if args.pg_cmd == "trace":
            t = lin.trace(args.store, args.record_id)
            data = {k: sorted(v) for k, v in t.items()}
            lines = [f"{args.store} {args.record_id}: {len(t['ancestors'])} ancestor write(s),"
                     f" {len(t['roots'])} root(s)"] + [f"  root {w}" for w in sorted(t["roots"])]
        elif args.pg_cmd == "blast-radius":
            b = lin.blast_radius(args.store, args.record_id, in_prompt_only=args.in_prompt_only)
            data = {"items": [list(i) for i in sorted(b["items"])], "writes": len(b["writes"])}
            lines = [f"{args.store} {args.record_id}: {len(b['items'])} item(s) derived from it,"
                     " on any node"] + [f"  {s} {r}" for s, r in sorted(b["items"])]
        elif args.pg_cmd == "excise":
            items = lin.excise(args.store, args.record_id, in_prompt_only=args.in_prompt_only,
                               dry_run=args.dry_run, reason=args.reason, actor=args.actor)
            data = {"items": [list(i) for i in items], "dry_run": args.dry_run}
            verb = "would quarantine" if args.dry_run else "quarantined, on every node at its next pass:"
            lines = [f"{verb} {len(items)} item(s)"] + [f"  {s} {r}" for s, r in items]
        elif args.pg_cmd == "release":
            items = [(args.store, r) for r in args.record_ids]
            lin.release(items, reason=args.reason, actor=args.actor)
            data = {"released": [list(i) for i in items]}
            lines = [f"released {len(items)} item(s) on every node"]
        else:
            q = sorted(lin.quarantined())
            data = {"quarantined": [list(i) for i in q]}
            lines = [f"{len(q)} quarantined item(s), across nodes"] + [f"  {s} {r}" for s, r in q]
    except KeyError:
        raise SystemExit(f"dod: no node recorded a write for {args.store} {args.record_id}")
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
    b.add_argument("--precise", action="store_true",
                   help="leave out writes that only read an earlier, clean version of an item")
    b.add_argument("--in-prompt-only", action="store_true",
                   help="follow only edges confirmed in a prompt")
    b.set_defaults(fn=cmd_blast)
    e = sub.add_parser("excise", parents=[common],
                       help="quarantine an item and everything derived from it")
    item(e)
    e.add_argument("--dry-run", action="store_true", help="show the targets, change nothing")
    e.add_argument("--in-prompt-only", action="store_true")
    e.add_argument("--precise", action="store_true")
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
    sh = sub.add_parser("ship", parents=[common],
                        help="copy the ledger to Postgres (runs until stopped)")
    sh.add_argument("--dsn", help="Postgres connection string (or $DOD_SHIP_DSN)")
    sh.add_argument("--node", required=True, help="this ledger's node name")
    sh.add_argument("--prefix", default="dod_", help="table name prefix (default dod_)")
    sh.add_argument("--interval", type=float, default=1.0, help="seconds between passes")
    sh.add_argument("--once", action="store_true", help="one pass, then exit")
    sh.add_argument("--no-pull", action="store_true",
                    help="don't pull other nodes' quarantine actions")
    sh.set_defaults(fn=cmd_ship)
    pgc = argparse.ArgumentParser(add_help=False)
    pgc.add_argument("--dsn", help="Postgres connection string (or $DOD_SHIP_DSN)")
    pgc.add_argument("--prefix", default="dod_", help="table name prefix (default dod_)")
    pgc.add_argument("--json", action="store_true", help="machine-readable output")
    pg = sub.add_parser("pg", help="lineage across every node shipped to Postgres")
    pgs = pg.add_subparsers(dest="pg_cmd", required=True)
    for name, help_ in (("trace", "where an item came from, on any node"),
                        ("blast-radius", "everything derived from an item, on any node"),
                        ("excise", "quarantine an item and what was derived from it, on every node")):
        x = pgs.add_parser(name, parents=[pgc], help=help_)
        item(x)
        if name != "trace":
            x.add_argument("--in-prompt-only", action="store_true")
        if name == "excise":
            x.add_argument("--dry-run", action="store_true")
            x.add_argument("--reason")
            x.add_argument("--actor")
        x.set_defaults(fn=cmd_pg)
    rel = pgs.add_parser("release", parents=[pgc], help="release items on every node")
    rel.add_argument("store")
    rel.add_argument("record_ids", nargs="+")
    rel.add_argument("--reason")
    rel.add_argument("--actor")
    rel.set_defaults(fn=cmd_pg)
    pgs.add_parser("quarantined", parents=[pgc], help="what is quarantined, across nodes"
                   ).set_defaults(fn=cmd_pg)
    v = sub.add_parser("verify", parents=[common],
                       help="check the audit chain (exit 1 if tampered)")
    v.add_argument("--expect-head", help="a digest from an earlier verify: must still be there")
    v.set_defaults(fn=cmd_verify)
    rp = sub.add_parser("report", parents=[common],
                        help="excision evidence for an item (Markdown, or --json)")
    item(rp)
    rp.set_defaults(fn=cmd_report)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
