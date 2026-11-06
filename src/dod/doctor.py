"""Health check for a ledger file: ``dod doctor``.

``check(path)`` reads the ledger (read-only; an audited one is also
verified) and returns findings, each ``ok``, ``warn`` or ``error``:

- **schema**: current, needs a migration (the next open runs it), or
  unknown to this version of the library.
- **wal**: the write-ahead log's size. The writer keeps it bounded; a file
  far past the restart size means restarts are failing (a long-lived reader
  holding a snapshot).
- **index**: how far the lineage index lags the ledger. Queries stay
  correct (they scan what the index hasn't reached), but a large lag makes
  them slower; it catches up while a ledger is open.
- **unbound**: writes recorded before a store call that were never bound to
  a record or tombstoned, older than ``unbound_after`` seconds. Usually a
  store call that failed after capture recorded it, or a process that died
  in between; a growing count is worth reconciling (``dod.reconcile``).
- **audit**: for an audited ledger, whether the hash chain verifies.
- **quarantine**: how many items are quarantined (informational).
"""

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import List

from dod.ledger import SCHEMA_VERSION, _MIGRATIONS, _WAL_RESTART_BYTES, LedgerSchemaError, \
    _check_schema

LEVELS = ("ok", "warn", "error")
INDEX_LAG_WARN = 50_000


@dataclass
class Finding:
    check: str
    level: str
    message: str
    value: object = None


def _ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def check(path: str, unbound_after: float = 300.0) -> List[Finding]:
    """Check the ledger file at ``path``; see the module docstring."""
    path = str(path)
    out: List[Finding] = []
    if not os.path.exists(path):
        return [Finding("schema", "error", f"no ledger at {path}")]
    conn = _ro(path)
    try:
        try:
            version = _check_schema(conn, path)
        except LedgerSchemaError as exc:
            return [Finding("schema", "error", str(exc))]
        except sqlite3.DatabaseError as exc:
            return [Finding("schema", "error", f"not a readable ledger: {exc}")]
        if version == SCHEMA_VERSION:
            out.append(Finding("schema", "ok", f"schema {version}", version))
        elif version in _MIGRATIONS:
            out.append(Finding("schema", "warn", f"schema {version}: the next open migrates it to"
                                                 f" {SCHEMA_VERSION}", version))
            return out  # the rest reads the current schema
        else:
            out.append(Finding("schema", "ok", "new, empty file", version))
            return out

        wal = os.path.getsize(path + "-wal") if os.path.exists(path + "-wal") else 0
        level = "warn" if wal > 4 * _WAL_RESTART_BYTES else "ok"
        out.append(Finding("wal", level, f"write-ahead log {wal / 2**20:.1f} MiB"
                           + ("; far past the restart size: a reader may be holding an old"
                              " snapshot" if level == "warn" else ""), wal))

        (top,) = conn.execute("SELECT coalesce(max(seq), 0) FROM writes").fetchone()
        index = path + "-lineage"
        if not os.path.exists(index):
            out.append(Finding("index", "warn" if top else "ok",
                               "no lineage index yet; it is built while a ledger is open", top))
        else:
            try:
                ic = _ro(index)
                try:
                    row = ic.execute("SELECT writes_upto FROM index_filled ORDER BY seq DESC"
                                     " LIMIT 1").fetchone()
                finally:
                    ic.close()
                lag = top - (row[0] if row else 0)
                level = "warn" if lag > INDEX_LAG_WARN else "ok"
                out.append(Finding("index", level, f"lineage index {lag} write(s) behind"
                                   + ("; queries scan the rest, so they're slower until it"
                                      " catches up" if level == "warn" else ""), lag))
            except sqlite3.DatabaseError:
                out.append(Finding("index", "warn", "lineage index unreadable; it is rebuilt on"
                                                    " the next open"))

        cutoff = time.time_ns() - int(unbound_after * 1e9)
        unbound = conn.execute(
            "SELECT w.store, count(*) FROM writes w"
            " WHERE w.ts < ? AND NOT EXISTS (SELECT 1 FROM bindings b WHERE b.write_id = w.write_id)"
            " AND NOT EXISTS (SELECT 1 FROM tombstones t WHERE t.write_id = w.write_id)"
            " GROUP BY w.store ORDER BY w.store", (cutoff,)).fetchall()
        n = sum(c for _, c in unbound)
        out.append(Finding(
            "unbound", "warn" if n else "ok",
            "no unbound writes" if not n else
            f"{n} write(s) never bound to a record ("
            + ", ".join(f"{s}: {c}" for s, c in unbound)
            + "): a store call failed after capture, or a process died in between;"
              " reconcile the store (dod.reconcile)", dict(unbound)))

        audited = conn.execute("SELECT 1 FROM audit_chain LIMIT 1").fetchone() is not None
        q = conn.execute(
            "SELECT count(*) FROM (SELECT store, record_id, action FROM quarantine q"
            " WHERE seq = (SELECT max(seq) FROM quarantine q2 WHERE q2.store = q.store"
            " AND q2.record_id = q.record_id)) WHERE action = 'quarantine'").fetchone()[0]
    finally:
        conn.close()

    if audited:
        from dod import audit
        from dod.ledger import Ledger
        with Ledger(path, audit=True, background_index=False) as led:
            v = audit.verify(led)
        out.append(Finding("audit", "ok" if v.ok else "error",
                           f"audit chain verified ({v.entries} entries)" if v.ok else
                           "audit chain NOT verified: " + "; ".join(v.problems[:3]), v.ok))
    else:
        out.append(Finding("audit", "ok", "not audited (Ledger(path, audit=True) enables it)"))
    out.append(Finding("quarantine", "ok", f"{q} item(s) quarantined", q))
    return out
