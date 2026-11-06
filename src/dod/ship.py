"""Ship a ledger to Postgres, so lineage can be queried across hosts.

Capture stays local: each process appends to its own SQLite ledger, with
the same latency and code as always. A ``Shipper`` copies committed rows to
shared Postgres tables in the background, tagged with the ledger's
``node_id``. ``PgLineage`` answers ``trace`` and ``blast_radius`` over every
node's rows at once, so a memory written on one host and read on another
links up. ``PgLineage.excise`` quarantines across nodes. Quarantine actions taken anywhere reach every node: each shipper
also pulls the other nodes' quarantine rows into its local ledger, so every
node's instrumented stores stop returning a quarantined item.

Run it as its own process, next to the agent::

    dod ship --ledger agent.db --dsn postgresql://... --node host-a

It reads the ledger file read-only (pulled quarantine actions are its only
writes), so capture in the agent's process is untouched. ``Shipper(...)
.start()`` runs the same loop as a thread in the agent's process instead,
which is simpler but competes with capture for the interpreter: expect p99
jitter on captured writes while a pass runs.

Requires ``psycopg`` (the ``[postgres]`` extra). Postgres sees only what the
ledger holds: IDs, hashes, edges and metadata, never memory content.

Ordering
    Within a node, rows ship in ``seq`` order, each table in one Postgres
    transaction per batch with the node's watermarks, so a crash re-ships at
    most one batch, and re-shipping is a no-op (inserts ignore conflicts).
    Across nodes there is no shared sequence, so ``PgLineage.trace`` resolves
    a parent to versions written at or before the child by timestamp, not
    by ``seq`` as the local ``trace`` does.

    Write IDs are random UUIDv7-layout IDs (74 random bits), so they are
    the global key in Postgres; a collision across nodes would be dropped by
    ``ON CONFLICT DO NOTHING``, which at that width doesn't happen in practice.
    Cross-node ``trace`` compares timestamps: keep node clocks synced (NTP),
    or a skewed node's versions can be ordered wrongly.

Quarantine
    A pulled row is appended locally with actor ``dod-ship:<node>`` and is
    not shipped back. Actions from different nodes are applied in the order
    they're pulled. Applying is at least once: a crash between the local
    append and the watermark commit re-applies the action, which appends a
    duplicate row (harmless: the latest action wins).
"""

import logging
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

from dod.ledger import Ledger

logger = logging.getLogger(__name__)

Item = Tuple[str, str]
PULLED = "dod-ship:"
# The node name quarantine actions taken from Postgres (PgLineage) go under.
CONSOLE = "dod-console"

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS {p}writes (
    node_id TEXT NOT NULL, seq BIGINT NOT NULL, write_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL, ts BIGINT NOT NULL, store TEXT NOT NULL,
    agent_id TEXT, session_id TEXT, source TEXT NOT NULL, context_set TEXT,
    UNIQUE (node_id, seq));
CREATE INDEX IF NOT EXISTS {p}writes_by_set ON {p}writes (context_set);
CREATE TABLE IF NOT EXISTS {p}direct_parents (
    write_id TEXT NOT NULL, parent_store TEXT NOT NULL, parent_record_id TEXT NOT NULL,
    tag TEXT NOT NULL, PRIMARY KEY (write_id, parent_store, parent_record_id));
CREATE INDEX IF NOT EXISTS {p}direct_parents_by_item
    ON {p}direct_parents (parent_store, parent_record_id);
CREATE TABLE IF NOT EXISTS {p}context_sets (
    set_hash TEXT PRIMARY KEY, size INTEGER NOT NULL, ts BIGINT NOT NULL, base TEXT,
    step INTEGER NOT NULL, upgrades INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS {p}context_chain (
    set_hash TEXT NOT NULL, part TEXT NOT NULL, PRIMARY KEY (set_hash, part));
CREATE INDEX IF NOT EXISTS {p}context_chain_by_part ON {p}context_chain (part);
CREATE TABLE IF NOT EXISTS {p}context_members (
    set_hash TEXT NOT NULL, store TEXT NOT NULL, record_id TEXT NOT NULL, tag TEXT NOT NULL,
    PRIMARY KEY (set_hash, store, record_id));
CREATE INDEX IF NOT EXISTS {p}context_members_by_item
    ON {p}context_members (store, record_id);
CREATE TABLE IF NOT EXISTS {p}bindings (
    node_id TEXT NOT NULL, rowid BIGINT NOT NULL, write_id TEXT NOT NULL,
    store TEXT NOT NULL, record_id TEXT NOT NULL, ts BIGINT NOT NULL, content_hash TEXT,
    PRIMARY KEY (node_id, rowid));
CREATE INDEX IF NOT EXISTS {p}bindings_by_record ON {p}bindings (store, record_id);
CREATE INDEX IF NOT EXISTS {p}bindings_by_write ON {p}bindings (write_id);
CREATE TABLE IF NOT EXISTS {p}tombstones (
    node_id TEXT NOT NULL, seq BIGINT NOT NULL, store TEXT NOT NULL,
    record_id TEXT NOT NULL, ts BIGINT NOT NULL, reason TEXT NOT NULL, write_id TEXT,
    agent_id TEXT, session_id TEXT, PRIMARY KEY (node_id, seq));
CREATE INDEX IF NOT EXISTS {p}tombstones_by_record ON {p}tombstones (store, record_id);
CREATE TABLE IF NOT EXISTS {p}quarantine (
    node_id TEXT NOT NULL, seq BIGINT NOT NULL, ts BIGINT NOT NULL, store TEXT NOT NULL,
    record_id TEXT NOT NULL, action TEXT NOT NULL, reason TEXT, actor TEXT,
    PRIMARY KEY (node_id, seq));
CREATE TABLE IF NOT EXISTS {p}ship_state (
    node_id TEXT NOT NULL, tbl TEXT NOT NULL, hi BIGINT NOT NULL,
    PRIMARY KEY (node_id, tbl));
CREATE OR REPLACE VIEW {p}parents (write_id, parent_store, parent_record_id, tag) AS
    SELECT write_id, parent_store, parent_record_id, tag FROM {p}direct_parents
    UNION ALL
    SELECT w.write_id, m.store, m.record_id, m.tag FROM {p}writes w
    JOIN {p}context_chain c ON c.set_hash = w.context_set
    JOIN {p}context_members m ON m.set_hash = c.part;
CREATE OR REPLACE FUNCTION {p}append_only() RETURNS trigger AS $$
BEGIN RAISE EXCEPTION '% is append-only', TG_TABLE_NAME; END $$ LANGUAGE plpgsql;
"""
_APPEND_ONLY = ("writes", "direct_parents", "context_sets", "context_chain",
                "context_members", "bindings", "tombstones", "quarantine")


def _connect(dsn: str):
    import psycopg
    return psycopg.connect(dsn, autocommit=False)


def ensure_schema(dsn: str, prefix: str = "dod_") -> None:
    """Create the shared tables (idempotent). Concurrent first runs are
    serialized with an advisory lock."""
    with _connect(dsn) as pg:
        pg.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (prefix + "schema",))
        pg.execute(_PG_SCHEMA.format(p=prefix))
        for t in _APPEND_ONLY:
            pg.execute(f"DROP TRIGGER IF EXISTS {prefix}{t}_append_only ON {prefix}{t}")
            pg.execute(f"CREATE TRIGGER {prefix}{t}_append_only BEFORE UPDATE OR DELETE"
                       f" ON {prefix}{t} FOR EACH ROW EXECUTE FUNCTION {prefix}append_only()")
        pg.commit()


class Shipper:
    """Copy ``ledger``'s committed rows to Postgres every ``interval``
    seconds, and pull other nodes' quarantine actions into it.

    ``ship_once()`` and ``pull_once()`` do one pass each (for tests, or a
    cron job instead of the thread). ``stop()`` runs a final pass. Failures
    are logged and retried on the next pass; capture never waits on Postgres.
    """

    def __init__(self, ledger: Ledger, dsn: str, node_id: str, *, prefix: str = "dod_",
                 interval: float = 1.0, batch: int = 5000, pull: bool = True) -> None:
        if not node_id or PULLED in node_id or node_id == CONSOLE:
            raise ValueError(f"node_id must be a non-empty name, not {CONSOLE!r},"
                             " without 'dod-ship:'")
        if not getattr(ledger, "_file", None):
            raise ValueError("shipping needs an on-disk ledger")
        self.ledger, self.dsn, self.node_id, self.p = ledger, dsn, node_id, prefix
        self.interval, self.batch, self.pull = interval, batch, pull
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()  # one pass at a time
        ensure_schema(dsn, prefix)

    # --- lifecycle -------------------------------------------------------------

    def start(self) -> "Shipper":
        """Ship from a thread in this process. Prefer ``dod ship`` (its own
        process): as a thread, each pass competes with capture for the
        interpreter, and captured writes see p99 spikes above 1 ms."""
        import warnings
        warnings.warn("Shipper.start() runs in the agent's process and adds capture latency;"
                      " run `dod ship` as its own process in production", RuntimeWarning,
                      stacklevel=2)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dod-shipper", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.sync()

    def __enter__(self) -> "Shipper":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sync()
            except Exception:
                logger.exception("dod: shipping to Postgres failed; retrying in %ss",
                                 self.interval)

    def sync(self) -> int:
        """Ship everything committed so far, then pull; return rows shipped."""
        n = 0
        while True:
            shipped = self.ship_once()
            n += shipped
            if shipped < self.batch:
                break
        if self.pull:
            self.pull_once()
        return n

    # --- shipping ----------------------------------------------------------------

    def _source(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.ledger._file}?mode=ro", uri=True,
                               isolation_level=None)
        conn.execute("BEGIN")  # one snapshot for the pass
        return conn

    def ship_once(self) -> int:
        """Ship up to ``batch`` rows per table; return the number shipped."""
        p, node = self.p, self.node_id
        with self._lock:
            src = self._source()
            try:
                with _connect(self.dsn) as pg:
                    hi = dict(pg.execute(f"SELECT tbl, hi FROM {p}ship_state WHERE node_id = %s",
                                         (node,)).fetchall())
                    n, marks = 0, {}

                    sets = src.execute(
                        "SELECT seq, set_hash, size, ts, base, step, upgrades FROM context_sets"
                        " WHERE seq > ? ORDER BY seq LIMIT ?",
                        (hi.get("context_sets", 0), self.batch)).fetchall()
                    if sets:
                        hashes = [r[1] for r in sets]
                        self._insert(pg, "context_sets", [r[1:] for r in sets])
                        self._insert(pg, "context_chain", _rows_for(
                            src, "SELECT set_hash, part FROM context_chain WHERE set_hash IN",
                            hashes))
                        self._insert(pg, "context_members", _rows_for(
                            src, "SELECT set_hash, store, record_id, tag FROM context_members"
                                 " WHERE set_hash IN", hashes))
                        marks["context_sets"] = sets[-1][0]
                        n += len(sets)

                    writes = src.execute(
                        "SELECT seq, write_id, content_hash, ts, store, agent_id, session_id,"
                        " source, context_set FROM writes WHERE seq > ? ORDER BY seq LIMIT ?",
                        (hi.get("writes", 0), self.batch)).fetchall()
                    if writes:
                        self._insert(pg, "writes", [(node, *r) for r in writes])
                        self._insert(pg, "direct_parents", _rows_for(
                            src, "SELECT write_id, parent_store, parent_record_id, tag"
                                 " FROM direct_parents WHERE write_id IN",
                            [r[1] for r in writes]))
                        marks["writes"] = writes[-1][0]
                        n += len(writes)

                    for table, cols, key in (
                        ("bindings", "rowid, write_id, store, record_id, ts, content_hash",
                         "rowid"),
                        ("tombstones", "seq, store, record_id, ts, reason, write_id, agent_id,"
                                       " session_id", "seq"),
                        ("quarantine", "seq, ts, store, record_id, action, reason, actor",
                         "seq"),
                    ):
                        rows = src.execute(
                            f"SELECT {cols} FROM {table} WHERE {key} > ? ORDER BY {key}"
                            " LIMIT ?", (hi.get(table, 0), self.batch)).fetchall()
                        if not rows:
                            continue
                        marks[table] = rows[-1][0]
                        if table == "quarantine":  # pulled rows don't go back
                            rows = [r for r in rows if not (r[6] or "").startswith(PULLED)]
                        self._insert(pg, table, [(node, *r) for r in rows])
                        n += len(rows)

                    for tbl, mark in marks.items():
                        pg.execute(
                            f"INSERT INTO {p}ship_state (node_id, tbl, hi) VALUES (%s, %s, %s)"
                            " ON CONFLICT (node_id, tbl) DO UPDATE SET hi = EXCLUDED.hi",
                            (node, tbl, mark))
                    pg.commit()
                    return n
            finally:
                src.close()

    def _insert(self, pg, table: str, rows: List[tuple]) -> None:
        if not rows:
            return
        marks = ", ".join(["%s"] * len(rows[0]))
        with pg.cursor() as cur:
            cur.executemany(f"INSERT INTO {self.p}{table} VALUES ({marks})"
                            " ON CONFLICT DO NOTHING", rows)

    # --- pulling -----------------------------------------------------------------

    def pull_once(self) -> int:
        """Apply other nodes' new quarantine actions locally; return how many."""
        p, node = self.p, self.node_id
        with self._lock, _connect(self.dsn) as pg:
            marks = {tbl[len("pull:"):]: hi for tbl, hi in pg.execute(
                f"SELECT tbl, hi FROM {p}ship_state WHERE node_id = %s AND tbl LIKE 'pull:%%'",
                (node,)).fetchall()}
            rows = pg.execute(
                f"SELECT node_id, seq, store, record_id, action, reason FROM {p}quarantine"
                " WHERE node_id <> %s ORDER BY ts, node_id, seq", (node,)).fetchall()
            applied = 0
            for remote, seq, store, record_id, action, reason in rows:
                if seq <= marks.get(remote, 0):
                    continue
                fn = self.ledger.quarantine if action == "quarantine" else self.ledger.release
                fn([(store, record_id)], reason=reason, actor=PULLED + remote)
                marks[remote] = seq
                applied += 1
                pg.execute(
                    f"INSERT INTO {p}ship_state (node_id, tbl, hi) VALUES (%s, %s, %s)"
                    " ON CONFLICT (node_id, tbl) DO UPDATE SET hi = EXCLUDED.hi",
                    (node, "pull:" + remote, seq))
                pg.commit()  # each applied action is remembered at once
            return applied


def _rows_for(src: sqlite3.Connection, select: str, keys: List[str]) -> List[tuple]:
    out: List[tuple] = []
    for i in range(0, len(keys), 500):
        chunk = keys[i:i + 500]
        out += src.execute(f"{select} ({', '.join('?' * len(chunk))})", chunk).fetchall()
    return out


class PgLineage:
    """``trace`` and ``blast_radius`` over every node's shipped rows."""

    def __init__(self, dsn: str, prefix: str = "dod_") -> None:
        self.dsn, self.p = dsn, prefix

    def blast_radius(self, store: str, record_id: str,
                     in_prompt_only: bool = False) -> Dict[str, Set[Item]]:
        """``{"items": derived items, "writes": derived write IDs}`` (the
        queried item excluded), across nodes. Raises ``KeyError`` if no node
        recorded a write bound to the item."""
        p = self.p
        only = "AND pa.tag = 'in_prompt'" if in_prompt_only else ""
        with _connect(self.dsn) as pg:
            if not pg.execute(f"SELECT 1 FROM {p}bindings WHERE store = %s AND record_id = %s"
                              " LIMIT 1", (store, record_id)).fetchone():
                raise KeyError((store, record_id))
            rows = pg.execute(f"""
                WITH RECURSIVE reach(store, record_id, write_id) AS (
                    SELECT %s::text, %s::text, NULL::text
                    UNION
                    SELECT b.store, b.record_id, pa.write_id
                    FROM reach r
                    JOIN {p}parents pa ON pa.parent_store = r.store
                        AND pa.parent_record_id = r.record_id {only}
                    JOIN {p}bindings b ON b.write_id = pa.write_id
                )
                SELECT store, record_id, write_id FROM reach WHERE write_id IS NOT NULL
            """, (store, record_id)).fetchall()
            # Writes derived from the item that were never bound to a record.
            unbound = pg.execute(f"""
                WITH RECURSIVE reach(store, record_id) AS (
                    SELECT %s::text, %s::text
                    UNION
                    SELECT b.store, b.record_id FROM reach r
                    JOIN {p}parents pa ON pa.parent_store = r.store
                        AND pa.parent_record_id = r.record_id {only}
                    JOIN {p}bindings b ON b.write_id = pa.write_id
                )
                SELECT DISTINCT pa.write_id FROM reach r
                JOIN {p}parents pa ON pa.parent_store = r.store
                    AND pa.parent_record_id = r.record_id {only}
            """, (store, record_id)).fetchall()
        items = {(s, r) for s, r, _ in rows} - {(store, record_id)}
        return {"items": items, "writes": {w for _, _, w in rows} | {w for (w,) in unbound}}

    def trace(self, store: str, record_id: str) -> Dict[str, object]:
        """``{"writes": write IDs bound to the item, "ancestors": ancestor
        write IDs, "roots": reached writes with no recorded parents}``. A
        parent resolves to the versions written at or before the child (by
        timestamp), on any node."""
        p = self.p
        with _connect(self.dsn) as pg:
            start = {w for (w,) in pg.execute(
                f"SELECT write_id FROM {p}bindings WHERE store = %s AND record_id = %s",
                (store, record_id)).fetchall()}
            if not start:
                raise KeyError((store, record_id))
            rows = pg.execute(f"""
                WITH RECURSIVE anc(write_id, ts) AS (
                    SELECT w.write_id, w.ts FROM {p}bindings b
                    JOIN {p}writes w ON w.write_id = b.write_id
                    WHERE b.store = %s AND b.record_id = %s
                    UNION
                    SELECT w2.write_id, w2.ts FROM anc a
                    JOIN {p}parents pa ON pa.write_id = a.write_id
                    JOIN {p}bindings b ON b.store = pa.parent_store
                        AND b.record_id = pa.parent_record_id
                    JOIN {p}writes w2 ON w2.write_id = b.write_id
                        AND w2.ts <= a.ts AND w2.write_id <> a.write_id
                )
                SELECT a.write_id,
                       NOT EXISTS (SELECT 1 FROM {p}parents pa WHERE pa.write_id = a.write_id)
                FROM anc a
            """, (store, record_id)).fetchall()
        return {"writes": start, "ancestors": {w for w, _ in rows} - start,
                "roots": {w for w, root in rows if root}}

    def quarantined(self) -> Set[Item]:
        """Items whose latest quarantine action across nodes (by time) is
        ``quarantine``."""
        with _connect(self.dsn) as pg:
            rows = pg.execute(f"""
                SELECT DISTINCT ON (store, record_id) store, record_id, action
                FROM {self.p}quarantine ORDER BY store, record_id, ts DESC, seq DESC
            """).fetchall()
        return {(s, r) for s, r, a in rows if a == "quarantine"}

    def quarantine(self, items: List[Item], reason: Optional[str] = None,
                   actor: Optional[str] = None, node: str = CONSOLE) -> None:
        """Quarantine ``items`` on every node: the rows go in under ``node``,
        and each node's shipper pulls them into its ledger on its next pass."""
        self._append(items, "quarantine", reason, actor, node)

    def release(self, items: List[Item], reason: Optional[str] = None,
                actor: Optional[str] = None, node: str = CONSOLE) -> None:
        """Release ``items`` on every node (pulled like ``quarantine``)."""
        self._append(items, "release", reason, actor, node)

    def excise(self, store: str, record_id: str, *, in_prompt_only: bool = False,
               dry_run: bool = False, reason: Optional[str] = None,
               actor: Optional[str] = None, node: str = CONSOLE) -> List[Item]:
        """Quarantine ``(store, record_id)`` and everything derived from it on
        any node; return the items, root first. ``dry_run`` only returns them."""
        derived = self.blast_radius(store, record_id, in_prompt_only=in_prompt_only)["items"]
        items = [(store, record_id)] + sorted(derived)
        if not dry_run:
            already = self.quarantined()
            todo = [i for i in items if i not in already]
            if todo:
                self._append(todo, "quarantine",
                             reason or f"excision of {store} {record_id}", actor, node)
        return items

    def _append(self, items: List[Item], action: str, reason: Optional[str],
                actor: Optional[str], node: str) -> None:
        items = list(items)
        if not items:
            return
        ts = time.time_ns()
        with _connect(self.dsn) as pg:
            # One writer per node name at a time, so its seq stays dense.
            pg.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (self.p + "q:" + node,))
            (top,) = pg.execute(f"SELECT coalesce(max(seq), 0) FROM {self.p}quarantine"
                                " WHERE node_id = %s", (node,)).fetchone()
            with pg.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self.p}quarantine (node_id, seq, ts, store, record_id, action,"
                    " reason, actor) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [(node, top + i + 1, ts, s, r, action, reason, actor)
                     for i, (s, r) in enumerate(items)])
            pg.commit()


__all__ = ["Shipper", "PgLineage", "ensure_schema"]
