"""Append-only SQLite ledger of memory writes, their parent edges, and deletions.

A memory item is identified by ``(store, record_id)``, so edges can cross
stores. A deleted item gets a tombstone row; nothing is ever removed. The
ledger never stores memory content: only IDs, content hashes, edges, and
metadata.

A write's parents are its context set plus its direct parents. A context set
is a scope's shared context, stored once per distinct set of
``(store, record_id, tag)`` members and addressed by their hash, so a scope's
writes reference it instead of each copying every edge. Direct parents (for
example a compaction's replaced items) are stored per write. The ``parents``
view joins both into one row per edge, ``(write_id, parent_store,
parent_record_id, tag)``; where an item is in both, the direct row carries
the merged tag and the context row is hidden.

``child_edges`` is the same edges keyed by parent item, for walking forward.
It is derived, never written on the append path: the background thread fills
it after the fact, and ``child_edges_filled`` records how far (the highest
write ``seq`` indexed). Queries read writes past that watermark from the
source tables.

A file records its schema version (``PRAGMA user_version``); opening one at
another version, or an unversioned file with tables, raises
``LedgerSchemaError`` before anything is read or written.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Iterable, List, Optional, Set, Tuple, Union

SOURCES = ("user", "tool", "document", "agent_inference")
TAGS = ("in_prompt", "retrieved_in_scope")
REASONS = ("consolidation", "excision", "forget", "other")

Item = Tuple[str, str]
Parent = Tuple[str, str, str]

# Stored in the file's ``PRAGMA user_version``. Bump it on any change to
# tables, columns, triggers or views, and ship a migration (or a "recreate"
# note in docs/decisions.md) with the bump.
SCHEMA_VERSION = 2  # 2: child_edges + child_edges_filled (#34)


class LedgerSchemaError(Exception):
    """A ledger file's schema version differs from this library's."""

_TABLES = (
    "writes", "direct_parents", "context_sets", "context_members", "bindings",
    "tombstones", "quarantine", "child_edges", "child_edges_filled",
)

# The parents view's edges for writes with lo < seq <= hi, as
# (parent_store, parent_record_id, seq, write_id, tag). Spelled out per branch
# so both seek writes by seq.
_EDGES_BETWEEN = """
SELECT d.parent_store, d.parent_record_id, w.seq, w.write_id, d.tag
FROM writes w JOIN direct_parents d ON d.write_id = w.write_id
WHERE w.seq > :lo AND w.seq <= :hi
UNION ALL
SELECT m.store, m.record_id, w.seq, w.write_id, m.tag
FROM writes w JOIN context_members m ON m.set_hash = w.context_set
WHERE w.seq > :lo AND w.seq <= :hi AND NOT EXISTS (
    SELECT 1 FROM direct_parents d
    WHERE d.write_id = w.write_id AND d.parent_store = m.store
        AND d.parent_record_id = m.record_id
)
"""

# Most edges and writes one fill transaction indexes.
_FILL_MAX = 500
# Each edge lands on a random leaf of child_edges, so a 500-edge fill holds
# the write lock for milliseconds. The background fill sizes its batches to
# hold it about _FILL_TARGET seconds, so an append that arrives mid-fill waits
# well under the 1 ms bar.
_FILL_TARGET = 0.0002
# Pause between fill transactions: at least _FILL_MIN_PAUSE seconds, and
# while writes are arriving, _FILL_BUSY_FACTOR times the last one's lock time.
_FILL_MIN_PAUSE = 0.0005
_FILL_BUSY_FACTOR = 100

_SCHEMA = """
CREATE TABLE IF NOT EXISTS writes (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    write_id     TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    ts           INTEGER NOT NULL,
    store        TEXT NOT NULL,
    agent_id     TEXT,
    session_id   TEXT,
    source       TEXT NOT NULL
        CHECK (source IN ('user', 'tool', 'document', 'agent_inference')),
    context_set  TEXT REFERENCES context_sets (set_hash)
);
CREATE TABLE IF NOT EXISTS context_sets (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    set_hash TEXT NOT NULL UNIQUE,
    size     INTEGER NOT NULL,
    ts       INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS context_members (
    set_hash  TEXT NOT NULL REFERENCES context_sets (set_hash),
    store     TEXT NOT NULL,
    record_id TEXT NOT NULL,
    tag       TEXT NOT NULL
        CHECK (tag IN ('in_prompt', 'retrieved_in_scope')),
    PRIMARY KEY (set_hash, store, record_id)
);
CREATE TABLE IF NOT EXISTS direct_parents (
    write_id         TEXT NOT NULL REFERENCES writes (write_id),
    parent_store     TEXT NOT NULL,
    parent_record_id TEXT NOT NULL,
    tag              TEXT NOT NULL
        CHECK (tag IN ('in_prompt', 'retrieved_in_scope')),
    PRIMARY KEY (write_id, parent_store, parent_record_id)
);
CREATE VIEW IF NOT EXISTS parents (write_id, parent_store, parent_record_id, tag) AS
SELECT write_id, parent_store, parent_record_id, tag FROM direct_parents
UNION ALL
SELECT w.write_id, m.store, m.record_id, m.tag
FROM writes w JOIN context_members m ON m.set_hash = w.context_set
WHERE NOT EXISTS (
    SELECT 1 FROM direct_parents d
    WHERE d.write_id = w.write_id AND d.parent_store = m.store
        AND d.parent_record_id = m.record_id
);
CREATE TABLE IF NOT EXISTS bindings (
    write_id  TEXT NOT NULL REFERENCES writes (write_id),
    store     TEXT NOT NULL,
    record_id TEXT NOT NULL,
    ts        INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS tombstones (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    store      TEXT NOT NULL,
    record_id  TEXT NOT NULL,
    ts         INTEGER NOT NULL,
    reason     TEXT NOT NULL
        CHECK (reason IN ('consolidation', 'excision', 'forget', 'other')),
    write_id   TEXT REFERENCES writes (write_id),
    agent_id   TEXT,
    session_id TEXT
);
CREATE TABLE IF NOT EXISTS quarantine (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        INTEGER NOT NULL,
    store     TEXT NOT NULL,
    record_id TEXT NOT NULL,
    action    TEXT NOT NULL CHECK (action IN ('quarantine', 'release')),
    reason    TEXT,
    actor     TEXT
);
CREATE TABLE IF NOT EXISTS child_edges (
    parent_store     TEXT NOT NULL,
    parent_record_id TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    write_id         TEXT NOT NULL,
    tag              TEXT NOT NULL
        CHECK (tag IN ('in_prompt', 'retrieved_in_scope')),
    PRIMARY KEY (parent_store, parent_record_id, seq)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS child_edges_filled (
    upto INTEGER PRIMARY KEY,
    ts   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS bindings_by_write ON bindings (write_id);
CREATE INDEX IF NOT EXISTS bindings_by_record ON bindings (store, record_id);
CREATE INDEX IF NOT EXISTS tombstones_by_record ON tombstones (store, record_id);
""" + "".join(
    f"""
CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t}
BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
"""
    for t in _TABLES
    for op in ("UPDATE", "DELETE")
) + "".join(
    # REPLACE deletes the conflicting row without firing DELETE triggers
    # unless the writing connection enabled recursive_triggers, which is a
    # per-connection setting. Reject the conflict before it happens instead.
    f"""
CREATE TRIGGER IF NOT EXISTS {t}_no_replace BEFORE INSERT ON {t}
WHEN EXISTS (SELECT 1 FROM {t} WHERE rowid = NEW.rowid){extra}
BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
"""
    for t, extra in (
        ("writes", "\n    OR EXISTS (SELECT 1 FROM writes WHERE write_id = NEW.write_id)"),
        (
            "direct_parents",
            "\n    OR EXISTS (SELECT 1 FROM direct_parents WHERE write_id = NEW.write_id"
            "\n        AND parent_store = NEW.parent_store"
            " AND parent_record_id = NEW.parent_record_id)",
        ),
        (
            "context_sets",
            "\n    OR EXISTS (SELECT 1 FROM context_sets WHERE set_hash = NEW.set_hash)",
        ),
        (
            "context_members",
            "\n    OR EXISTS (SELECT 1 FROM context_members WHERE set_hash = NEW.set_hash"
            "\n        AND store = NEW.store AND record_id = NEW.record_id)",
        ),
        ("bindings", ""),
        ("tombstones", ""),
        ("quarantine", ""),
        ("child_edges_filled", ""),
    )
) + """
CREATE TRIGGER IF NOT EXISTS child_edges_no_replace BEFORE INSERT ON child_edges
WHEN EXISTS (SELECT 1 FROM child_edges WHERE parent_store = NEW.parent_store
    AND parent_record_id = NEW.parent_record_id AND seq = NEW.seq)
BEGIN SELECT RAISE(ABORT, 'child_edges is append-only'); END;
"""


# The last write ID issued in this process. Taken inside a Ledger's _lock
# (always in that order), so IDs strictly increase within the process and,
# per Ledger, with seq.
_id_lock = threading.Lock()
_last_id = 0


def _write_id(ts: int, last: int) -> int:
    """A UUIDv7-layout ID as an int: 48-bit Unix ms, then random bits.

    Time-ordered, so a new write's entries in the ``write_id`` indexes land on
    the rightmost B-tree leaf instead of a random one. With random IDs every
    write in a commit dirtied its own leaf in each index, and the extra WAL
    pages dominated a multi-write commit.

    Strictly greater than ``last``: within a millisecond, or if the clock
    steps back, the ID is ``last`` plus one. Format with ``:032x``.
    """
    ms = max(ts, 0) // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (
        (ms & (2**48 - 1)) << 80
        | 0x7 << 76
        | (rand >> 68) << 64
        | 0b10 << 62
        | rand & (2**62 - 1)
    )
    return value if value > last else last + 1


def _statements(script: str):
    """Split ``script`` into single statements (trigger bodies kept whole)."""
    buf = ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            yield buf
            buf = ""


def _check_schema(conn: sqlite3.Connection, path) -> bool:
    """Raise ``LedgerSchemaError`` unless the file is new or at
    ``SCHEMA_VERSION``; return whether it is new (no version, no tables)."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return False
    if version == 0 and not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchone():
        return True
    found = "unversioned (created before schema versioning)" if version == 0 else version
    raise LedgerSchemaError(
        f"ledger file {str(path)!r} has schema version {found}; this library"
        f" uses version {SCHEMA_VERSION}. Migrate or recreate the file."
    )


def _dedupe_parents(parents: Iterable[Parent]) -> list:
    merged = {}
    for p in parents:
        if not isinstance(p, (tuple, list)) or len(p) != 3:
            raise ValueError(f"parent must be (store, record_id, tag), got {p!r}")
        store, record_id, tag = p
        if tag not in TAGS:
            raise ValueError(f"unknown parent tag {tag!r}; expected one of {TAGS}")
        key = (store, record_id)
        if merged.get(key) != "in_prompt":
            merged[key] = tag
    return [(s, r, t) for (s, r), t in merged.items()]


def _context_set(context: Iterable[Parent]) -> Tuple[Optional[str], list]:
    """Deduped ``context`` members and their content address.

    The address is the sha256 of the sorted members, tags included, so the
    same context always maps to the same set, and a grown or re-tagged
    context to a new one. An empty context has no set (``None``).
    """
    # Keyed as stored: the TEXT columns store a non-str ID as its str().
    members = sorted(_dedupe_parents(context), key=lambda m: (str(m[0]), str(m[1])))
    if not members:
        return None, []
    key = [(str(s), str(r), t) for s, r, t in members]
    digest = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest(), members


def _tombstone_row(t: dict, ts: int) -> tuple:
    reason = t["reason"]
    if reason not in REASONS:
        raise ValueError(f"unknown tombstone reason {reason!r}; expected one of {REASONS}")
    return (
        t["store"], t["record_id"], ts, reason,
        t.get("write_id"), t.get("agent_id"), t.get("session_id"),
    )


class Ledger:
    """Append-only ledger backed by one persistent SQLite connection.

    The connection uses WAL mode with ``synchronous=NORMAL``. Durability
    trade-off: a committed append survives an application crash, but an OS
    crash or power loss can lose the most recent commits, because the WAL is
    only fsynced at checkpoints. The database itself is never corrupted. In
    exchange, commits skip fsync, which keeps append overhead well under a
    millisecond.

    Checkpoints (which do fsync) are kept off the append path: auto-checkpoint
    is disabled on the write connection, and a daemon thread runs a PASSIVE
    checkpoint on its own connection every ``checkpoint_interval`` seconds.
    ``close()`` stops it and runs a final checkpoint.

    The same thread fills ``child_edges`` in short transactions sized to hold
    the write lock ~0.2 ms each. While writes are arriving it pauses ~100x
    that between them, so appends rarely wait behind one; when they stop, it
    catches up at full speed.

    Safe to share across threads: appends are serialized with a lock.
    Quarantine reads use a third connection, so they never wait behind an
    append's commit (WAL readers don't block on the writer).
    """

    def __init__(self, path: Union[str, os.PathLike], checkpoint_interval: float = 1.0):
        self._lock = threading.Lock()
        self._conn = conn = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False
        )
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            # Checked before anything writes to the file, then again under
            # the write lock in case another opener created it meanwhile.
            _check_schema(conn, path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA recursive_triggers=ON")
            conn.execute("PRAGMA wal_autocheckpoint=0")
            conn.execute("BEGIN IMMEDIATE")
            try:
                if _check_schema(conn, path):
                    for stmt in _statements(_SCHEMA):
                        conn.execute(stmt)
                    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        except BaseException:
            conn.close()
            raise

        self._read_lock = threading.Lock()
        self._read_conn = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False
        )
        self._read_conn.execute("PRAGMA busy_timeout=5000")

        self._ckpt_conn = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False
        )
        self._ckpt_conn.execute("PRAGMA busy_timeout=5000")
        self._ckpt_lock = threading.Lock()
        self._stop = threading.Event()
        self._ckpt_thread = threading.Thread(
            target=self._checkpoint_loop,
            args=(checkpoint_interval,),
            name="dod-ledger-checkpoint",
            daemon=True,
        )
        self._ckpt_thread.start()

    def _checkpoint_loop(self, interval: float) -> None:
        next_ckpt = time.monotonic() + interval
        pause = interval
        last_seq = None
        limit = 16
        while not self._stop.wait(pause):
            now = time.monotonic()
            if now >= next_ckpt:
                next_ckpt = now + interval
                try:
                    with self._ckpt_lock:
                        self._ckpt_conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass  # retried next interval
            seq = last_seq
            try:
                with self._ckpt_lock:
                    seq = self._ckpt_conn.execute("SELECT max(seq) FROM writes").fetchone()[0]
                held = self._fill_child_edges(limit)
            except sqlite3.Error:
                held = None  # retried next interval
            if held is None:
                pause = max(0.0, next_ckpt - time.monotonic())
            else:
                fit = int(limit * _FILL_TARGET / max(held, 1e-6))
                limit = max(1, min(_FILL_MAX, 2 * limit, fit))
                # While writes are arriving, hold the write lock a small
                # fraction of the time, so few appends ever wait behind a
                # fill; when they stop, catch up at full speed.
                busy = last_seq is not None and seq != last_seq
                pause = max(_FILL_MIN_PAUSE, held * (_FILL_BUSY_FACTOR if busy else 1))
            last_seq = seq

    def _fill_child_edges(self, limit: int = _FILL_MAX) -> Optional[float]:
        """Index the next writes past the watermark into ``child_edges``.

        At most ``limit`` writes and (unless one write has more) ``limit``
        edges, in one short transaction that also appends the new watermark.
        Idempotent across connections and processes: the transaction
        re-reads the watermark under the write lock and does nothing if
        another connection moved it first. Returns the seconds the write
        lock was held, or ``None`` if every write is already indexed.
        """
        with self._ckpt_lock:
            conn = self._ckpt_conn
            (wm,) = conn.execute(
                "SELECT coalesce(max(upto), 0) FROM child_edges_filled"
            ).fetchone()
            # Upper bound on each write's edges (a direct parent that is
            # also in the context set is one edge, counted twice here).
            rows = conn.execute(
                "SELECT w.seq,"
                " (SELECT count(*) FROM direct_parents d WHERE d.write_id = w.write_id)"
                " + coalesce((SELECT size FROM context_sets s"
                " WHERE s.set_hash = w.context_set), 0)"
                " FROM writes w WHERE w.seq > ? ORDER BY w.seq LIMIT ?",
                (wm, limit),
            ).fetchall()
        if not rows:
            return None
        hi, total = rows[0][0], 0
        for seq, n in rows:
            if total and total + n > limit:
                break
            hi, total = seq, total + n
        with self._lock:
            start = time.perf_counter()
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                (now_wm,) = conn.execute(
                    "SELECT coalesce(max(upto), 0) FROM child_edges_filled"
                ).fetchone()
                if now_wm != wm:
                    conn.execute("ROLLBACK")  # filled by another connection
                    return time.perf_counter() - start
                conn.execute(
                    f"INSERT INTO child_edges {_EDGES_BETWEEN}", {"lo": wm, "hi": hi}
                )
                conn.execute(
                    "INSERT INTO child_edges_filled VALUES (?, ?)", (hi, time.time_ns())
                )
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            return time.perf_counter() - start

    def append_write(
        self,
        *,
        content: Union[str, bytes],
        store: str,
        source: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parents: Iterable[Parent] = (),
        context: Iterable[Parent] = (),
    ) -> str:
        """Record a write and its parent edges in one transaction.

        ``content`` is hashed (sha256) and discarded; it is never stored.
        ``parents`` (the write's own, stored per write) and ``context`` (the
        scope's shared context, stored once per distinct set) are
        ``(store, record_id, tag)`` tuples. The write's parent edges are both
        together: duplicates are one edge, and if they disagree on tag,
        ``in_prompt`` wins.

        Each write gets a strictly increasing ``seq``; order writes by ``seq``.
        ``ts`` is wall-clock time for display only, since clocks can step back.
        """
        (write_id,) = self.append_batch(
            [
                dict(
                    content=content,
                    store=store,
                    source=source,
                    agent_id=agent_id,
                    session_id=session_id,
                    parents=parents,
                    context=context,
                )
            ]
        )
        return write_id

    def append_batch(
        self,
        writes: Iterable[dict] = (),
        bindings: Iterable[Tuple[str, str, str]] = (),
        tombstones: Iterable[dict] = (),
    ) -> List[str]:
        """Record several writes, bindings and tombstones in one transaction.

        Each of ``writes`` is a dict of ``append_write``'s keyword arguments
        (writes passing the same ``context`` object have it hashed once),
        plus an optional ``record_id`` that binds the new write in the same
        transaction. ``bindings`` are ``(write_id, store, record_id)`` tuples
        for writes recorded earlier. ``tombstones`` are dicts as for
        ``append_tombstones``. Returns the new write IDs in order. All or
        nothing: if anything is invalid or fails, nothing is recorded.
        """
        # Everything but the write IDs is prepared outside the lock.
        prepared, binding_rows = [], []
        # id(context) -> (context, set_hash, members, tags): writes in one
        # batch usually pass the same context object, so hash it once.
        contexts = {}
        sets = {}  # set_hash -> members, for sets this batch references
        ts = time.time_ns()
        for w in writes:
            source = w["source"]
            if source not in SOURCES:
                raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
            content = w["content"]
            if isinstance(content, str):
                content = content.encode("utf-8")
            context = w.get("context", ())
            if id(context) not in contexts:
                set_hash, members = _context_set(context)
                tags = {(str(s), str(r)): t for s, r, t in members}
                contexts[id(context)] = (context, set_hash, members, tags)
            _, set_hash, members, tags = contexts[id(context)]
            if set_hash is not None:
                sets[set_hash] = members
            row = (
                hashlib.sha256(content).hexdigest(), ts, w["store"],
                w.get("agent_id"), w.get("session_id"), source, set_hash,
            )
            # A direct parent shadows the context's edge in the parents view,
            # so it carries the merged tag.
            direct = [
                (s, r, "in_prompt" if tags.get((str(s), str(r))) == "in_prompt" else t)
                for s, r, t in _dedupe_parents(w.get("parents", ()))
            ]
            prepared.append((row, direct, w["store"], w.get("record_id")))
        later_bindings = [(wid, s, r, ts) for wid, s, r in bindings]
        tombstone_rows = [_tombstone_row(t, ts) for t in tombstones]
        with self._lock:
            # IDs are issued under the lock, so they increase with seq across
            # concurrent writers. A failed batch just skips its IDs.
            global _last_id
            write_rows, parent_rows, write_ids = [], [], []
            with _id_lock:
                last = _last_id
                for row, direct, store, record_id in prepared:
                    last = _write_id(ts, last)
                    write_id = f"{last:032x}"
                    write_rows.append((write_id,) + row)
                    parent_rows += [(write_id, s, r, t) for s, r, t in direct]
                    if record_id is not None:
                        binding_rows.append((write_id, store, record_id, ts))
                    write_ids.append(write_id)
                _last_id = last
            binding_rows += later_bindings
            conn = self._conn
            # IMMEDIATE takes the write lock up front (waiting via
            # busy_timeout), so the context-set check below reads the latest
            # state. A deferred BEGIN would read a snapshot, and if another
            # connection committed the same set before our first INSERT, the
            # upgrade fails with SQLITE_BUSY_SNAPSHOT, which is never retried.
            conn.execute("BEGIN IMMEDIATE")
            try:
                for set_hash, members in sets.items():
                    # Content-addressed and append-only: a set already
                    # recorded (by any connection) is referenced, not rewritten.
                    if conn.execute(
                        "SELECT 1 FROM context_sets WHERE set_hash = ?", (set_hash,)
                    ).fetchone():
                        continue
                    conn.execute(
                        "INSERT INTO context_sets (set_hash, size, ts) VALUES (?, ?, ?)",
                        (set_hash, len(members), ts),
                    )
                    conn.executemany(
                        "INSERT INTO context_members VALUES (?, ?, ?, ?)",
                        [(set_hash, s, r, t) for s, r, t in members],
                    )
                conn.executemany(
                    "INSERT INTO writes (write_id, content_hash, ts, store, agent_id,"
                    " session_id, source, context_set) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    write_rows,
                )
                conn.executemany("INSERT INTO direct_parents VALUES (?, ?, ?, ?)", parent_rows)
                conn.executemany("INSERT INTO bindings VALUES (?, ?, ?, ?)", binding_rows)
                conn.executemany(
                    "INSERT INTO tombstones (store, record_id, ts, reason, write_id,"
                    " agent_id, session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    tombstone_rows,
                )
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        return write_ids

    def append_tombstones(self, tombstones: Iterable[dict]) -> None:
        """Record that items were deleted from their store, in one transaction.

        Each tombstone is a dict with ``store``, ``record_id`` and ``reason``
        (one of ``REASONS``), plus what caused it: an optional ``write_id``
        (a recorded write whose save deleted the item) and optional
        ``agent_id`` / ``session_id``. Each gets its own strictly increasing
        ``seq`` (separate from writes'). A tombstone never removes the
        item's writes, edges or bindings; queries report it alongside them.
        """
        self.append_batch(tombstones=tombstones)

    def bind(self, write_id: str, store: str, record_id: str) -> None:
        """Append a binding from a write to the record ID its store assigned."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO bindings VALUES (?, ?, ?, ?)",
                (write_id, store, record_id, time.time_ns()),
            )

    def quarantine(
        self, items: Iterable[Item], reason: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> None:
        """Quarantine ``(store, record_id)`` items in one transaction.

        Nothing is deleted: instrumented stores stop returning the items until
        they are released. ``reason`` and ``actor`` are stored as given; keep
        memory content out of them.
        """
        self._append_quarantine(items, "quarantine", reason, actor)

    def release(
        self, items: Iterable[Item], reason: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> None:
        """Release quarantined ``(store, record_id)`` items in one transaction."""
        self._append_quarantine(items, "release", reason, actor)

    def quarantined(self) -> Set[Item]:
        """Items whose latest quarantine action, by ``seq``, is ``quarantine``."""
        with self._read_lock:
            # SQLite takes bare columns from the row max() picked.
            rows = self._read_conn.execute(
                "SELECT store, record_id, action, max(seq) FROM quarantine"
                " GROUP BY store, record_id"
            ).fetchall()
        return {(s, r) for s, r, action, _ in rows if action == "quarantine"}

    def quarantine_log(self, after: int = 0) -> List[Tuple[int, str, str, str]]:
        """Quarantine actions with ``seq > after``, as ``(seq, store,
        record_id, action)`` in ``seq`` order.

        Lets a cache of ``quarantined()`` catch up on what changed since the
        last ``seq`` it saw, from any connection, without re-reading the table.
        """
        with self._read_lock:
            return self._read_conn.execute(
                "SELECT seq, store, record_id, action FROM quarantine"
                " WHERE seq > ? ORDER BY seq",
                (after,),
            ).fetchall()

    def _append_quarantine(self, items, action, reason, actor) -> None:
        ts = time.time_ns()
        rows = []
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(f"item must be (store, record_id), got {item!r}")
            store, record_id = item
            rows.append((ts, store, record_id, action, reason, actor))
        if not rows:
            return
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN")
            try:
                conn.executemany(
                    "INSERT INTO quarantine (ts, store, record_id, action, reason, actor)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._ckpt_thread.join()
        self._ckpt_conn.close()
        with self._read_lock:
            self._read_conn.close()
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
