"""Append-only SQLite ledger of memory writes, their parent edges, and deletions.

A memory item is identified by ``(store, record_id)``, so edges can cross
stores. A deleted item gets a tombstone row; nothing is ever removed. The
ledger never stores memory content: only IDs, content hashes, edges, and
metadata.
"""

import hashlib
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

_TABLES = ("writes", "parents", "bindings", "tombstones", "quarantine")

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
        CHECK (source IN ('user', 'tool', 'document', 'agent_inference'))
);
CREATE TABLE IF NOT EXISTS parents (
    write_id         TEXT NOT NULL REFERENCES writes (write_id),
    parent_store     TEXT NOT NULL,
    parent_record_id TEXT NOT NULL,
    tag              TEXT NOT NULL
        CHECK (tag IN ('in_prompt', 'retrieved_in_scope')),
    PRIMARY KEY (write_id, parent_store, parent_record_id)
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
            "parents",
            "\n    OR EXISTS (SELECT 1 FROM parents WHERE write_id = NEW.write_id"
            "\n        AND parent_store = NEW.parent_store"
            " AND parent_record_id = NEW.parent_record_id)",
        ),
        ("bindings", ""),
        ("tombstones", ""),
        ("quarantine", ""),
    )
)


_id_lock = threading.Lock()
_last_id = 0


def _write_id(ts: int) -> str:
    """A UUIDv7-layout ID: 48-bit Unix ms, then random bits.

    Time-ordered, so a new write's entries in the ``write_id`` indexes land on
    the rightmost B-tree leaf instead of a random one. With random IDs every
    write in a commit dirtied its own leaf in each index, and the extra WAL
    pages dominated a multi-write commit.

    IDs from one process are strictly increasing: within a millisecond, or if
    the clock steps back, the next ID is the previous one plus one.
    """
    global _last_id
    ms = max(ts, 0) // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (
        (ms & (2**48 - 1)) << 80
        | 0x7 << 76
        | (rand >> 68) << 64
        | 0b10 << 62
        | rand & (2**62 - 1)
    )
    with _id_lock:
        if value <= _last_id:
            value = _last_id + 1
        _last_id = value
    return f"{value:032x}"


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

    Safe to share across threads: appends are serialized with a lock.
    Quarantine reads use a third connection, so they never wait behind an
    append's commit (WAL readers don't block on the writer).
    """

    def __init__(self, path: Union[str, os.PathLike], checkpoint_interval: float = 1.0):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA recursive_triggers=ON")
        self._conn.execute("PRAGMA wal_autocheckpoint=0")
        self._conn.executescript(_SCHEMA)

        self._read_lock = threading.Lock()
        self._read_conn = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False
        )
        self._read_conn.execute("PRAGMA busy_timeout=5000")

        self._ckpt_conn = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False
        )
        self._ckpt_conn.execute("PRAGMA busy_timeout=5000")
        self._stop = threading.Event()
        self._ckpt_thread = threading.Thread(
            target=self._checkpoint_loop,
            args=(checkpoint_interval,),
            name="dod-ledger-checkpoint",
            daemon=True,
        )
        self._ckpt_thread.start()

    def _checkpoint_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self._ckpt_conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass  # retried next interval

    def append_write(
        self,
        *,
        content: Union[str, bytes],
        store: str,
        source: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parents: Iterable[Parent] = (),
    ) -> str:
        """Record a write and its parent edges in one transaction.

        ``content`` is hashed (sha256) and discarded; it is never stored.
        ``parents`` are ``(store, record_id, tag)`` tuples. Duplicates are
        stored once; if they disagree on tag, ``in_prompt`` wins.

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

        Each of ``writes`` is a dict of ``append_write``'s keyword arguments,
        plus an optional ``record_id`` that binds the new write in the same
        transaction. ``bindings`` are ``(write_id, store, record_id)`` tuples
        for writes recorded earlier. ``tombstones`` are dicts as for
        ``append_tombstones``. Returns the new write IDs in order. All or
        nothing: if anything is invalid or fails, nothing is recorded.
        """
        write_rows, parent_rows, binding_rows, write_ids = [], [], [], []
        ts = time.time_ns()
        for w in writes:
            source = w["source"]
            if source not in SOURCES:
                raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
            content = w["content"]
            if isinstance(content, str):
                content = content.encode("utf-8")
            write_id = _write_id(ts)
            write_rows.append((
                write_id, hashlib.sha256(content).hexdigest(), ts, w["store"],
                w.get("agent_id"), w.get("session_id"), source,
            ))
            parent_rows += [(write_id, s, r, t) for s, r, t in _dedupe_parents(w.get("parents", ()))]
            if w.get("record_id") is not None:
                binding_rows.append((write_id, w["store"], w["record_id"], ts))
            write_ids.append(write_id)
        binding_rows += [(wid, s, r, ts) for wid, s, r in bindings]
        tombstone_rows = [_tombstone_row(t, ts) for t in tombstones]
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN")
            try:
                conn.executemany(
                    "INSERT INTO writes (write_id, content_hash, ts, store,"
                    " agent_id, session_id, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    write_rows,
                )
                conn.executemany("INSERT INTO parents VALUES (?, ?, ?, ?)", parent_rows)
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
