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

A context that grew is stored as a delta: a base set plus the members added
or upgraded to ``in_prompt`` since it. ``context_chain`` lists, per set, the
sets whose ``context_members`` rows make it up (itself included). Bases are
picked Fenwick-style by ``step`` (deltas since the last full set), so a chain
has at most ``log2(_CHAIN_STEPS) + 1`` parts and a full set is re-stored
every ``_CHAIN_STEPS`` steps.

The lineage index holds the same lineage as integers, for walking forward,
in the shape it is stored in rather than one row per edge, so its size
follows what the ledger stores: ``items`` interns each ``(store,
record_id)``; ``direct_children`` lists the writes that name an item as a
direct parent; ``part_items`` the context sets (by ``seq``) whose own member
rows hold an item; ``set_parts`` the sets whose chain includes a part;
``set_writes`` the writes that use each set; and ``write_items`` the items
bound to each write. It is derived, so it lives in a companion file (the
ledger's path plus ``-lineage``, attached as ``lineage``) that is never
written on the append path: the background thread fills it after the fact,
in transactions that take only that file's write lock, and ``index_filled``
records how far (the highest write ``seq``, context set ``seq`` and
``bindings`` rowid indexed, each with the key of the row it stopped at).
Queries read what lies past those watermarks from the source tables. A
companion file at another index version, unreadable, or filled from a
different ledger (its keys don't match) is deleted and rebuilt.

A file records its schema version (``PRAGMA user_version``); opening one at
another version, or an unversioned file with tables, raises
``LedgerSchemaError`` before anything is read or written.
"""

import hashlib
import itertools
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Iterable, List, Optional, Set, Tuple, Union

SOURCES = ("user", "tool", "document", "agent_inference")
TAGS = ("in_prompt", "retrieved_in_scope")
REASONS = ("consolidation", "excision", "forget", "other")

Item = Tuple[str, str]
Parent = Tuple[str, str, str]

# Stored in the file's ``PRAGMA user_version``. Bump it on any change to
# tables, columns, triggers or views, and ship a migration (or a "recreate"
# note in docs/decisions.md) with the bump.
SCHEMA_VERSION = 5  # 5: bindings.content_hash; 4: child_edges dropped; 3: delta context sets (#52)
# Older files migrate in place, one version at a time: version 4 only drops
# the derived child_edges index, which the lineage file replaces; version 5
# adds a nullable column.
_MIGRATIONS = {
    3: ("DROP TABLE IF EXISTS child_edges", "DROP TABLE IF EXISTS child_edges_filled"),
    4: ("ALTER TABLE bindings ADD COLUMN content_hash TEXT",),
}
# The lineage file's ``PRAGMA user_version``. It is derived, so a file at
# another version is rebuilt rather than migrated.
INDEX_VERSION = 1

# A context set's chain restarts with a full set every _CHAIN_STEPS steps, so
# chains have at most log2(_CHAIN_STEPS) + 1 = 13 parts.
_CHAIN_STEPS = 4096
# Recent context sets kept in memory per (agent_id, session_id), as bases for
# the next grown context. Each holds its members, so this bounds memory.
_CHAIN_HINTS = 256


class LedgerSchemaError(Exception):
    """A ledger file's schema version differs from this library's."""

_TABLES = (
    "writes", "direct_parents", "context_sets", "context_members", "context_chain",
    "bindings", "tombstones", "quarantine",
)
_INDEX_TABLES = (
    "items", "direct_children", "part_items", "set_parts", "set_writes", "write_items",
    "index_filled",
)

# Context edges of writes ``w``: every member row of every part of the
# write's set, except a direct parent's (which carries the merged tag) and a
# retrieved_in_scope row that a later part upgraded to in_prompt.
_CONTEXT_EDGES = """
FROM writes w JOIN context_sets s ON s.set_hash = w.context_set
JOIN context_chain c ON c.set_hash = w.context_set
JOIN context_members m ON m.set_hash = c.part
WHERE NOT EXISTS (
    SELECT 1 FROM direct_parents d
    WHERE d.write_id = w.write_id AND d.parent_store = m.store
        AND d.parent_record_id = m.record_id
) AND (m.tag = 'in_prompt' OR s.upgrades = 0 OR NOT EXISTS (
    SELECT 1 FROM context_chain c2 JOIN context_members m2 ON m2.set_hash = c2.part
    WHERE c2.set_hash = w.context_set AND m2.store = m.store
        AND m2.record_id = m.record_id AND m2.tag = 'in_prompt'
))"""

# A fill indexes writes (:lo, :hi], the context sets (:slo, :shi] and
# bindings (:blo, :bhi], entirely in SQL on the fill connection (the ledger,
# with the lineage file attached): it interns their items, then appends the
# integer rows. A binding appended twice is one row.
#
# New names and bindings are deduplicated in staging tables on the fill
# connection (in memory, emptied after each use) rather than with DISTINCT,
# UNION or an INSERT that reads its own target, each of which builds and
# frees a temporary b-tree per statement. SQLite's allocator takes a
# process-wide mutex, so that churn would slow appends on other threads.
_FILL_STAGING = (
    "CREATE TEMP TABLE IF NOT EXISTS fill_names (store TEXT NOT NULL,"
    " record_id TEXT NOT NULL, PRIMARY KEY (store, record_id)) WITHOUT ROWID",
    "CREATE TEMP TABLE IF NOT EXISTS fill_binds (seq INTEGER NOT NULL,"
    " item INTEGER NOT NULL, PRIMARY KEY (seq, item)) WITHOUT ROWID",
)
_FILL_SQL = (
    """INSERT OR IGNORE INTO temp.fill_names
    SELECT d.parent_store, d.parent_record_id
    FROM writes w CROSS JOIN direct_parents d ON d.write_id = w.write_id
    WHERE w.seq > :lo AND w.seq <= :hi""",
    """INSERT OR IGNORE INTO temp.fill_names
    SELECT m.store, m.record_id
    FROM context_sets cs CROSS JOIN context_members m ON m.set_hash = cs.set_hash
    WHERE cs.seq > :slo AND cs.seq <= :shi""",
    """INSERT OR IGNORE INTO temp.fill_names
    SELECT store, record_id FROM bindings WHERE rowid > :blo AND rowid <= :bhi""",
    """DELETE FROM temp.fill_names WHERE EXISTS (
        SELECT 1 FROM lineage.items i
        WHERE i.store = fill_names.store AND i.record_id = fill_names.record_id
    )""",
    "INSERT INTO lineage.items (store, record_id) SELECT store, record_id FROM temp.fill_names",
    "DELETE FROM temp.fill_names",
    """INSERT INTO lineage.direct_children
    SELECT i.item, w.seq, d.tag = 'in_prompt'
    FROM writes w CROSS JOIN direct_parents d ON d.write_id = w.write_id
    CROSS JOIN lineage.items i ON i.store = d.parent_store AND i.record_id = d.parent_record_id
    WHERE w.seq > :lo AND w.seq <= :hi""",
    """INSERT INTO lineage.part_items
    SELECT i.item, cs.seq, m.tag = 'in_prompt'
    FROM context_sets cs CROSS JOIN context_members m ON m.set_hash = cs.set_hash
    CROSS JOIN lineage.items i ON i.store = m.store AND i.record_id = m.record_id
    WHERE cs.seq > :slo AND cs.seq <= :shi""",
    """INSERT INTO lineage.set_parts
    SELECT p.seq, s.seq FROM context_sets s
    CROSS JOIN context_chain c ON c.set_hash = s.set_hash
    CROSS JOIN context_sets p ON p.set_hash = c.part
    WHERE s.seq > :slo AND s.seq <= :shi""",
    """INSERT INTO lineage.set_writes
    SELECT cs.seq, w.seq FROM writes w
    CROSS JOIN context_sets cs ON cs.set_hash = w.context_set
    WHERE w.seq > :lo AND w.seq <= :hi""",
    """INSERT OR IGNORE INTO temp.fill_binds
    SELECT w.seq, i.item FROM bindings b
    CROSS JOIN writes w ON w.write_id = b.write_id
    CROSS JOIN lineage.items i ON i.store = b.store AND i.record_id = b.record_id
    WHERE b.rowid > :blo AND b.rowid <= :bhi""",
    """DELETE FROM temp.fill_binds WHERE EXISTS (
        SELECT 1 FROM lineage.write_items x
        WHERE x.seq = fill_binds.seq AND x.item = fill_binds.item
    )""",
    "INSERT INTO lineage.write_items SELECT seq, item FROM temp.fill_binds",
    "DELETE FROM temp.fill_binds",
    """INSERT INTO lineage.index_filled (
        writes_upto, write_key, sets_upto, set_key, bindings_upto, binding_key, ts)
    VALUES (:hi, (SELECT write_id FROM writes WHERE seq = :hi),
        :shi, (SELECT set_hash FROM context_sets WHERE seq = :shi),
        :bhi, (SELECT write_id FROM bindings WHERE rowid = :bhi), :ts)""",
)

# Whether the lineage file was filled from this ledger: the rows its newest
# watermarks stopped at are still there, with the same keys.
_INDEX_MATCHES = """
SELECT f.write_key IS (SELECT write_id FROM writes WHERE seq = f.writes_upto)
    AND f.set_key IS (SELECT set_hash FROM context_sets WHERE seq = f.sets_upto)
    AND f.binding_key IS (SELECT write_id FROM bindings WHERE rowid = f.bindings_upto)
FROM lineage.index_filled f ORDER BY f.seq DESC LIMIT 1"""

# Most writes (and, separately, bindings) one fill transaction indexes, while
# writes are arriving (busy) and when they aren't (idle).
_FILL_BUSY_MAX = 200
_FILL_MAX = 1000
# A fill never holds the ledger's write lock, but it still slows appends on
# other threads while it runs: SQLite's allocator takes one process-wide
# mutex (its memory statistics) on every allocation, so SQLite work on any
# thread contends with every append. While writes are arriving, the fill
# runs at most _FILL_BUSY_SHARE of the time; that keeps up with sustained
# writes at realistic rates, and under heavier load the index lags (queries
# read the rest from the source tables) and catches up once writes pause.
_FILL_BUSY_SHARE = 0.02
# Writes count as arriving until none has for _FILL_QUIET seconds. An idle
# background thread looks for new writes every _FILL_IDLE_POLL seconds, and
# fills back to back while a full batch or more is waiting.
_FILL_QUIET = 0.1
_FILL_IDLE_POLL = 0.02

def _append_only(tables, prefix: str = "", replace=()) -> str:
    """Triggers that reject UPDATE and DELETE on ``tables`` and an INSERT
    that would replace a row. ``replace`` maps each table to the condition
    on ``NEW`` under which an insert conflicts (the rowid by default)."""
    keys = dict(replace)
    out = "".join(
        f"""
CREATE TRIGGER IF NOT EXISTS {prefix}{t}_no_{op.lower()} BEFORE {op} ON {t}
BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
"""
        for t in tables
        for op in ("UPDATE", "DELETE")
    )
    # REPLACE deletes the conflicting row without firing DELETE triggers
    # unless the writing connection enabled recursive_triggers, which is a
    # per-connection setting. Reject the conflict before it happens instead.
    return out + "".join(
        f"""
CREATE TRIGGER IF NOT EXISTS {prefix}{t}_no_replace BEFORE INSERT ON {t}
WHEN EXISTS (SELECT 1 FROM {t} WHERE {keys.get(t, "rowid = NEW.rowid")})
BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
"""
        for t in tables
    )


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
    ts       INTEGER NOT NULL,
    base     TEXT REFERENCES context_sets (set_hash),
    step     INTEGER NOT NULL,
    upgrades INTEGER NOT NULL CHECK (upgrades IN (0, 1))
);
CREATE TABLE IF NOT EXISTS context_chain (
    set_hash TEXT NOT NULL REFERENCES context_sets (set_hash),
    part     TEXT NOT NULL REFERENCES context_sets (set_hash),
    PRIMARY KEY (set_hash, part)
) WITHOUT ROWID;
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
SELECT w.write_id, m.store, m.record_id, m.tag""" + _CONTEXT_EDGES + """;
CREATE TABLE IF NOT EXISTS bindings (
    write_id     TEXT NOT NULL REFERENCES writes (write_id),
    store        TEXT NOT NULL,
    record_id    TEXT NOT NULL,
    ts           INTEGER NOT NULL,
    content_hash TEXT
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
    )
) + """
CREATE TRIGGER IF NOT EXISTS context_chain_no_replace BEFORE INSERT ON context_chain
WHEN EXISTS (SELECT 1 FROM context_chain WHERE set_hash = NEW.set_hash
    AND part = NEW.part)
BEGIN SELECT RAISE(ABORT, 'context_chain is append-only'); END;
"""

# The lineage file, attached as ``lineage``. Part and set IDs are
# ``context_sets.seq``; a tag is 1 for in_prompt.
_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS lineage.items (
    item      INTEGER PRIMARY KEY,
    store     TEXT NOT NULL,
    record_id TEXT NOT NULL,
    UNIQUE (store, record_id)
);
CREATE TABLE IF NOT EXISTS lineage.direct_children (
    item INTEGER NOT NULL,
    seq  INTEGER NOT NULL,
    tag  INTEGER NOT NULL CHECK (tag IN (0, 1)),
    PRIMARY KEY (item, seq)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS lineage.part_items (
    item INTEGER NOT NULL,
    part INTEGER NOT NULL,
    tag  INTEGER NOT NULL CHECK (tag IN (0, 1)),
    PRIMARY KEY (item, part)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS lineage.set_parts (
    part INTEGER NOT NULL,
    sid  INTEGER NOT NULL,
    PRIMARY KEY (part, sid)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS lineage.set_writes (
    sid INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    PRIMARY KEY (sid, seq)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS lineage.write_items (
    seq  INTEGER NOT NULL,
    item INTEGER NOT NULL,
    PRIMARY KEY (seq, item)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS lineage.write_items_by_item ON write_items (item, seq);
CREATE TABLE IF NOT EXISTS lineage.index_filled (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    writes_upto   INTEGER NOT NULL,
    write_key     TEXT,
    sets_upto     INTEGER NOT NULL,
    set_key       TEXT,
    bindings_upto INTEGER NOT NULL,
    binding_key   TEXT,
    ts            INTEGER NOT NULL
);
""" + _append_only(
    _INDEX_TABLES,
    "lineage.",
    {
        "items": "item = NEW.item OR (store = NEW.store AND record_id = NEW.record_id)",
        "direct_children": "item = NEW.item AND seq = NEW.seq",
        "part_items": "item = NEW.item AND part = NEW.part",
        "set_parts": "part = NEW.part AND sid = NEW.sid",
        "set_writes": "sid = NEW.sid AND seq = NEW.seq",
        "write_items": "seq = NEW.seq AND item = NEW.item",
    },
)


_INSERT_BINDING = (
    "INSERT INTO bindings (write_id, store, record_id, ts, content_hash) VALUES (?, ?, ?, ?, ?)"
)

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


def _check_schema(conn: sqlite3.Connection, path) -> int:
    """Raise ``LedgerSchemaError`` unless the file is new, at
    ``SCHEMA_VERSION``, or at a version ``_MIGRATIONS`` upgrades; return its
    version (0 if new: no version, no tables)."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION or version in _MIGRATIONS:
        return version
    if version == 0 and not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchone():
        return 0
    found = "unversioned (created before schema versioning)" if version == 0 else version
    raise LedgerSchemaError(
        f"ledger file {str(path)!r} has schema version {found}; this library"
        f" uses version {SCHEMA_VERSION}. Migrate or recreate the file."
    )


def _remove_index(path: str) -> None:
    """Delete a lineage file and its WAL companions."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.remove(path + suffix)
        except FileNotFoundError:
            pass


def _attach_index(conn: sqlite3.Connection, path: str) -> None:
    """Attach the lineage file at ``path`` (``:memory:`` for an in-memory
    ledger) to ``conn`` as ``lineage``, creating it if new and rebuilding it
    if it is at another index version, unreadable, or filled from a
    different ledger."""
    try:
        conn.execute("ATTACH DATABASE ? AS lineage", (path,))
        attached = True
        version = conn.execute("PRAGMA lineage.user_version").fetchone()[0]
        stale = version not in (0, INDEX_VERSION) or (
            version == INDEX_VERSION
            and (conn.execute(_INDEX_MATCHES).fetchone() or (1,))[0] != 1
        )
    except sqlite3.DatabaseError:  # not a database, or damaged
        attached = "lineage" in {r[1] for r in conn.execute("PRAGMA database_list")}
        version, stale = 0, True
    if stale and path != ":memory:":
        if attached:
            conn.execute("DETACH DATABASE lineage")
        _remove_index(path)
        conn.execute("ATTACH DATABASE ? AS lineage", (path,))
        version = 0
    conn.execute("PRAGMA lineage.journal_mode=WAL")
    conn.execute("PRAGMA lineage.synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    for stmt in _FILL_STAGING:
        conn.execute(stmt)
    if version != INDEX_VERSION:
        # Deferred: the first CREATE takes the file's write lock (waiting out
        # another opener), and IF NOT EXISTS makes a second creation a no-op.
        conn.execute("BEGIN")
        try:
            for stmt in _statements(_INDEX_SCHEMA):
                conn.execute(stmt)
            conn.execute(f"PRAGMA lineage.user_version={INDEX_VERSION}")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")


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


# A read context: (the input as a list, or None if it can't be reused;
# set_hash; members as {(store, record_id): tag}; sorted member keys).
_Read = Tuple[Optional[list], Optional[str], dict, list]


def _context_set(context: Iterable[Parent], prev: Optional[_Read] = None) -> _Read:
    """Deduped ``context`` members and their content address.

    The address is the sha256 of the sorted members, tags included, so the
    same context always maps to the same set (however it is stored), and a
    grown or re-tagged context to a new one. An empty context has no set
    (``None``).

    A scope's context grows by appending, so when ``context`` starts with
    ``prev``'s input only the rest is read. Inputs with a non-str ID aren't
    reused, since an equal ID of another type (``1`` and ``1.0``) is stored
    as a different string.
    """
    items = list(context)
    reusable = prev is not None and prev[0] is not None
    n = len(prev[0]) if reusable else 0
    if reusable and len(items) >= n and items[:n] == prev[0]:
        if len(items) == n:
            return prev
        _, _, merged, keys = prev
        merged, tail = merged.copy(), items[n:]
    else:
        merged, keys, tail = {}, [], items
    added = []
    for p in tail:
        if p.__class__ is not tuple and not isinstance(p, (tuple, list)) or len(p) != 3:
            raise ValueError(f"parent must be (store, record_id, tag), got {p!r}")
        s, r, t = p
        if t not in TAGS:
            raise ValueError(f"unknown parent tag {t!r}; expected one of {TAGS}")
        if s.__class__ is not str or r.__class__ is not str:
            # Keyed as stored: the TEXT columns store a non-str ID as its str().
            s, r, items = str(s), str(r), None
        key = (s, r)
        if key not in merged:
            added.append(key)
            merged[key] = t
        elif t == "in_prompt":
            merged[key] = t
    if not merged:
        return items, None, {}, []
    if added:
        keys = sorted(keys + added)
    tags = [merged[k] for k in keys]
    # Fields joined by NUL: unambiguous when no ID contains one, which the
    # count proves; otherwise JSON.
    blob = "\0".join(itertools.chain.from_iterable(keys))
    if blob.count("\0") == 2 * len(keys) - 1:
        blob = "\0" + blob + "\0" + "\0".join(tags)
    else:
        blob = _ENCODER.encode([keys, tags])
    return items, hashlib.sha256(blob.encode("utf-8")).hexdigest(), merged, keys


_ENCODER = json.JSONEncoder(separators=(",", ":"), check_circular=False)

# A chain of context sets, oldest first: (step, set_hash, members it stores,
# whether the chain upgrades any member). The first is a full set at step 0;
# each later one is a delta over the last earlier one at its Fenwick base step.
_Chain = Tuple[Tuple[int, str, dict, bool], ...]


def _covers(members: dict, last: dict) -> bool:
    """Whether ``members`` hold every member of ``last``, with the same tag
    or upgraded to ``in_prompt``."""
    if last.items() <= members.items():
        return True
    return all(members.get(k) in (t, "in_prompt") for k, t in last.items())


def _grow(hint: Optional[Tuple[_Chain, dict]], set_hash: str, members: dict) -> _Chain:
    """The chain storing new set ``set_hash`` (``members``): a delta over the
    hint's chain (and its last set's members) where those are a subset, else
    a full set.

    The base of step ``n`` is step ``n & (n - 1)`` (``n`` with its lowest set
    bit cleared), so a set's chain has one part per set bit of its step, and
    step ``n`` stores the members added in the last ``n & -n`` steps.
    """
    if hint:
        chain, last = hint
        step = chain[-1][0] + 1
        if step < _CHAIN_STEPS and _covers(members, last):
            prefix = tuple(part for part in chain if part[0] <= step & (step - 1))
            base: dict = {}
            for _, _, stored, _ in prefix:
                base.update(stored)  # a later part only upgrades a tag
            # Added members, and members whose tag was upgraded.
            delta = {k: members[k] for k in members.keys() - base.keys()}
            upgrades = prefix[-1][3]
            if not base.items() <= members.items():
                upgrades = True
                delta.update((k, "in_prompt") for k, t in base.items() if members[k] != t)
            return prefix + ((step, set_hash, delta, upgrades),)
    return ((0, set_hash, members, False),)


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

    Unless ``background_index`` is false (the index is then filled only by
    calling ``_fill_index``), the same thread fills the lineage file. A fill
    never takes the ledger's write lock, so appends don't wait for it; while
    writes are arriving it still runs at most ~2% of the time (see
    ``_FILL_BUSY_SHARE``), and when they stop it catches up at full speed.

    Safe to share across threads: appends are serialized with a lock.
    Quarantine reads use a third connection, so they never wait behind an
    append's commit (WAL readers don't block on the writer).
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        checkpoint_interval: float = 1.0,
        background_index: bool = True,
    ):
        self._lock = threading.Lock()
        # (agent_id, session_id) -> (chain, members) of the last context set
        # this Ledger stored for it, most recently used last. Guarded by _lock.
        self._chains: "OrderedDict[tuple, Tuple[_Chain, dict]]" = OrderedDict()
        # (agent_id, session_id) -> the last context read for it, so a grown
        # context reads only its new items.
        self._reads_lock = threading.Lock()
        self._reads: "OrderedDict[tuple, _Read]" = OrderedDict()
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
                found = version = _check_schema(conn, path)
                if version == 0:
                    for stmt in _statements(_SCHEMA):
                        conn.execute(stmt)
                    version = SCHEMA_VERSION
                while version in _MIGRATIONS:
                    for stmt in _MIGRATIONS[version]:
                        conn.execute(stmt)
                    version += 1
                if version != found:
                    conn.execute(f"PRAGMA user_version={version}")
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

        # The ledger's file; empty for an in-memory ledger.
        self._file = next(
            (f for _, name, f in conn.execute("PRAGMA database_list") if name == "main"), ""
        )
        # Idle connections for read-only walks (``dod.query``), kept so a walk
        # doesn't pay to open a connection, parse the schema, create its temp
        # tables and warm a page cache. Each walk takes one and gives it back.
        self._walk_pool: List[sqlite3.Connection] = []
        self._walk_pool_lock = threading.Lock()

        self._ckpt_conn = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False
        )
        self._ckpt_conn.execute("PRAGMA busy_timeout=5000")
        self._ckpt_lock = threading.Lock()
        # The lineage index is filled on the checkpoint connection, which
        # never writes the ledger. An in-memory ledger has no file to open
        # twice, so its index is in memory too, filled on the write connection.
        if self._file:
            self._index_path = self._file + "-lineage"
            self._index_conn, self._index_lock = self._ckpt_conn, self._ckpt_lock
        else:
            self._index_path = ":memory:"
            self._index_conn, self._index_lock = conn, self._lock
        try:
            _attach_index(self._index_conn, self._index_path)
        except BaseException:
            for c in (self._ckpt_conn, self._read_conn, conn):
                c.close()
            raise
        self._stop = threading.Event()
        self._ckpt_thread = threading.Thread(
            target=self._checkpoint_loop,
            args=(checkpoint_interval, background_index),
            name="dod-ledger-checkpoint",
            daemon=True,
        )
        self._ckpt_thread.start()

    def _checkpoint_loop(self, interval: float, index: bool) -> None:
        next_ckpt = time.monotonic() + interval
        pause = min(interval, _FILL_IDLE_POLL) if index else interval
        last_change, seen = -1e9, None
        # While writes are arriving, the next fill waits until this time, so
        # fills take at most _FILL_BUSY_SHARE of the time.
        rest_until = 0.0
        while not self._stop.wait(pause):
            if time.monotonic() >= next_ckpt:
                next_ckpt = time.monotonic() + interval
                try:
                    with self._ckpt_lock:
                        # Every attached file: the ledger and its lineage file.
                        self._ckpt_conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass  # retried next interval
            until_ckpt = max(0.0, next_ckpt - time.monotonic())
            if not index:
                pause = until_ckpt
                continue
            try:
                with self._ckpt_lock:
                    now = self._ckpt_conn.execute(
                        "SELECT (SELECT max(seq) FROM writes), (SELECT max(rowid) FROM bindings)"
                    ).fetchone()
                if now != seen:
                    if seen is not None:
                        last_change = time.monotonic()
                    seen = now
                busy = time.monotonic() - last_change < _FILL_QUIET
                if busy and time.monotonic() < rest_until:
                    # Resting, but look again within _FILL_QUIET: once writes
                    # stop, catch up right away instead of resting it out.
                    pause = min(rest_until - time.monotonic(), _FILL_QUIET, until_ckpt)
                    continue
                limit = _FILL_BUSY_MAX if busy else _FILL_MAX
                began = time.monotonic()
                filled = self._fill_index(limit)
                spent = time.monotonic() - began
            except sqlite3.Error:
                busy, filled, spent = False, None, 0.0  # retried next round
            if busy:
                rest_until = time.monotonic() + spent * (1 / _FILL_BUSY_SHARE - 1)
                pause = min(rest_until - time.monotonic(), _FILL_QUIET) if filled else (
                    _FILL_IDLE_POLL)
            elif filled and filled >= limit:
                pause = 0.0  # idle and a full batch behind: catch up
            else:
                pause = _FILL_IDLE_POLL
            pause = max(0.0, min(pause, until_ckpt))

    def _index_watermarks(self, conn: sqlite3.Connection) -> Tuple[int, int, int]:
        """``(writes_upto, sets_upto, bindings_upto)``: the last write
        ``seq``, context set ``seq`` and ``bindings`` rowid the lineage index
        covers. Every context set an indexed write uses is indexed. Reads
        only the lineage file, so called first in a transaction, it fixes
        that file's snapshot before the ledger's: the ledger then holds
        everything the index was filled from."""
        row = conn.execute(
            "SELECT writes_upto, sets_upto, bindings_upto FROM lineage.index_filled"
            " ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row if row else (0, 0, 0)

    def _fill_index(self, limit: int = _FILL_MAX) -> Optional[int]:
        """Index up to ``limit`` writes past the watermarks, the context sets
        they use, and up to ``limit`` bindings, in one transaction on the
        lineage file. Returns how many writes and bindings it indexed, or
        ``None`` if everything already is.

        Runs entirely in SQL, so it holds neither the ledger's write lock nor
        the GIL: appends never wait for it. Idempotent across connections and
        processes: it takes the lineage file's write lock before reading the
        watermarks, so two fills never index the same rows.
        """
        with self._index_lock:
            conn = self._index_conn
            conn.execute("BEGIN")
            try:
                # Take the lineage file's write lock first (a no-op write), so
                # the transaction reads its newest state and can't fail later
                # on a stale snapshot when it starts writing.
                conn.execute("DELETE FROM lineage.index_filled WHERE 0")
                wm, swm, bwm = self._index_watermarks(conn)
                top, btop = conn.execute(
                    "SELECT (SELECT coalesce(max(seq), 0) FROM writes),"
                    " (SELECT coalesce(max(rowid), 0) FROM bindings)"
                ).fetchone()
                if top <= wm and btop <= bwm:
                    conn.execute("ROLLBACK")
                    return None
                hi, bhi = min(top, wm + limit), min(btop, bwm + limit)
                # The sets these writes use. A set is appended with (or
                # before) its first write, and a chain's parts before the set,
                # so indexing every set up to the newest one used covers them.
                (used,) = conn.execute(
                    "SELECT max(cs.seq) FROM writes w"
                    " CROSS JOIN context_sets cs ON cs.set_hash = w.context_set"
                    " WHERE w.seq > ? AND w.seq <= ?",
                    (wm, hi),
                ).fetchone()
                args = {
                    "lo": wm, "hi": hi, "slo": swm, "shi": max(swm, used or 0),
                    "blo": bwm, "bhi": bhi, "ts": time.time_ns(),
                }
                for sql in _FILL_SQL:
                    conn.execute(sql, args if ":" in sql else ())
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            return (hi - wm) + (bhi - bwm)

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
        bindings: Iterable[Tuple[str, ...]] = (),
        tombstones: Iterable[dict] = (),
    ) -> List[str]:
        """Record several writes, bindings and tombstones in one transaction.

        Each of ``writes`` is a dict of ``append_write``'s keyword arguments
        (writes passing the same ``context`` object have it hashed once),
        plus an optional ``record_id`` that binds the new write in the same
        transaction. ``bindings`` are ``(write_id, store, record_id)`` tuples,
        optionally with a fourth item, the binding's ``content_hash`` (see ``bind``),
        for writes recorded earlier. ``tombstones`` are dicts as for
        ``append_tombstones``. Returns the new write IDs in order. All or
        nothing: if anything is invalid or fails, nothing is recorded.
        """
        # Everything but the write IDs is prepared outside the lock.
        prepared, binding_rows = [], []
        # id(context) -> (context, set_hash, members): writes in one batch
        # usually pass the same context object, so hash it once.
        contexts = {}
        # set_hash -> (members, chain hint key), for sets this batch references
        sets = {}
        ts = time.time_ns()
        for w in writes:
            source = w["source"]
            if source not in SOURCES:
                raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
            content = w["content"]
            if isinstance(content, str):
                content = content.encode("utf-8")
            context = w.get("context", ())
            key = (w.get("agent_id"), w.get("session_id"))
            if id(context) not in contexts:
                with self._reads_lock:
                    prev = self._reads.get(key)
                read = _context_set(context, prev)
                if read[0] is not None:
                    with self._reads_lock:
                        self._reads[key] = read
                        self._reads.move_to_end(key)
                        if len(self._reads) > _CHAIN_HINTS:
                            self._reads.popitem(last=False)
                contexts[id(context)] = (context, read[1], read[2])
            _, set_hash, tags = contexts[id(context)]
            if set_hash is not None and set_hash not in sets:
                sets[set_hash] = (tags, key)
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
        later_bindings = [(b[0], b[1], b[2], ts, b[3] if len(b) > 3 else None)
                          for b in bindings]
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
                        binding_rows.append((write_id, store, record_id, ts, None))
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
            hints = {}  # chains this batch stored, by hint key
            try:
                for set_hash, (members, key) in sets.items():
                    # Content-addressed and append-only: a set already
                    # recorded (by any connection) is referenced, not rewritten.
                    if conn.execute(
                        "SELECT 1 FROM context_sets WHERE set_hash = ?", (set_hash,)
                    ).fetchone():
                        continue
                    # Only chains this Ledger stored are hints, so the parts
                    # in memory are exactly the parts in the file.
                    chain = _grow(hints.get(key) or self._chains.get(key), set_hash, members)
                    hints[key] = (chain, members)
                    step, _, stored, upgrades = chain[-1]
                    conn.execute(
                        "INSERT INTO context_sets (set_hash, size, ts, base, step, upgrades)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (set_hash, len(members), ts, chain[-2][1] if step else None,
                         step, int(upgrades)),
                    )
                    conn.executemany(
                        "INSERT INTO context_members VALUES (?, ?, ?, ?)",
                        [(set_hash, s, r, t) for (s, r), t in stored.items()],
                    )
                    conn.executemany(
                        "INSERT INTO context_chain VALUES (?, ?)",
                        [(set_hash, part[1]) for part in chain],
                    )
                conn.executemany(
                    "INSERT INTO writes (write_id, content_hash, ts, store, agent_id,"
                    " session_id, source, context_set) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    write_rows,
                )
                if parent_rows:
                    conn.executemany("INSERT INTO direct_parents VALUES (?, ?, ?, ?)", parent_rows)
                if binding_rows:
                    conn.executemany(_INSERT_BINDING, binding_rows)
                if tombstone_rows:
                    conn.executemany(
                        "INSERT INTO tombstones (store, record_id, ts, reason, write_id,"
                        " agent_id, session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        tombstone_rows,
                    )
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            for key, hint in hints.items():
                self._chains[key] = hint
                self._chains.move_to_end(key)
                if len(self._chains) > _CHAIN_HINTS:
                    self._chains.popitem(last=False)
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

    def bind(
        self, write_id: str, store: str, record_id: str, content_hash: Optional[str] = None
    ) -> None:
        """Append a binding from a write to the record ID its store assigned.

        ``content_hash``, if the integration knows it, is the hash of what the
        store holds for the record right after the write, where that differs
        from (or can't be derived from) the write's own content hash.
        """
        with self._lock:
            self._conn.execute(_INSERT_BINDING, (write_id, store, record_id, time.time_ns(),
                                                 content_hash))

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
        with self._ckpt_lock:
            self._ckpt_conn.close()
        with self._walk_pool_lock:
            for conn in self._walk_pool:
                conn.close()
            self._walk_pool.clear()
        with self._read_lock:
            self._read_conn.close()
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
