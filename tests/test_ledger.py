import hashlib
import os
import shutil
import sqlite3
import threading
import time

import pytest

import dod.ledger as ledger_mod
from dod.ledger import SCHEMA_VERSION, Ledger, LedgerSchemaError

TABLES = (
    "writes", "direct_parents", "context_sets", "context_members", "context_chain",
    "bindings", "tombstones",
)
CONTEXT = [("mem", "c1", "retrieved_in_scope"), ("pg", "c2", "in_prompt")]


@pytest.fixture
def path(tmp_path):
    return tmp_path / "ledger.db"


@pytest.fixture
def ledger(path):
    with Ledger(path) as led:
        yield led


def raw(path):
    # Autocommit, so a rejected statement doesn't leave a transaction open.
    return sqlite3.connect(str(path), isolation_level=None)


def set_of(path, write_id):
    return raw(path).execute(
        "SELECT context_set FROM writes WHERE write_id=?", (write_id,)
    ).fetchone()[0]


def edges(path, write_id):
    return raw(path).execute(
        "SELECT parent_store, parent_record_id, tag FROM parents WHERE write_id=? ORDER BY 1, 2",
        (write_id,),
    ).fetchall()


def count(path, table):
    return raw(path).execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def write(ledger, parents=(), content="x", **kw):
    kw.setdefault("store", "mem")
    kw.setdefault("source", "user")
    return ledger.append_write(content=content, parents=parents, **kw)


def test_creates_schema_with_wal(path, ledger):
    conn = raw(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(TABLES) <= tables
    views = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")}
    assert views == {"parents"}
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_append_write_records_row_and_parents(path, ledger):
    wid = write(
        ledger,
        parents=[("mem", "r1", "in_prompt"), ("pg", "r2", "retrieved_in_scope")],
        content="hello",
        agent_id="a1",
        session_id="s1",
        source="tool",
    )
    conn = raw(path)
    row = conn.execute(
        "SELECT content_hash, store, agent_id, session_id, source FROM writes WHERE write_id=?",
        (wid,),
    ).fetchone()
    assert row == (hashlib.sha256(b"hello").hexdigest(), "mem", "a1", "s1", "tool")
    parents = conn.execute(
        "SELECT parent_store, parent_record_id, tag FROM parents WHERE write_id=? ORDER BY 1",
        (wid,),
    ).fetchall()
    assert parents == [("mem", "r1", "in_prompt"), ("pg", "r2", "retrieved_in_scope")]


def test_content_is_not_stored(path, ledger):
    write(ledger, content="secret-memory-content")
    assert b"secret-memory-content" not in path.read_bytes()
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        assert b"secret-memory-content" not in wal.read_bytes()


def test_bytes_content_hashed_like_str(path, ledger):
    raw_bytes = b"\xffnot-utf8-secret"
    wid = write(ledger, content=raw_bytes)
    wid_str = write(ledger, content="hello")
    wid_bytes = write(ledger, content="hello".encode("utf-8"))
    conn = raw(path)
    hashes = dict(conn.execute("SELECT write_id, content_hash FROM writes"))
    assert hashes[wid] == hashlib.sha256(raw_bytes).hexdigest()
    assert hashes[wid_str] == hashes[wid_bytes]
    assert raw_bytes not in path.read_bytes()
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        assert raw_bytes not in wal.read_bytes()


def test_bind_appends(path, ledger):
    wid = write(ledger)
    ledger.bind(wid, "mem", "rec-1")
    ledger.bind(wid, "mem", "rec-2")
    rows = raw(path).execute(
        "SELECT store, record_id FROM bindings WHERE write_id=? ORDER BY record_id", (wid,)
    ).fetchall()
    assert rows == [("mem", "rec-1"), ("mem", "rec-2")]


def _batch_row(**kw):
    row = dict(content="x", store="mem", source="user")
    row.update(kw)
    return row


def test_append_batch_records_writes_and_bindings(path, ledger):
    earlier = write(ledger)
    w1, w2 = ledger.append_batch(
        [
            _batch_row(content="a", parents=[("mem", "p", "in_prompt")], record_id="rec-a"),
            _batch_row(content="b", agent_id="a1", session_id="s1", source="tool"),
        ],
        [(earlier, "mem", "rec-0")],
    )
    conn = raw(path)
    rows = conn.execute(
        "SELECT write_id, content_hash, agent_id, session_id, source FROM writes ORDER BY seq"
    ).fetchall()
    assert rows[1:] == [
        (w1, hashlib.sha256(b"a").hexdigest(), None, None, "user"),
        (w2, hashlib.sha256(b"b").hexdigest(), "a1", "s1", "tool"),
    ]
    assert conn.execute("SELECT * FROM parents WHERE write_id=?", (w1,)).fetchall() == [
        (w1, "mem", "p", "in_prompt")
    ]
    assert conn.execute("SELECT write_id, store, record_id FROM bindings ORDER BY record_id").fetchall() == [
        (earlier, "mem", "rec-0"), (w1, "mem", "rec-a"),
    ]


def test_append_batch_is_one_transaction(ledger):
    commits = []
    ledger._conn.set_trace_callback(lambda sql: sql == "COMMIT" and commits.append(sql))
    ledger.append_batch([_batch_row(record_id=f"r{i}") for i in range(5)])
    ledger._conn.set_trace_callback(None)
    assert commits == ["COMMIT"]


@pytest.mark.parametrize("bad", [
    {"writes": [_batch_row(), _batch_row(source="nope")]},
    {"writes": [_batch_row(), _batch_row(parents=[("mem", "p", "bad-tag")])]},
    {"writes": [_batch_row()], "bindings": [("no-such-write", "mem", "r")]},
])
def test_append_batch_is_all_or_nothing(path, ledger, bad):
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        ledger.append_batch(**bad)
    conn = raw(path)
    for table in ("writes", "parents", "bindings"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


def test_bind_unknown_write_raises(ledger):
    with pytest.raises(sqlite3.IntegrityError):
        ledger.bind("nope", "mem", "rec-1")


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE writes SET store='other'",
        "DELETE FROM writes",
        "UPDATE direct_parents SET tag='in_prompt'",
        "DELETE FROM direct_parents",
        "UPDATE context_sets SET size=0",
        "DELETE FROM context_sets",
        "UPDATE context_members SET tag='in_prompt'",
        "DELETE FROM context_members",
        "UPDATE context_chain SET part='other'",
        "DELETE FROM context_chain",
        "UPDATE bindings SET record_id='other'",
        "DELETE FROM bindings",
        "UPDATE tombstones SET reason='other'",
        "DELETE FROM tombstones",
    ],
)
def test_update_and_delete_are_rejected(path, ledger, sql):
    wid = write(ledger, parents=[("mem", "r1", "retrieved_in_scope")], context=CONTEXT)
    ledger.bind(wid, "mem", "rec-1")
    ledger.append_tombstones([dict(store="mem", record_id="rec-1", reason="forget")])
    conn = raw(path)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(sql)


@pytest.mark.parametrize("recursive_triggers", [False, True])
@pytest.mark.parametrize(
    "sql",
    [
        # Conflict on each unique key, and on the rowid of every table.
        "INSERT OR REPLACE INTO writes (write_id, content_hash, ts, store, source)"
        " VALUES (:wid, 'h', 0, 'other', 'user')",
        "INSERT OR REPLACE INTO writes (seq, write_id, content_hash, ts, store, source)"
        " VALUES (1, 'other', 'h', 0, 'other', 'user')",
        "REPLACE INTO direct_parents VALUES (:wid, 'mem', 'r1', 'in_prompt')",
        "INSERT OR REPLACE INTO direct_parents"
        " (rowid, write_id, parent_store, parent_record_id, tag)"
        " VALUES (1, :wid, 'other', 'other', 'in_prompt')",
        "INSERT OR REPLACE INTO context_sets (set_hash, size, ts)"
        " SELECT set_hash, 0, 0 FROM context_sets",
        "INSERT OR REPLACE INTO context_sets (seq, set_hash, size, ts)"
        " VALUES (1, 'other', 0, 0)",
        "REPLACE INTO context_members"
        " SELECT set_hash, store, record_id, 'in_prompt' FROM context_members",
        "INSERT OR REPLACE INTO context_members (rowid, set_hash, store, record_id, tag)"
        " SELECT 1, set_hash, 'other', 'other', 'in_prompt' FROM context_members LIMIT 1",
        "REPLACE INTO context_chain SELECT set_hash, part FROM context_chain",
        "INSERT OR REPLACE INTO bindings (rowid, write_id, store, record_id, ts)"
        " VALUES (1, :wid, 'other', 'other', 0)",
        "INSERT INTO direct_parents VALUES (:wid, 'mem', 'r1', 'in_prompt')"
        " ON CONFLICT DO UPDATE SET tag = excluded.tag",
        "INSERT OR REPLACE INTO tombstones (seq, store, record_id, ts, reason)"
        " VALUES (1, 'other', 'other', 0, 'other')",
        "INSERT INTO tombstones (seq, store, record_id, ts, reason)"
        " VALUES (1, 'other', 'other', 0, 'other') ON CONFLICT DO UPDATE SET reason = 'other'",
    ],
)
def test_replace_cannot_alter_or_remove_rows(path, ledger, sql, recursive_triggers):
    wid = write(ledger, parents=[("mem", "r1", "retrieved_in_scope")], context=CONTEXT)
    ledger.bind(wid, "mem", "rec-1")
    ledger.append_tombstones([dict(store="mem", record_id="rec-1", reason="forget")])
    conn = raw(path)
    # A connection that never enabled recursive_triggers must be blocked too.
    conn.execute(f"PRAGMA recursive_triggers={'ON' if recursive_triggers else 'OFF'}")
    # context_chain is WITHOUT ROWID; its primary key is the whole row.
    queries = [f"SELECT {'' if t == 'context_chain' else 'rowid, '}* FROM {t}" for t in TABLES]
    snapshot = [conn.execute(q).fetchall() for q in queries]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(sql, {"wid": wid})
    assert [conn.execute(q).fetchall() for q in queries] == snapshot


def test_append_tombstones_records_rows(path, ledger, monkeypatch):
    wid = write(ledger)
    monkeypatch.setattr(time, "time_ns", lambda: 123)
    ledger.append_tombstones([
        dict(store="mem", record_id="r1", reason="consolidation", write_id=wid,
             agent_id="a1", session_id="s1"),
        dict(store="pg", record_id="r2", reason="forget"),
    ])
    ledger.append_tombstones([dict(store="mem", record_id="r1", reason="excision")])
    rows = raw(path).execute("SELECT * FROM tombstones ORDER BY seq").fetchall()
    assert rows == [
        (1, "mem", "r1", 123, "consolidation", wid, "a1", "s1"),
        (2, "pg", "r2", 123, "forget", None, None, None),
        (3, "mem", "r1", 123, "excision", None, None, None),
    ]


def test_tombstone_is_content_free(path, ledger):
    # A tombstone has no content field, so nothing it is given can be stored.
    cols = [r[1] for r in raw(path).execute("PRAGMA table_info(tombstones)")]
    assert cols == ["seq", "store", "record_id", "ts", "reason", "write_id",
                    "agent_id", "session_id"]
    ledger.append_tombstones([dict(store="mem", record_id="r1", reason="forget",
                                   content="secret-memory-content")])
    assert b"secret-memory-content" not in path.read_bytes()
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        assert b"secret-memory-content" not in wal.read_bytes()


def test_append_batch_records_tombstones_in_same_transaction(path, ledger):
    commits = []
    ledger._conn.set_trace_callback(lambda sql: sql == "COMMIT" and commits.append(sql))
    (wid,) = ledger.append_batch(
        [_batch_row(record_id="r2")], (),
        [dict(store="mem", record_id="r1", reason="consolidation")],
    )
    ledger._conn.set_trace_callback(None)
    assert commits == ["COMMIT"]
    assert raw(path).execute("SELECT record_id FROM tombstones").fetchall() == [("r1",)]


@pytest.mark.parametrize("bad", [
    dict(store="mem", record_id="r1", reason="rumor"),
    dict(store="mem", record_id="r1", reason="forget", write_id="no-such-write"),
    dict(store="mem", record_id="r1"),
])
def test_bad_tombstone_records_nothing(path, ledger, bad):
    with pytest.raises((ValueError, KeyError, sqlite3.IntegrityError)):
        ledger.append_batch(
            [_batch_row()], (), [dict(store="mem", record_id="ok", reason="forget"), bad]
        )
    conn = raw(path)
    for table in TABLES:
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


def test_db_rejects_hand_inserted_bad_reason(path, ledger):
    with pytest.raises(sqlite3.IntegrityError):
        raw(path).execute(
            "INSERT INTO tombstones (store, record_id, ts, reason) VALUES ('m', 'r', 0, 'rumor')"
        )


def test_ledger_connection_enables_recursive_triggers(ledger):
    assert ledger._conn.execute("PRAGMA recursive_triggers").fetchone() == (1,)


def test_unknown_source_raises(ledger):
    with pytest.raises(ValueError):
        write(ledger, source="rumor")


@pytest.mark.parametrize(
    "parent",
    [("mem", "r1", "maybe"), ("mem", "r1", None), ("mem", "r1")],
)
def test_bad_or_missing_tag_raises(path, ledger, parent):
    with pytest.raises(ValueError):
        write(ledger, parents=[parent])
    assert raw(path).execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 0


@pytest.mark.parametrize("table", ["direct_parents", "context_members"])
@pytest.mark.parametrize("tag", ["maybe", None])
def test_db_rejects_hand_inserted_bad_tag(path, ledger, tag, table):
    wid = write(ledger, context=CONTEXT)
    key = wid if table == "direct_parents" else set_of(path, wid)
    with pytest.raises(sqlite3.IntegrityError):
        raw(path).execute(f"INSERT INTO {table} VALUES (?, 'mem', 'new', ?)", (key, tag))


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO parents VALUES ('w', 'mem', 'r1', 'in_prompt')",
        "UPDATE parents SET tag='in_prompt'",
        "DELETE FROM parents",
    ],
)
def test_parents_view_is_read_only(path, ledger, sql):
    write(ledger, parents=[("mem", "r1", "retrieved_in_scope")], context=CONTEXT)
    with pytest.raises(sqlite3.OperationalError, match="view"):
        raw(path).execute(sql)


def test_context_set_stored_once_for_many_writes(path, ledger):
    context = [("mem", f"c{i}", "retrieved_in_scope") for i in range(10)]
    wids = [write(ledger, context=list(context)) for _ in range(3)]
    wids += ledger.append_batch([_batch_row(context=context) for _ in range(5)])
    assert count(path, "context_sets") == 1
    assert count(path, "context_members") == 10
    assert count(path, "direct_parents") == 0
    assert len({set_of(path, w) for w in wids}) == 1
    for w in wids:
        assert edges(path, w) == sorted(context)
    assert count(path, "parents") == 80


def test_context_set_is_content_addressed(path, ledger):
    # Order and duplicates don't matter; the same members hash the same.
    a = write(ledger, context=[("mem", "a", "retrieved_in_scope"), ("pg", "b", "in_prompt")])
    b = write(ledger, context=[
        ("pg", "b", "retrieved_in_scope"), ("pg", "b", "in_prompt"),
        ("mem", "a", "retrieved_in_scope"),
    ])
    assert set_of(path, a) == set_of(path, b)
    # The address depends only on the members, so another ledger agrees.
    with Ledger(path.with_name("other.db")) as other:
        c = write(other, context=[("pg", "b", "in_prompt"), ("mem", "a", "retrieved_in_scope")])
    other_hash = raw(path.with_name("other.db")).execute(
        "SELECT context_set FROM writes WHERE write_id=?", (c,)
    ).fetchone()[0]
    assert other_hash == set_of(path, a)


def test_grown_or_retagged_context_is_a_new_set(path, ledger):
    base = [("mem", "a", "retrieved_in_scope")]
    w1 = write(ledger, context=base)
    w2 = write(ledger, context=base + [("mem", "b", "retrieved_in_scope")])
    w3 = write(ledger, context=[("mem", "a", "in_prompt"), ("mem", "b", "retrieved_in_scope")])
    w4 = write(ledger, context=base)
    assert len({set_of(path, w) for w in (w1, w2, w3)}) == 3
    assert set_of(path, w4) == set_of(path, w1)
    rows = raw(path).execute(
        "SELECT seq, size, base IS NOT NULL, step, upgrades FROM context_sets ORDER BY seq"
    ).fetchall()
    # w2 stores only b over w1's set; w3 (step 2, based on step 0) stores b
    # and the upgraded a.
    assert rows == [(1, 1, 0, 0, 0), (2, 2, 1, 1, 0), (3, 2, 1, 2, 1)]
    assert count(path, "context_members") == 4
    assert edges(path, w3) == [("mem", "a", "in_prompt"), ("mem", "b", "retrieved_in_scope")]
    assert edges(path, w2) == [("mem", "a", "retrieved_in_scope"), ("mem", "b", "retrieved_in_scope")]


def members_of(path, write_id):
    return raw(path).execute(
        "SELECT store, record_id, tag FROM context_members WHERE set_hash=?",
        (set_of(path, write_id),),
    ).fetchall()


def test_grown_context_stores_only_new_members(path, ledger):
    context, wids = [], []
    for i in range(20):
        context.append(("mem", f"c{i}", "retrieved_in_scope"))
        wids.append(write(ledger, context=list(context), agent_id="a", session_id="s"))
    # Step n stores the members added in the last (n & -n) steps.
    for n, w in enumerate(wids):
        assert len(members_of(path, w)) == (n & -n if n else 1)
        assert edges(path, w) == sorted(context[: n + 1])
    assert count(path, "context_members") < 20 * 21 // 2
    chain = raw(path).execute(
        "SELECT COUNT(*) FROM context_chain WHERE set_hash=?", (set_of(path, wids[-1]),)
    ).fetchone()[0]
    assert chain == bin(19).count("1") + 1  # step 19 = 0b10011: 3 deltas + the full set


def test_delta_set_matches_the_same_context_stored_full(path, ledger):
    # Built by growth here, in one go in another ledger: same address, same edges.
    context = []
    for i in range(10):
        context.append(("mem", f"c{i}", "in_prompt" if i % 3 else "retrieved_in_scope"))
        wid = write(ledger, context=list(context), agent_id="a")
    assert len(members_of(path, wid)) < len(context)
    with Ledger(path.with_name("full.db")) as full:
        other = write(full, context=list(context))
    assert set_of(path.with_name("full.db"), other) == set_of(path, wid)
    assert edges(path.with_name("full.db"), other) == edges(path, wid) == sorted(context)
    # Referencing it again, from another session, stores nothing new.
    before = count(path, "context_members")
    write(ledger, context=list(reversed(context)), agent_id="b")
    assert count(path, "context_members") == before


def test_upgraded_member_is_one_in_prompt_edge(path, ledger):
    ctx = {f"c{i}": "retrieved_in_scope" for i in range(8)}
    wids = []
    for step, upgrade in enumerate(["c0", None, "c3", None, "c5", "c1", None]):
        if upgrade:
            ctx[upgrade] = "in_prompt"
        ctx[f"n{step}"] = "retrieved_in_scope"
        wids.append((write(ledger, context=[("mem", r, t) for r, t in ctx.items()],
                           agent_id="a"), dict(ctx)))
    for wid, expected in wids:
        assert edges(path, wid) == sorted(("mem", r, t) for r, t in expected.items())


def test_shrunk_or_downgraded_context_is_stored_full(path, ledger):
    a = [("mem", "a", "in_prompt"), ("mem", "b", "retrieved_in_scope")]
    write(ledger, context=a, agent_id="x")
    w2 = write(ledger, context=[("mem", "a", "retrieved_in_scope"), ("mem", "c", "in_prompt"),
                                ("mem", "b", "retrieved_in_scope")], agent_id="x")
    w3 = write(ledger, context=[("mem", "c", "in_prompt")], agent_id="x")
    for w in (w2, w3):
        assert raw(path).execute(
            "SELECT base, step FROM context_sets WHERE set_hash=?", (set_of(path, w),)
        ).fetchone() == (None, 0)
    assert edges(path, w2) == [("mem", "a", "retrieved_in_scope"), ("mem", "b", "retrieved_in_scope"),
                               ("mem", "c", "in_prompt")]
    assert edges(path, w3) == [("mem", "c", "in_prompt")]


def test_chain_restarts_full_every_chain_steps(path, ledger, monkeypatch):
    monkeypatch.setattr(ledger_mod, "_CHAIN_STEPS", 8)
    context = []
    for i in range(20):
        context.append(("mem", f"c{i}", "retrieved_in_scope"))
        wid = write(ledger, context=list(context), agent_id="a")
        assert edges(path, wid) == sorted(context)
    steps = [r[0] for r in raw(path).execute("SELECT step FROM context_sets ORDER BY seq")]
    assert steps == [i % 8 for i in range(20)]
    assert max(r[0] for r in raw(path).execute(
        "SELECT COUNT(*) FROM context_chain GROUP BY set_hash")) == 4  # log2(8) + 1


def test_context_list_mutated_in_place_between_writes(path, ledger):
    # The ledger reads only a grown context's new items; a caller reusing
    # (and editing) one list must not make it skip or keep stale items.
    ctx = [("mem", "a", "retrieved_in_scope")]
    w1 = write(ledger, context=ctx, agent_id="a")
    ctx.append(("mem", "b", "retrieved_in_scope"))
    w2 = write(ledger, context=ctx, agent_id="a")
    ctx[0] = ("mem", "a", "in_prompt")
    w3 = write(ledger, context=ctx, agent_id="a")
    del ctx[1]
    w4 = write(ledger, context=ctx, agent_id="a")
    assert edges(path, w1) == [("mem", "a", "retrieved_in_scope")]
    assert edges(path, w2) == [("mem", "a", "retrieved_in_scope"), ("mem", "b", "retrieved_in_scope")]
    assert edges(path, w3) == [("mem", "a", "in_prompt"), ("mem", "b", "retrieved_in_scope")]
    assert edges(path, w4) == [("mem", "a", "in_prompt")]


def test_non_str_ids_are_never_reused_as_equal_ids(path, ledger):
    # 1 == 1.0, but they are stored as "1" and "1.0".
    w1 = write(ledger, context=[("mem", 1, "retrieved_in_scope")], agent_id="a")
    w2 = write(ledger, context=[("mem", 1.0, "retrieved_in_scope"),
                                ("mem", "b", "retrieved_in_scope")], agent_id="a")
    assert edges(path, w1) == [("mem", "1", "retrieved_in_scope")]
    assert edges(path, w2) == [("mem", "1.0", "retrieved_in_scope"), ("mem", "b", "retrieved_in_scope")]


def test_id_with_nul_still_content_addressed(path, ledger):
    # ("a\0b", "c") and ("a", "b\0c") must not hash alike.
    w1 = write(ledger, context=[("a\0b", "c", "in_prompt")])
    w2 = write(ledger, context=[("a", "b\0c", "in_prompt")])
    assert set_of(path, w1) != set_of(path, w2)
    assert edges(path, w2) == [("a", "b\0c", "in_prompt")]


def test_rolled_back_batch_leaves_no_chain_hint(path, ledger):
    write(ledger, context=[("mem", "a", "retrieved_in_scope")], agent_id="a")
    grown = [("mem", "a", "retrieved_in_scope"), ("mem", "b", "retrieved_in_scope")]
    with pytest.raises(sqlite3.IntegrityError):
        ledger.append_batch([_batch_row(context=grown, agent_id="a")], [("no-such-write", "mem", "r")])
    # The next growth is based on the committed set, not the rolled-back one.
    w = write(ledger, context=grown + [("mem", "c", "retrieved_in_scope")], agent_id="a")
    assert edges(path, w) == sorted(grown + [("mem", "c", "retrieved_in_scope")])


def test_empty_context_has_no_set(path, ledger):
    wid = write(ledger, parents=[("mem", "r", "in_prompt")], context=[])
    assert set_of(path, wid) is None
    assert count(path, "context_sets") == 0
    assert edges(path, wid) == [("mem", "r", "in_prompt")]


@pytest.mark.parametrize(
    "direct_tag,context_tag",
    [
        ("in_prompt", "retrieved_in_scope"),
        ("retrieved_in_scope", "in_prompt"),
        ("retrieved_in_scope", "retrieved_in_scope"),
        ("in_prompt", "in_prompt"),
    ],
)
def test_item_in_direct_parents_and_context_is_one_edge(path, ledger, direct_tag, context_tag):
    wid = write(
        ledger,
        parents=[("mem", "x", direct_tag)],
        context=[("mem", "x", context_tag), ("mem", "y", "retrieved_in_scope")],
    )
    merged = "in_prompt" if "in_prompt" in (direct_tag, context_tag) else "retrieved_in_scope"
    assert edges(path, wid) == [("mem", "x", merged), ("mem", "y", "retrieved_in_scope")]
    # Another write sharing the set still sees the context's own tag.
    other = write(ledger, context=[("mem", "x", context_tag), ("mem", "y", "retrieved_in_scope")])
    assert edges(path, other) == [("mem", "x", context_tag), ("mem", "y", "retrieved_in_scope")]


def test_context_set_is_content_free(path, ledger):
    cols = {t: [r[1] for r in raw(path).execute(f"PRAGMA table_info({t})")]
            for t in ("context_sets", "context_members", "context_chain")}
    assert cols == {
        "context_sets": ["seq", "set_hash", "size", "ts", "base", "step", "upgrades"],
        "context_members": ["set_hash", "store", "record_id", "tag"],
        "context_chain": ["set_hash", "part"],
    }
    write(ledger, content="secret-memory-content", context=CONTEXT)
    assert b"secret-memory-content" not in path.read_bytes()
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        assert b"secret-memory-content" not in wal.read_bytes()


def test_bad_context_records_nothing(path, ledger):
    with pytest.raises(ValueError):
        ledger.append_batch([
            _batch_row(context=CONTEXT),
            _batch_row(context=[("mem", "c", "bad-tag")]),
        ])
    for table in TABLES:
        assert count(path, table) == 0


def test_failed_batch_leaves_no_context_set(path, ledger):
    with pytest.raises(sqlite3.IntegrityError):
        ledger.append_batch([_batch_row(context=CONTEXT)], [("no-such-write", "mem", "r")])
    for table in TABLES:
        assert count(path, table) == 0
    # The set is recorded normally afterwards.
    wid = write(ledger, context=CONTEXT)
    assert edges(path, wid) == sorted(CONTEXT)


def test_empty_parents(path, ledger):
    wid = write(ledger, parents=[])
    conn = raw(path)
    assert conn.execute("SELECT COUNT(*) FROM writes WHERE write_id=?", (wid,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0] == 0


def test_duplicate_parents_stored_once_in_prompt_wins(path, ledger):
    wid = write(
        ledger,
        parents=[
            ("mem", "r1", "retrieved_in_scope"),
            ("mem", "r1", "in_prompt"),
            ("mem", "r1", "retrieved_in_scope"),
            ("mem", "r2", "retrieved_in_scope"),
            ("mem", "r2", "retrieved_in_scope"),
        ],
    )
    rows = raw(path).execute(
        "SELECT parent_record_id, tag FROM parents WHERE write_id=? ORDER BY 1", (wid,)
    ).fetchall()
    assert rows == [("r1", "in_prompt"), ("r2", "retrieved_in_scope")]


def test_same_content_hash_kept_as_distinct_writes(path, ledger):
    a = write(ledger, content="same")
    b = write(ledger, content="same")
    assert a != b
    hashes = raw(path).execute("SELECT content_hash FROM writes").fetchall()
    assert len(hashes) == 2 and hashes[0] == hashes[1]


def test_arbitrary_new_store_accepted(path, ledger):
    # No store registry: any store name is valid, for writes and parents alike.
    wid = write(ledger, store="crm-tickets-v2", parents=[("brand-new-store", "r1", "in_prompt")])
    conn = raw(path)
    assert conn.execute("SELECT store FROM writes WHERE write_id=?", (wid,)).fetchone() == (
        "crm-tickets-v2",
    )
    assert conn.execute("SELECT parent_store FROM parents WHERE write_id=?", (wid,)).fetchone() == (
        "brand-new-store",
    )


def test_seq_orders_writes_even_if_clock_steps_back(path, ledger, monkeypatch):
    clock = iter([3_000, 2_000, 1_000])
    monkeypatch.setattr("dod.ledger.time.time_ns", lambda: next(clock))
    wids = [write(ledger) for _ in range(3)]
    rows = raw(path).execute("SELECT write_id, seq, ts FROM writes ORDER BY seq").fetchall()
    assert [r[0] for r in rows] == wids
    assert [r[1] for r in rows] == [1, 2, 3]
    assert [r[2] for r in rows] == [3_000, 2_000, 1_000]


def test_write_ids_sort_by_creation(path, ledger):
    # Singles and a batch: many share a millisecond, and a batch shares a ts.
    wids = [write(ledger) for _ in range(200)]
    wids += ledger.append_batch([dict(content=str(i), store="s", source="user") for i in range(50)])
    wids += [write(ledger) for _ in range(200)]
    assert wids == sorted(wids) and len(set(wids)) == len(wids)
    assert all(len(w) == 32 and w[12] == "7" for w in wids)  # UUIDv7 layout
    ms = int(wids[0][:12], 16)
    assert abs(ms - time.time_ns() // 1_000_000) < 60_000
    rows = raw(path).execute("SELECT write_id FROM writes ORDER BY seq").fetchall()
    assert [r[0] for r in rows] == wids


def test_write_ids_keep_increasing_if_clock_steps_back(ledger, monkeypatch):
    clock = iter([3_000_000_000, 2_000_000_000, 1_000_000_000])
    monkeypatch.setattr("dod.ledger.time.time_ns", lambda: next(clock))
    wids = [write(ledger) for _ in range(3)]
    assert wids == sorted(wids) and len(set(wids)) == 3


def test_seq_continues_after_reopen(path):
    with Ledger(path) as led:
        write(led)
        write(led)
    with Ledger(path) as led:
        wid = write(led)
    assert raw(path).execute("SELECT seq FROM writes WHERE write_id=?", (wid,)).fetchone() == (3,)


def test_reopen_keeps_data(path):
    with Ledger(path) as led:
        wid = write(led, parents=[("mem", "r1", "in_prompt")])
        led.bind(wid, "mem", "rec-1")
    with Ledger(path) as led:
        write(led)
    conn = raw(path)
    assert conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM bindings").fetchone()[0] == 1


def user_version(path):
    return raw(path).execute("PRAGMA user_version").fetchone()[0]


def test_new_file_gets_schema_version(path, ledger):
    assert user_version(path) == SCHEMA_VERSION


def test_empty_file_gets_schema_version(path):
    path.touch()
    with Ledger(path):
        pass
    assert user_version(path) == SCHEMA_VERSION


def test_reopen_at_same_version(path):
    with Ledger(path) as led:
        write(led)
    with Ledger(path) as led:
        write(led)
    assert user_version(path) == SCHEMA_VERSION
    assert count(path, "writes") == 2


# The schema before context sets (#46): one parents row per edge, no version.
PRE_46_SCHEMA = """
CREATE TABLE writes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, write_id TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL, ts INTEGER NOT NULL, store TEXT NOT NULL,
    agent_id TEXT, session_id TEXT, source TEXT NOT NULL
);
CREATE TABLE parents (
    write_id TEXT NOT NULL REFERENCES writes (write_id),
    parent_store TEXT NOT NULL, parent_record_id TEXT NOT NULL, tag TEXT NOT NULL,
    PRIMARY KEY (write_id, parent_store, parent_record_id)
);
CREATE TABLE bindings (
    write_id TEXT NOT NULL, store TEXT NOT NULL, record_id TEXT NOT NULL, ts INTEGER NOT NULL
);
INSERT INTO writes (write_id, content_hash, ts, store, source)
    VALUES ('w1', 'h', 0, 'mem', 'user');
INSERT INTO parents VALUES ('w1', 'mem', 'r0', 'in_prompt');
"""


def test_pre_46_ledger_raises_before_touching_file(path):
    conn = raw(path)
    conn.executescript(PRE_46_SCHEMA)
    conn.close()
    before = path.read_bytes()
    with pytest.raises(LedgerSchemaError) as e:
        Ledger(path)
    msg = str(e.value)
    assert "unversioned" in msg and f"version {SCHEMA_VERSION}" in msg and str(path) in msg
    assert path.read_bytes() == before
    tables = {r[0] for r in raw(path).execute("SELECT name FROM sqlite_master")}
    assert "context_sets" not in tables
    assert user_version(path) == 0


def test_future_version_raises(path):
    with Ledger(path):
        pass
    raw(path).execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    with pytest.raises(LedgerSchemaError) as e:
        Ledger(path)
    msg = str(e.value)
    assert f"schema version {SCHEMA_VERSION + 1}" in msg
    assert f"uses version {SCHEMA_VERSION}" in msg
    assert user_version(path) == SCHEMA_VERSION + 1


def _as_version_5(path):
    """Turn a current ledger into a version-5 file: no audit chain."""
    conn = raw(path)
    conn.execute("DROP TABLE audit_chain")
    conn.execute("PRAGMA user_version=5")
    conn.close()


def _as_version_4(path):
    """Turn a current ledger into a version-4 file: also no bindings.content_hash."""
    _as_version_5(path)
    conn = raw(path)
    conn.execute("ALTER TABLE bindings DROP COLUMN content_hash")
    conn.execute("PRAGMA user_version=4")
    conn.close()


def test_version_4_migrates_in_place(path):
    with Ledger(path) as led:
        first = write(led)
        led.bind(first, "mem", "r1")
    _as_version_4(path)
    with Ledger(path) as led:
        second = write(led)
        led.bind(second, "mem", "r2", content_hash="h2")
    assert user_version(path) == SCHEMA_VERSION == 6
    assert raw(path).execute(
        "SELECT write_id, record_id, content_hash FROM bindings ORDER BY rowid").fetchall() == [
        (first, "r1", None), (second, "r2", "h2")]


def test_version_3_migrates_in_place(path):
    # Version 3 differs by the derived child_edges index, which version 4
    # drops (the lineage file replaces it), and by bindings.content_hash.
    with Ledger(path) as led:
        first = write(led)
    _as_version_4(path)
    conn = raw(path)
    conn.executescript("""
    CREATE TABLE child_edges (parent_store TEXT, parent_record_id TEXT, seq INTEGER,
        write_id TEXT, tag TEXT, PRIMARY KEY (parent_store, parent_record_id, seq));
    CREATE TABLE child_edges_filled (upto INTEGER PRIMARY KEY, ts INTEGER NOT NULL);
    CREATE TRIGGER child_edges_no_delete BEFORE DELETE ON child_edges
    BEGIN SELECT RAISE(ABORT, 'child_edges is append-only'); END;
    PRAGMA user_version=3;
    """)
    conn.close()
    with Ledger(path) as led:
        write(led)
    assert user_version(path) == SCHEMA_VERSION
    names = {r[0] for r in raw(path).execute("SELECT name FROM sqlite_master")}
    assert not any(n.startswith("child_edges") for n in names)
    cols = [r[1] for r in raw(path).execute("PRAGMA table_info(bindings)")]
    assert cols[-1] == "content_hash"
    assert count(path, "writes") == 2
    assert raw(path).execute("SELECT 1 FROM writes WHERE write_id = ?", (first,)).fetchone()


def test_version_2_still_raises(path):
    with Ledger(path):
        pass
    raw(path).execute("PRAGMA user_version=2")
    with pytest.raises(LedgerSchemaError, match="schema version 2"):
        Ledger(path)


def test_close_checkpoints_wal(path):
    led = Ledger(path)
    for _ in range(100):
        write(led)
    led.close()
    led.close()  # idempotent
    wal = path.with_name(path.name + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    assert raw(path).execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 100


def rows_in_main_file(path, tmp_path):
    # Copy only the main database file, leaving the WAL behind, so the count
    # reflects what a checkpoint has already moved out of the WAL.
    copy = tmp_path / "main-only.db"
    shutil.copyfile(path, copy)
    try:
        return raw(copy).execute("SELECT COUNT(*) FROM writes").fetchone()[0]
    except sqlite3.Error:  # schema not checkpointed yet, or copied mid-checkpoint
        return 0
    finally:
        copy.unlink()


def test_background_checkpoint_runs_before_close(path, tmp_path):
    with Ledger(path, checkpoint_interval=0.05) as led:
        for _ in range(100):
            write(led)
        deadline = time.monotonic() + 5
        while rows_in_main_file(path, tmp_path) < 100 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert rows_in_main_file(path, tmp_path) == 100


def test_no_checkpoint_on_append_path(path, tmp_path):
    # Control for the test above: without the background thread, appends
    # alone never move rows into the main file.
    with Ledger(path, checkpoint_interval=3600, background_index=False) as led:
        for _ in range(100):
            write(led)
        time.sleep(0.2)
        assert rows_in_main_file(path, tmp_path) == 0


def test_concurrent_appends(path, ledger):
    errors = []

    def worker(t):
        try:
            for i in range(500):
                write(ledger, parents=[("mem", f"{t}-{i}", "in_prompt")], agent_id=str(t))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    conn = raw(path)
    assert conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 4000
    assert conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0] == 4000
    seqs = [r[0] for r in conn.execute("SELECT seq FROM writes ORDER BY seq")]
    assert seqs == list(range(1, 4001))
    # IDs are issued under the append lock, so they increase with seq.
    ids = [r[0] for r in conn.execute("SELECT write_id FROM writes ORDER BY seq")]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_write_ids_increase_across_ledgers_in_a_process(tmp_path):
    with Ledger(tmp_path / "a.db") as a, Ledger(tmp_path / "b.db") as b:
        wids = [write(led) for _ in range(100) for led in (a, b)]
    assert wids == sorted(wids) and len(set(wids)) == len(wids)


def test_two_ledgers_racing_on_the_same_new_context(path):
    # Two connections on one file record the same new context set at once.
    # Neither write may fail, and the set is stored once.
    rounds = 200
    barrier = threading.Barrier(2)
    errors, wids = [], [[], []]

    def context(i):
        return [("mem", f"r{i}-a", "retrieved_in_scope"), ("pg", f"r{i}-b", "in_prompt")]

    def worker(led, n):
        try:
            for i in range(rounds):
                barrier.wait()
                wids[n].append(write(led, context=context(i), agent_id=str(n)))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            barrier.abort()

    with Ledger(path) as a, Ledger(path) as b:
        threads = [threading.Thread(target=worker, args=(led, n)) for n, led in enumerate((a, b))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    assert errors == []
    assert count(path, "writes") == 2 * rounds
    assert count(path, "context_sets") == rounds
    assert count(path, "context_members") == 2 * rounds
    for i, (wa, wb) in enumerate(zip(*wids)):
        assert set_of(path, wa) == set_of(path, wb)
        assert edges(path, wa) == edges(path, wb) == sorted(context(i))


def test_wal_stays_bounded_under_continuous_writes(tmp_path, monkeypatch):
    # With autocheckpoint off, SQLite rarely restarts the WAL by itself while
    # writes keep coming; the writer restarts it at _WAL_RESTART_BYTES.
    import dod.ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "_WAL_RESTART_BYTES", 1 << 20)
    path = tmp_path / "ledger.db"
    with Ledger(path, checkpoint_interval=0.02) as led:
        peak = 0
        # ~17 MiB of WAL without restarts; with them, a few times the 1 MiB
        # limit at most (the checkpoint thread on a slow CI runner lags).
        for i in range(600):
            time.sleep(0.002)  # leave the checkpoint thread a pass now and then
            led.append_tombstones([{"store": "s", "record_id": f"r{i}-{j}", "reason": "forget"}
                                   for j in range(200)])
            peak = max(peak, os.path.getsize(str(path) + "-wal"))
        (n,) = led._conn.execute("SELECT count(*) FROM tombstones").fetchone()
    assert n == 600 * 200
    assert peak < 8 << 20, f"WAL grew to {peak} bytes"


def test_wal_restart_backs_off_while_the_index_fill_holds_the_lock(tmp_path, monkeypatch):
    import dod.ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "_WAL_RESTART_BYTES", 1 << 16)
    path = tmp_path / "ledger.db"
    with Ledger(path, checkpoint_interval=3600) as led:
        led._wal_log = 10_000  # as if the last pass found a large WAL
        with led._ckpt_lock:  # as if the fill were reading
            led.append_write(content="x", store="s", source="user")
            assert led._wal_retry_at > 0 and led._wal_log == 10_000  # backed off
        led.append_write(content="y", store="s", source="user")
        assert led._wal_log == 10_000  # still backing off
        led._wal_retry_at = 0.0
        led.append_write(content="z", store="s", source="user")
        assert led._wal_log == 0  # restarted once the lock was free
