import hashlib
import shutil
import sqlite3
import threading
import time

import pytest

from dod.ledger import Ledger

TABLES = ("writes", "parents", "bindings")


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


def write(ledger, parents=(), content="x", **kw):
    kw.setdefault("store", "mem")
    kw.setdefault("source", "user")
    return ledger.append_write(content=content, parents=parents, **kw)


def test_creates_schema_with_wal(path, ledger):
    conn = raw(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"writes", "parents", "bindings"} <= tables
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
        "UPDATE parents SET tag='in_prompt'",
        "DELETE FROM parents",
        "UPDATE bindings SET record_id='other'",
        "DELETE FROM bindings",
    ],
)
def test_update_and_delete_are_rejected(path, ledger, sql):
    wid = write(ledger, parents=[("mem", "r1", "retrieved_in_scope")])
    ledger.bind(wid, "mem", "rec-1")
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
        "REPLACE INTO parents VALUES (:wid, 'mem', 'r1', 'in_prompt')",
        "INSERT OR REPLACE INTO parents (rowid, write_id, parent_store, parent_record_id, tag)"
        " VALUES (1, :wid, 'other', 'other', 'in_prompt')",
        "INSERT OR REPLACE INTO bindings (rowid, write_id, store, record_id, ts)"
        " VALUES (1, :wid, 'other', 'other', 0)",
        "INSERT INTO parents VALUES (:wid, 'mem', 'r1', 'in_prompt')"
        " ON CONFLICT DO UPDATE SET tag = excluded.tag",
    ],
)
def test_replace_cannot_alter_or_remove_rows(path, ledger, sql, recursive_triggers):
    wid = write(ledger, parents=[("mem", "r1", "retrieved_in_scope")])
    ledger.bind(wid, "mem", "rec-1")
    conn = raw(path)
    # A connection that never enabled recursive_triggers must be blocked too.
    conn.execute(f"PRAGMA recursive_triggers={'ON' if recursive_triggers else 'OFF'}")
    snapshot = [conn.execute(f"SELECT rowid, * FROM {t}").fetchall() for t in TABLES]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(sql, {"wid": wid})
    assert [conn.execute(f"SELECT rowid, * FROM {t}").fetchall() for t in TABLES] == snapshot


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


@pytest.mark.parametrize("tag", ["maybe", None])
def test_db_rejects_hand_inserted_bad_tag(path, ledger, tag):
    wid = write(ledger)
    conn = raw(path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO parents VALUES (?, 'mem', 'r1', ?)", (wid, tag))


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
    with Ledger(path, checkpoint_interval=3600) as led:
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
