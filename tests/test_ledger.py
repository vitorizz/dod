import hashlib
import sqlite3
import threading

import pytest

from dod.ledger import Ledger


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


def test_bind_appends(path, ledger):
    wid = write(ledger)
    ledger.bind(wid, "mem", "rec-1")
    ledger.bind(wid, "mem", "rec-2")
    rows = raw(path).execute(
        "SELECT store, record_id FROM bindings WHERE write_id=? ORDER BY record_id", (wid,)
    ).fetchall()
    assert rows == [("mem", "rec-1"), ("mem", "rec-2")]


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
