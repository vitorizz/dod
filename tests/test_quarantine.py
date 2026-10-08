import sqlite3

import pytest
from test_crewai import StubMemory

from dod.capture import Capture
from dod.integrations import crewai as integ
from dod.integrations.crewai import agent_task_scope, instrument
from dod.ledger import Ledger


@pytest.fixture
def path(tmp_path):
    return tmp_path / "ledger.db"


@pytest.fixture
def ledger(path):
    with Ledger(path) as led:
        yield led


@pytest.fixture(autouse=True)
def clear_blocked():
    integ._blocked.clear()
    yield
    integ._blocked.clear()


def parents(path, write_id):
    conn = sqlite3.connect(str(path))
    rows = conn.execute(
        "SELECT parent_store, parent_record_id FROM parents WHERE write_id = ?",
        (write_id,),
    ).fetchall()
    conn.close()
    return set(rows)


# --- Ledger ------------------------------------------------------------------


def test_quarantine_release_round_trip(path, ledger):
    assert ledger.quarantined() == set()
    ledger.quarantine([("mem", "a"), ("mem", "b")], reason="planted", actor="alice")
    assert ledger.quarantined() == {("mem", "a"), ("mem", "b")}
    ledger.release([("mem", "a")], actor="alice")
    assert ledger.quarantined() == {("mem", "b")}

    conn = sqlite3.connect(str(path))
    rows = conn.execute(
        "SELECT seq, store, record_id, action, reason, actor FROM quarantine ORDER BY seq"
    ).fetchall()
    assert rows == [
        (1, "mem", "a", "quarantine", "planted", "alice"),
        (2, "mem", "b", "quarantine", "planted", "alice"),
        (3, "mem", "a", "release", None, "alice"),
    ]


def test_latest_action_wins(ledger):
    item = ("mem", "a")
    ledger.quarantine([item])
    ledger.release([item])
    ledger.quarantine([item])
    assert ledger.quarantined() == {item}
    ledger.release([item])
    assert ledger.quarantined() == set()
    # Same record ID in another store is a different item.
    ledger.quarantine([("other", "a")])
    assert ledger.quarantined() == {("other", "a")}


def test_release_of_never_quarantined_item_is_harmless(ledger):
    ledger.release([("mem", "a")])
    assert ledger.quarantined() == set()


def test_quarantine_log_returns_actions_after_seq(ledger):
    assert ledger.quarantine_log() == []
    ledger.quarantine([("mem", "a"), ("mem", "b")])
    ledger.release([("mem", "a")])
    assert ledger.quarantine_log() == [
        (1, "mem", "a", "quarantine"),
        (2, "mem", "b", "quarantine"),
        (3, "mem", "a", "release"),
    ]
    assert ledger.quarantine_log(2) == [(3, "mem", "a", "release")]
    assert ledger.quarantine_log(3) == []


def test_quarantine_sees_other_connections(path, ledger):
    with Ledger(path) as other:
        other.quarantine([("mem", "a")])
    assert ledger.quarantined() == {("mem", "a")}


@pytest.mark.parametrize("bad", [("mem",), ("mem", "a", "b"), "mem:a", None])
def test_bad_item_raises_and_records_nothing(ledger, bad):
    with pytest.raises(ValueError):
        ledger.quarantine([("mem", "ok"), bad])
    assert ledger.quarantine_log() == []


def test_db_rejects_unknown_action(path, ledger):
    conn = sqlite3.connect(str(path))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO quarantine (ts, store, record_id, action) VALUES (0, 'm', 'a', 'purge')"
        )


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE quarantine SET action = 'release'",
        "DELETE FROM quarantine",
        "INSERT OR REPLACE INTO quarantine (seq, ts, store, record_id, action)"
        " VALUES (1, 0, 'mem', 'a', 'release')",
        "REPLACE INTO quarantine (seq, ts, store, record_id, action)"
        " VALUES (1, 0, 'mem', 'a', 'release')",
    ],
)
@pytest.mark.parametrize("recursive_triggers", [False, True])
def test_quarantine_table_is_append_only(path, ledger, sql, recursive_triggers):
    ledger.quarantine([("mem", "a")], reason="r")
    conn = sqlite3.connect(str(path))
    conn.execute(f"PRAGMA recursive_triggers={'ON' if recursive_triggers else 'OFF'}")
    before = conn.execute("SELECT * FROM quarantine").fetchall()
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(sql)
    assert conn.execute("SELECT * FROM quarantine").fetchall() == before
    assert ledger.quarantined() == {("mem", "a")}


def test_quarantine_is_content_free(path):
    with Ledger(path) as ledger:
        ledger.quarantine([("mem", "a")], reason="planted", actor="alice")
    conn = sqlite3.connect(str(path))
    cols = [r[1] for r in conn.execute("PRAGMA table_info(quarantine)")]
    assert cols == ["seq", "ts", "store", "record_id", "action", "reason", "actor"]


# --- CrewAI recall filtering -------------------------------------------------


def setup_memory(ledger):
    mem = StubMemory(pool=False)
    mem.results["q"] = mem.add("good", "bad")
    instrument(mem, Capture(ledger))
    return mem


def test_quarantined_record_is_not_recalled_or_recorded_as_parent(path, ledger):
    mem = setup_memory(ledger)
    ledger.quarantine([("crewai", "bad")], reason="planted")
    with agent_task_scope("agent", "task"):
        assert [m.record.id for m in mem.recall("q")] == ["good"]
        mem.remember("derived")
    (write_id,) = [r[0] for r in sqlite3.connect(str(path)).execute(
        "SELECT write_id FROM writes")]
    assert parents(path, write_id) == {("crewai", "good")}


def test_release_makes_record_visible_again(path, ledger):
    mem = setup_memory(ledger)
    ledger.quarantine([("crewai", "bad")])
    assert [m.record.id for m in mem.recall("q")] == ["good"]
    ledger.release([("crewai", "bad")])
    with agent_task_scope("agent", "task"):
        assert [m.record.id for m in mem.recall("q")] == ["good", "bad"]
        mem.remember("derived")
    (write_id,) = [r[0] for r in sqlite3.connect(str(path)).execute(
        "SELECT write_id FROM writes")]
    assert parents(path, write_id) == {("crewai", "good"), ("crewai", "bad")}


def test_quarantine_from_another_connection_is_picked_up(path, ledger):
    mem = setup_memory(ledger)
    assert len(mem.recall("q")) == 2  # caches the empty set
    with Ledger(path) as other:
        other.quarantine([("crewai", "bad")])
    assert [m.record.id for m in mem.recall("q")] == ["good"]


def test_quarantine_is_per_store(ledger):
    mem = setup_memory(ledger)
    ledger.quarantine([("other-store", "bad")])
    assert len(mem.recall("q")) == 2


def test_blocked_retrievals_are_counted_per_item(ledger):
    mem = setup_memory(ledger)
    mem.recall("q")
    assert integ.blocked_retrievals() == {}
    ledger.quarantine([("crewai", "bad"), ("crewai", "good")])
    mem.recall("q")
    mem.recall("q")
    ledger.release([("crewai", "good")])
    mem.recall("q")
    assert integ.blocked_retrievals() == {("crewai", "bad"): 3, ("crewai", "good"): 2}


def test_quarantine_cache_reads_only_new_actions(ledger, monkeypatch):
    mem = setup_memory(ledger)
    ledger.quarantine([("crewai", "bad")])
    rows = []
    orig = ledger.quarantine_log
    monkeypatch.setattr(ledger, "quarantine_log",
                        lambda after=0: rows.append(orig(after)) or rows[-1])
    for _ in range(5):
        mem.recall("q")
    assert [len(r) for r in rows] == [1, 0, 0, 0, 0]
    ledger.quarantine([("crewai", "good")])
    ledger.release([("crewai", "bad")])
    assert [m.record.id for m in mem.recall("q")] == ["bad"]
    assert len(rows[-1]) == 2
