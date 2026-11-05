import sqlite3

import pytest
from test_crewai import StubMemory

import dod
from dod import audit
from dod.capture import Capture
from dod.integrations import crewai
from dod.ledger import Ledger
from dod.scope import agent_task_scope


@pytest.fixture
def path(tmp_path):
    return tmp_path / "ledger.db"


@pytest.fixture
def ledger(path):
    with Ledger(path) as led:
        yield led


@pytest.fixture(autouse=True)
def clear_blocked():
    crewai._blocked.clear()
    yield
    crewai._blocked.clear()


def raw(path):
    return sqlite3.connect(str(path), isolation_level=None)


def tamper(path, sql, *args):
    """Run ``sql`` with the append-only triggers dropped, as an attacker could."""
    conn = raw(path)
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {name}")
    conn.execute(sql, args)
    conn.close()


@pytest.fixture
def chain(ledger):
    """bad -> d1 -> d2, plus good unrelated."""
    mem = crewai.instrument(StubMemory(pool=False), Capture(ledger))
    with agent_task_scope("agent", "plant"):
        bad = mem.remember("the user is allergic to peanuts")
        good = mem.remember("the user lives in Brooklyn")
    with agent_task_scope("agent", "t1"):
        mem.results["bad"] = [bad]
        mem.recall("bad")
        d1 = mem.remember("avoid Thai restaurants")
    with agent_task_scope("agent", "t2"):
        mem.results["d1"] = [d1]
        mem.recall("d1")
        d2 = mem.remember("suggested Italian for dinner")
    return mem, bad, good, d1, d2


def test_exported_from_package():
    assert dod.audit is audit


def test_empty_ledger_verifies(ledger):
    v = audit.verify(ledger)
    assert v.ok and v.entries == 0 and v.head is None


def test_every_append_is_chained(path, ledger, chain):
    _, bad, *_ = chain
    dod.excise(ledger, "crewai", bad.id)
    v = audit.verify(ledger)
    assert v.ok, v.problems
    assert v.entries > 0 and v.head == audit.head(ledger)
    # Each entry links to the one before it.
    rows = raw(path).execute("SELECT digest, prev FROM audit_chain ORDER BY seq").fetchall()
    assert rows[0][1] is None
    assert all(rows[i][1] == rows[i - 1][0] for i in range(1, len(rows)))


def test_chain_is_append_only(path, ledger, chain):
    conn = raw(path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE audit_chain SET digest = 'x' WHERE seq = 1")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM audit_chain")


def test_edited_row_is_detected(path, ledger, chain):
    tamper(path, "UPDATE writes SET content_hash = 'forged' WHERE seq = 1")
    v = audit.verify(ledger)
    assert not v.ok and any("digest mismatch" in p for p in v.problems)


def test_deleted_edge_is_detected(path, ledger, chain):
    tamper(path, "DELETE FROM bindings WHERE rowid = (SELECT max(rowid) FROM bindings)")
    v = audit.verify(ledger)
    assert not v.ok and any("digest mismatch" in p for p in v.problems)


def test_edited_context_member_is_detected(path, ledger, chain):
    tamper(path, "UPDATE context_members SET record_id = 'forged'"
                 " WHERE rowid = (SELECT min(rowid) FROM context_members)")
    v = audit.verify(ledger)
    assert not v.ok and any("address mismatch" in p for p in v.problems)


def test_added_context_member_is_detected(path, ledger, chain):
    tamper(path, "INSERT INTO context_members (set_hash, store, record_id, tag)"
                 " SELECT set_hash, 'crewai', 'forged', 'in_prompt' FROM context_sets LIMIT 1")
    v = audit.verify(ledger)
    assert not v.ok and any("address mismatch" in p for p in v.problems)


def test_direct_insert_is_unchained(path, ledger, chain):
    _, bad, *_ = chain
    tamper(path, "INSERT INTO quarantine (ts, store, record_id, action, reason, actor)"
                 " VALUES (1, 'crewai', ?, 'release', NULL, NULL)", bad.id)
    v = audit.verify(ledger)
    assert not v.ok and any("in no chain entry" in p for p in v.problems)


def test_edited_entry_breaks_the_links(path, ledger, chain):
    tamper(path, "UPDATE audit_chain SET digest = 'z' || substr(digest, 2) WHERE seq = 1")
    v = audit.verify(ledger)
    assert not v.ok and any("link broken" in p for p in v.problems)


def test_truncation_needs_an_anchor(path, ledger, chain):
    _, bad, *_ = chain
    anchor = audit.head(ledger)
    dod.excise(ledger, "crewai", bad.id)
    assert audit.verify(ledger, expected_head=anchor).ok
    # Cut the excision off: its rows and its entries. The file alone still verifies.
    (n,) = raw(path).execute("SELECT seq FROM audit_chain WHERE digest = ?", (anchor,)).fetchone()
    tamper(path, "DELETE FROM quarantine")
    tamper(path, "DELETE FROM audit_chain WHERE seq > ?", n)
    assert audit.verify(ledger).ok
    later = audit.head(ledger)
    assert audit.verify(ledger, expected_head=anchor).ok  # the anchor itself survives
    tamper(path, "DELETE FROM audit_chain WHERE seq = ?", n)
    tamper(path, "DELETE FROM writes WHERE seq = (SELECT max(seq) FROM writes)")
    v = audit.verify(ledger, expected_head=later)
    assert not v.ok and any("expected head" in p for p in v.problems)


def test_migration_seals_existing_rows(path, ledger, chain):
    ledger.close()
    conn = raw(path)
    conn.execute("DROP TABLE audit_chain")
    conn.execute("PRAGMA user_version=5")
    conn.close()
    with Ledger(path) as led:
        v = audit.verify(led)
        assert v.ok and v.entries == 1  # one entry seals the existing rows
        (w,) = raw(path).execute("SELECT w_hi FROM audit_chain").fetchone()
        assert w == 4
        mem = crewai.instrument(StubMemory(pool=False), Capture(led))
        with agent_task_scope("agent", "after"):
            mem.remember("new fact")
        assert audit.verify(led).ok
    tamper(path, "UPDATE writes SET agent_id = 'x' WHERE seq = 1")
    with Ledger(path) as led:
        assert not audit.verify(led).ok


def test_report_covers_the_blast_radius(ledger, chain):
    _, bad, good, d1, d2 = chain
    dod.excise(ledger, "crewai", bad.id)
    rep = audit.report(ledger, "crewai", bad.id)
    assert {i["record_id"] for i in rep["items"]} == {bad.id, d1.id, d2.id}
    assert rep["summary"] == {"deleted": 0, "quarantined": 3, "released": 0, "live": 0}
    assert rep["chain"]["ok"]
    for i in rep["items"]:
        assert i["writes"] and i["quarantine"][0]["chain"]["digest"]
    md = audit.to_markdown(rep)
    assert "verified" in md and bad.id in md and "NOT" not in md


def test_report_shows_deletions(ledger, chain):
    mem, bad, good, d1, d2 = chain
    dod.excise(ledger, "crewai", bad.id, mode="delete",
               stores={"crewai": crewai.DeleteAdapter(mem)})
    rep = audit.report(ledger, "crewai", bad.id)
    assert rep["summary"]["deleted"] == 3
    assert all(i["deletions"][0]["chain"] for i in rep["items"])


def test_report_on_a_live_item(ledger, chain):
    _, _, good, *_ = chain
    rep = audit.report(ledger, "crewai", good.id)
    assert rep["summary"] == {"deleted": 0, "quarantined": 0, "released": 0, "live": 1}


def test_report_unknown_item_raises(ledger, chain):
    with pytest.raises(KeyError):
        audit.report(ledger, "crewai", "nope")


def test_report_holds_no_content(ledger, chain):
    _, bad, *_ = chain
    dod.excise(ledger, "crewai", bad.id)
    assert "peanuts" not in repr(audit.report(ledger, "crewai", bad.id))


def test_delta_chained_sets_verify(ledger):
    # A context that grows and upgrades tags is stored as Fenwick deltas.
    ctx = []
    for i in range(40):
        ctx.append(("mem", f"r{i}", "retrieved_in_scope"))
        if i % 7 == 3:
            ctx.append(("mem", f"r{i - 2}", "in_prompt"))
        ledger.append_write(content=str(i), store="mem", source="agent_inference",
                            context=list(ctx))
    v = audit.verify(ledger)
    assert v.ok, v.problems
