import sqlite3

import pytest
from test_crewai import StubMemory

import dod
from dod.capture import Capture
from dod.excise import ExciseResult, Skipped, deletion_reason
from dod.integrations import crewai
from dod.ledger import Ledger
from dod.scope import agent_task_scope

NS = ("users", "u1")


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


def tombstones(path):
    return raw(path).execute(
        "SELECT store, record_id, reason FROM tombstones ORDER BY seq"
    ).fetchall()


def raw(path):
    return sqlite3.connect(str(path), isolation_level=None)


def ids(matches):
    return {m.record.id for m in matches}


# --- One store (CrewAI stub; no optional dependency) --------------------------


@pytest.fixture
def chain(ledger):
    """bad -> d1 -> d2 in CrewAI, plus good -> g1 unrelated to bad."""
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
    with agent_task_scope("agent", "t3"):
        mem.results["good"] = [good]
        mem.recall("good")
        g1 = mem.remember("suggested a Brooklyn venue")
    mem.results["all"] = [bad, good, d1, d2, g1]
    return mem, bad, good, d1, d2, g1


def test_exported_from_package():
    from dod.excise import excise

    assert dod.excise is excise and dod.ExciseResult is ExciseResult


def test_quarantine_is_default_and_touches_no_store(ledger, chain):
    mem, bad, good, d1, d2, g1 = chain
    before = dict(mem._storage.records)
    r = dod.excise(ledger, "crewai", bad.id)
    assert r.mode == "quarantine" and not r.dry_run
    assert r.targets == {"crewai": [bad.id, d1.id, d2.id]}
    assert r.done == r.targets and r.skipped == []
    assert ledger.quarantined() == {("crewai", bad.id), ("crewai", d1.id), ("crewai", d2.id)}
    assert mem._storage.records == before
    assert ids(mem.recall("all")) == {good.id, g1.id}


def test_quarantine_reason_names_the_excised_item(path, ledger, chain):
    _, bad, *_ = chain
    dod.excise(ledger, "crewai", bad.id)
    reasons = {r for (r,) in raw(path).execute("SELECT reason FROM quarantine")}
    assert reasons == {f'excision of ["crewai", "{bad.id}"]'}


def test_excising_a_descendant_leaves_its_ancestors(ledger, chain):
    _, bad, _, d1, d2, _ = chain
    r = dod.excise(ledger, "crewai", d1.id)
    assert r.targets == {"crewai": [d1.id, d2.id]}
    assert ("crewai", bad.id) not in ledger.quarantined()


def test_quarantine_is_idempotent(ledger, chain):
    _, bad, *_ = chain
    dod.excise(ledger, "crewai", bad.id)
    n = len(ledger.quarantine_log())
    r = dod.excise(ledger, "crewai", bad.id)
    assert r.done == {}
    assert [s.reason for s in r.skipped] == ["already quarantined"] * 3
    assert len(ledger.quarantine_log()) == n


def test_dry_run_changes_nothing(path, ledger, chain):
    mem, bad, _, d1, d2, _ = chain
    before = dict(mem._storage.records)
    for mode, stores in (("quarantine", None), ("delete", {"crewai": crewai.DeleteAdapter(mem)})):
        r = dod.excise(ledger, "crewai", bad.id, mode=mode, stores=stores, dry_run=True)
        assert r.dry_run and r.targets == {"crewai": [bad.id, d1.id, d2.id]}
        assert r.done == {} and r.skipped == []
    assert ledger.quarantine_log() == []
    assert tombstones(path) == []
    assert mem._storage.records == before


def test_delete_removes_targets_and_tombstones_them(path, ledger, chain):
    mem, bad, good, d1, d2, g1 = chain
    r = dod.excise(ledger, "crewai", bad.id, mode="delete",
                   stores={"crewai": crewai.DeleteAdapter(mem)})
    assert r.done == {"crewai": [bad.id, d1.id, d2.id]} and r.skipped == []
    assert set(mem._storage.records) == {good.id, g1.id}
    assert tombstones(path) == [("crewai", x.id, "excision") for x in (bad, d1, d2)]
    assert ledger.quarantine_log() == []


def test_delete_is_idempotent(path, ledger, chain):
    mem, bad, *_ = chain
    stores = {"crewai": crewai.DeleteAdapter(mem)}
    dod.excise(ledger, "crewai", bad.id, mode="delete", stores=stores)
    r = dod.excise(ledger, "crewai", bad.id, mode="delete", stores=stores)
    assert r.done == {}
    assert [s.reason for s in r.skipped] == ["not in store"] * 3
    assert len(tombstones(path)) == 3


def test_forget_outside_excise_is_still_forget(path, ledger, chain):
    mem, bad, *_ = chain
    assert deletion_reason() == "forget"
    mem.forget(record_ids=[bad.id])
    assert tombstones(path) == [("crewai", bad.id, "forget")]


def test_delete_requires_stores_and_an_adapter_per_target_store(path, ledger, chain):
    mem, bad, *_ = chain
    before = dict(mem._storage.records)
    with pytest.raises(ValueError, match="requires stores"):
        dod.excise(ledger, "crewai", bad.id, mode="delete")
    with pytest.raises(ValueError, match="no delete adapter.*crewai"):
        dod.excise(ledger, "crewai", bad.id, mode="delete", stores={})
    with pytest.raises(ValueError, match="no delete adapter"):
        dod.excise(ledger, "crewai", bad.id, mode="delete", stores={}, dry_run=True)
    assert mem._storage.records == before and tombstones(path) == []


def test_bad_arguments(ledger, chain):
    _, bad, *_ = chain
    with pytest.raises(ValueError, match="unknown mode"):
        dod.excise(ledger, "crewai", bad.id, mode="wipe")
    with pytest.raises(KeyError):
        dod.excise(ledger, "crewai", "never-recorded")
    with pytest.raises(TypeError, match="instrumented"):
        crewai.DeleteAdapter(StubMemory(pool=False))


def test_failed_delete_is_skipped_and_the_rest_go_ahead(path, ledger, chain):
    mem, bad, _, d1, d2, _ = chain

    class Flaky:
        def delete(self, record_id):
            if record_id == d1.id:
                raise RuntimeError("store down")
            return crewai.DeleteAdapter(mem).delete(record_id)

    r = dod.excise(ledger, "crewai", bad.id, mode="delete", stores={"crewai": Flaky()})
    assert r.done == {"crewai": [bad.id, d2.id]}
    assert r.skipped == [Skipped("crewai", d1.id, "delete failed: RuntimeError: store down")]
    assert deletion_reason() == "forget"


# --- Across stores (needs langgraph) ------------------------------------------


@pytest.fixture
def two_stores(ledger):
    pytest.importorskip("langgraph.store.memory")
    from langgraph.store.memory import InMemoryStore

    from dod.integrations import langgraph

    cap = Capture(ledger)
    mem = crewai.instrument(StubMemory(pool=False), cap)
    lg = langgraph.instrument(InMemoryStore(), cap)
    rid = langgraph.record_id

    with agent_task_scope("agent", "plant"):
        bad = mem.remember("the user is allergic to peanuts")  # planted
        good = mem.remember("the user lives in Brooklyn")
        lg.put(NS, "home", {"text": "Brooklyn"})
    # bad -> (crewai d1, langgraph plan) -> crewai d2 -> langgraph shop
    with agent_task_scope("agent", "t1"):
        mem.results["bad"] = [bad]
        mem.recall("bad")
        d1 = mem.remember("avoid Thai restaurants")
        lg.put(NS, "plan", {"text": "no peanuts in meal plans"})
    with agent_task_scope("agent", "t2"):
        lg.get(NS, "plan")
        d2 = mem.remember("removed peanut butter from the grocery list")
    with agent_task_scope("agent", "t3"):
        mem.results["d2"] = [d2]
        mem.recall("d2")
        lg.put(NS, "shop", {"text": "grocery order without peanuts"})
    # Unrelated work in both stores, after the bad root was planted.
    with agent_task_scope("agent", "t4"):
        mem.results["good"] = [good]
        mem.recall("good")
        lg.get(NS, "home")
        g1 = mem.remember("suggested a Brooklyn venue")
        lg.put(NS, "venue", {"text": "a Brooklyn venue"})

    mem.results["all"] = [bad, good, d1, d2, g1]
    expected = {
        "crewai": [bad.id, d1.id, d2.id],
        "langgraph": [rid(NS, "plan"), rid(NS, "shop")],
    }
    unrelated = {"crewai": {good.id, g1.id}, "langgraph": {"home", "venue"}}
    return langgraph, mem, lg, expected, unrelated


def keys(lg):
    return {i.key for i in lg.search(NS, limit=100)}


@pytest.mark.langgraph
def test_quarantine_across_stores_keeps_unrelated_retrievable(ledger, two_stores):
    _, mem, lg, expected, unrelated = two_stores
    bad_id = expected["crewai"][0]
    r = dod.excise(ledger, "crewai", bad_id)
    assert r.targets == expected and r.done == expected
    assert ledger.quarantined() == {(s, x) for s, xs in expected.items() for x in xs}
    # Hidden from reads in both stores; every unrelated memory still returned.
    assert ids(mem.recall("all")) == unrelated["crewai"]
    assert keys(lg) == unrelated["langgraph"]
    assert crewai.blocked_retrievals()[("langgraph", expected["langgraph"][0])] == 1
    assert lg.get(NS, "plan") is None and lg.get(NS, "home") is not None
    # Nothing was deleted from either store.
    assert len(mem._storage.records) == 5
    assert len(lg._data[NS]) == 4


@pytest.mark.langgraph
def test_delete_across_stores_keeps_unrelated(path, ledger, two_stores):
    langgraph, mem, lg, expected, unrelated = two_stores
    stores = {"crewai": crewai.DeleteAdapter(mem), "langgraph": langgraph.DeleteAdapter(lg)}
    bad_id = expected["crewai"][0]

    r = dod.excise(ledger, "crewai", bad_id, mode="delete", stores=stores)
    assert r.done == expected and r.skipped == []
    assert set(mem._storage.records) == unrelated["crewai"]
    assert keys(lg) == unrelated["langgraph"]
    assert sorted(tombstones(path)) == sorted(
        (s, x, "excision") for s, xs in expected.items() for x in xs
    )

    again = dod.excise(ledger, "crewai", bad_id, mode="delete", stores=stores)
    assert again.done == {} and len(again.skipped) == 5
    assert {s.reason for s in again.skipped} == {"not in store"}
    assert len(tombstones(path)) == 5


@pytest.mark.langgraph
def test_missing_adapter_raises_before_any_delete(path, ledger, two_stores):
    _, mem, lg, expected, _ = two_stores
    with pytest.raises(ValueError, match=r"\['langgraph'\]"):
        dod.excise(ledger, "crewai", expected["crewai"][0], mode="delete",
                   stores={"crewai": crewai.DeleteAdapter(mem)})
    assert len(mem._storage.records) == 5 and len(lg._data[NS]) == 4
    assert tombstones(path) == []


@pytest.mark.langgraph
def test_langgraph_adapter_requires_instrumented_store(two_stores):
    langgraph = two_stores[0]
    from langgraph.store.memory import InMemoryStore

    with pytest.raises(TypeError, match="instrumented"):
        langgraph.DeleteAdapter(InMemoryStore())
