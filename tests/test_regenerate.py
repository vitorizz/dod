import sqlite3

import pytest
from test_crewai import StubMemory

import dod
from dod.capture import Capture
from dod.integrations import crewai
from dod.ledger import Ledger
from dod.query import blast_radius, trace
from dod.regenerate import Failed, Parent
from dod.scope import agent_task_scope, current_key


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


@pytest.fixture
def chain(ledger):
    """bad + good -> d1 -> d2 in CrewAI, plus good -> g1 unrelated to bad."""
    cap = Capture(ledger)
    mem = crewai.instrument(StubMemory(pool=False), cap)
    with agent_task_scope("agent", "plant"):
        bad = mem.remember("the user is allergic to peanuts")
        good = mem.remember("the user lives in Brooklyn")
    with agent_task_scope("agent", "t1"):
        mem.results["q1"] = [bad, good]
        mem.recall("q1")
        d1 = mem.remember("avoid Thai restaurants in Brooklyn")
    with agent_task_scope("agent", "t2"):
        mem.results["q2"] = [d1]
        mem.recall("q2")
        d2 = mem.remember("suggested Italian for dinner")
    with agent_task_scope("agent", "t3"):
        mem.results["q3"] = [good]
        mem.recall("q3")
        g1 = mem.remember("suggested a Brooklyn venue")
    return cap, mem, bad, good, d1, d2, g1


def rewriter(mem, log=None):
    """A rewrite that recalls the task's clean parents and saves a new record."""

    def rewrite(task):
        if log is not None:
            log.append((task.record_id, list(task.parents), current_key()))
        records = [mem._storage.get_record(r) for _, r in task.clean_parents()]
        mem.results["clean"] = [r for r in records if r is not None]
        mem.recall("clean")
        new = mem.remember(f"regenerated from {len(records)} clean input(s)")
        return ("crewai", new.id)

    return rewrite


def recalled(mem, *records):
    mem.results["all"] = list(records)
    return {m.record.id for m in mem.recall("all")}


def test_exported_from_package():
    from dod.regenerate import RegenerateResult, RegenerateTask, regenerate

    assert dod.regenerate is regenerate
    assert dod.RegenerateResult is RegenerateResult and dod.RegenerateTask is RegenerateTask


def test_rewrites_descendants_in_order_from_clean_parents(ledger, chain):
    cap, mem, bad, good, d1, d2, g1 = chain
    log = []
    r = dod.regenerate(ledger, "crewai", bad.id, rewriter(mem, log))

    assert [t.record_id for t in r.tasks] == [d1.id, d2.id]
    assert set(r.replaced) == {("crewai", d1.id), ("crewai", d2.id)}
    assert r.failed == [] and r.dropped == []
    new_d1, new_d2 = r.replaced[("crewai", d1.id)], r.replaced[("crewai", d2.id)]

    # Each rewrite saw what had happened to its parents, in its own scope.
    (rid1, p1, key1), (rid2, p2, key2) = log
    assert rid1 == d1.id and key1 == ("dod-regenerate", "regenerate-0")
    assert {(p.record_id, p.status) for p in p1} == {(bad.id, "bad"), (good.id, "clean")}
    assert rid2 == d2.id and key2 == ("dod-regenerate", "regenerate-1")
    assert p2 == [Parent("crewai", d1.id, "retrieved_in_scope", "replaced", new_d1)]

    # Old targets are quarantined; replacements and unrelated items are live.
    assert r.done == {"crewai": [bad.id, d1.id, d2.id]}
    assert ledger.quarantined() == {("crewai", x.id) for x in (bad, d1, d2)}
    assert recalled(mem, bad, good, d1, d2, g1) == {good.id, g1.id}

    # The replacements' lineage is real and clean.
    t2 = trace(ledger, *new_d2)
    assert {(w.store, w.write_id) for w in t2.ancestors} >= {
        ("crewai", w.write_id) for w in trace(ledger, *new_d1).items
    }
    reached = {e.parent_record_id for e in t2.edges}
    assert reached == {new_d1[1], good.id}
    b = blast_radius(ledger, "crewai", bad.id)
    assert not {i for its in b.bindings.values() for i in its} & {new_d1, new_d2}


def test_dry_run_returns_the_plan_and_changes_nothing(path, ledger, chain):
    cap, mem, bad, good, d1, d2, g1 = chain
    before = dict(mem._storage.records)
    r = dod.regenerate(ledger, "crewai", bad.id, dry_run=True)
    assert r.dry_run and [t.record_id for t in r.tasks] == [d1.id, d2.id]
    assert r.tasks[1].parents == [Parent("crewai", d1.id, "retrieved_in_scope", "replaced", None)]
    assert r.tasks[0].clean_parents() == [("crewai", good.id)]
    assert r.tasks[1].clean_parents() == []
    assert r.targets == {"crewai": [bad.id, d1.id, d2.id]}
    assert ledger.quarantine_log() == [] and mem._storage.records == before


def test_dropped_and_failed_items_propagate_to_dependents(ledger, chain):
    cap, mem, bad, good, d1, d2, g1 = chain
    seen = {}

    def drop_first(task):
        seen[task.record_id] = list(task.parents)
        return None if task.record_id == d1.id else rewriter(mem)(task)

    r = dod.regenerate(ledger, "crewai", bad.id, drop_first)
    assert r.dropped == [("crewai", d1.id)]
    assert [p.status for p in seen[d2.id]] == ["dropped"]
    assert ("crewai", d2.id) in r.replaced


def test_raising_rewrite_is_reported_and_the_rest_go_ahead(ledger, chain):
    cap, mem, bad, good, d1, d2, g1 = chain

    def boom(task):
        if task.record_id == d1.id:
            raise RuntimeError("model down")
        return rewriter(mem)(task)

    r = dod.regenerate(ledger, "crewai", bad.id, boom)
    assert r.failed == [Failed("crewai", d1.id, "rewrite raised RuntimeError: model down")]
    assert ("crewai", d2.id) in r.replaced
    assert ("crewai", d1.id) in ledger.quarantined()


def test_in_place_rewrite_is_refused(ledger, chain):
    cap, mem, bad, good, d1, d2, g1 = chain

    def in_place(task):
        mem.update(task.record_id, "rewritten in place")
        return task.item

    r = dod.regenerate(ledger, "crewai", bad.id, in_place)
    assert [f.record_id for f in r.failed] == [d1.id, d2.id]
    assert all("not in place" in f.reason for f in r.failed)
    assert r.replaced == {}


def test_uncaptured_replacement_is_refused(ledger, chain):
    cap, mem, bad, good, d1, d2, g1 = chain

    def uncaptured(task):
        (rec,) = mem.add("raw-" + task.record_id)  # straight into storage
        return ("crewai", rec.id)

    r = dod.regenerate(ledger, "crewai", bad.id, uncaptured)
    assert all("no recorded write" in f.reason for f in r.failed)
    assert ("crewai", "raw-" + d1.id) in ledger.quarantined()


def test_replacement_derived_from_a_target_is_refused_and_quarantined(ledger, chain):
    cap, mem, bad, good, d1, d2, g1 = chain

    def contaminated(task):
        # A compaction naming the old item as its parent, as reading it would.
        w = cap.on_compaction(current_key(), [task.item], "copy", "crewai", "agent_inference")
        new = "copy-" + task.record_id
        cap.bind(w, new)
        return ("crewai", new)

    r = dod.regenerate(ledger, "crewai", bad.id, contaminated)
    assert all("derived from target" in f.reason for f in r.failed)
    assert ("crewai", "copy-" + d1.id) in ledger.quarantined()


def test_then_delete_removes_old_targets_through_adapters(path, ledger, chain):
    cap, mem, bad, good, d1, d2, g1 = chain
    r = dod.regenerate(ledger, "crewai", bad.id, rewriter(mem), then="delete",
                       stores={"crewai": crewai.DeleteAdapter(mem)})
    assert r.done == {"crewai": [bad.id, d1.id, d2.id]}
    live = set(mem._storage.records)
    assert not live & {bad.id, d1.id, d2.id}
    assert {r for _, r in r.replaced.values()} <= live and {good.id, g1.id} <= live
    reasons = sqlite3.connect(str(path)).execute(
        "SELECT DISTINCT reason FROM tombstones").fetchall()
    assert reasons == [("excision",)]


def test_bad_arguments(path, ledger, chain):
    cap, mem, bad, *_ = chain
    before = dict(mem._storage.records)
    with pytest.raises(ValueError, match="rewrite is required"):
        dod.regenerate(ledger, "crewai", bad.id)
    with pytest.raises(ValueError, match="unknown then"):
        dod.regenerate(ledger, "crewai", bad.id, rewriter(mem), then="wipe")
    with pytest.raises(ValueError, match="requires stores"):
        dod.regenerate(ledger, "crewai", bad.id, rewriter(mem), then="delete")
    with pytest.raises(ValueError, match="no delete adapter"):
        dod.regenerate(ledger, "crewai", bad.id, rewriter(mem), then="delete", stores={})
    with pytest.raises(KeyError):
        dod.regenerate(ledger, "crewai", "never-recorded", rewriter(mem))
    assert ledger.quarantine_log() == [] and mem._storage.records == before


def test_cycles_are_broken_and_every_item_is_visited(ledger, chain):
    cap, mem, bad, good, d1, d2, g1 = chain
    # d1 is later rewritten with d2 (its own descendant) in context.
    with agent_task_scope("agent", "t4"):
        mem.results["q4"] = [d2]
        mem.recall("q4")
        mem.update(d1.id, "avoid Thai and Italian")
    r = dod.regenerate(ledger, "crewai", bad.id, rewriter(mem))
    assert sorted(t.record_id for t in r.tasks) == sorted([d1.id, d2.id])
    assert len(r.replaced) == 2 and r.failed == []


def test_across_stores(ledger):
    pytest.importorskip("langgraph.store.memory")
    from langgraph.store.memory import InMemoryStore

    from dod.integrations import langgraph

    cap = Capture(ledger)
    mem = crewai.instrument(StubMemory(pool=False), cap)
    lg = langgraph.instrument(InMemoryStore(), cap)
    ns = ("users", "u1")
    with agent_task_scope("agent", "plant"):
        bad = mem.remember("refunds within 90 days")
    with agent_task_scope("agent", "t1"):
        mem.results["q"] = [bad]
        mem.recall("q")
        lg.put(ns, "reply", {"text": "yes, refund ok"})

    def rewrite(task):
        assert task.store == "langgraph"
        assert lg.get(ns, "reply") is None  # quarantined while rewriting
        lg.put(ns, "reply-v2", {"text": "checking the 30-day policy"})
        return ("langgraph", langgraph.record_id(ns, "reply-v2"))

    r = dod.regenerate(ledger, "crewai", bad.id, rewrite)
    assert r.replaced == {("langgraph", langgraph.record_id(ns, "reply")):
                          ("langgraph", langgraph.record_id(ns, "reply-v2"))}
    assert lg.get(ns, "reply-v2") is not None
