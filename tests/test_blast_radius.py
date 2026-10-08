import random
import sqlite3
import time

import pytest

from dod.ledger import Ledger
from dod.query import Edge, blast_radius, trace


@pytest.fixture
def path(tmp_path):
    return tmp_path / "ledger.db"


@pytest.fixture
def ledger(path):
    with Ledger(path) as led:
        yield led


def put(ledger, store, record_id, parents=(), source="agent_inference"):
    """Append a write with ``(store, record_id, tag)`` parents and bind it."""
    wid = ledger.append_write(content=record_id, store=store, source=source, parents=parents)
    ledger.bind(wid, store, record_id)
    return wid


def depths(result):
    return {w.write_id: w.depth for w in result.descendants}


def test_chain_depths_bindings_and_counts(ledger):
    a = put(ledger, "mem", "a", source="user")
    b = put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    c = put(ledger, "mem", "c", [("mem", "b", "in_prompt")])
    r = blast_radius(ledger, "mem", "a")
    assert [w.write_id for w in r.items] == [a]
    assert depths(r) == {b: 1, c: 2}
    assert [w.write_id for w in r.descendants] == [b, c]
    assert r.bindings == {a: [("mem", "a")], b: [("mem", "b")], c: [("mem", "c")]}
    assert r.store_counts == {"mem": 2}
    assert not r.truncated


def test_descendant_in_second_store_included(ledger):
    a = put(ledger, "mongo", "m1", source="user")
    b = put(ledger, "pgvector", "v1", [("mongo", "m1", "in_prompt")])
    c = put(ledger, "mongo", "m2", [("pgvector", "v1", "retrieved_in_scope")])
    d = put(ledger, "pgvector", "v2", [("mongo", "m2", "in_prompt")])
    put(ledger, "mongo", "unrelated")
    r = blast_radius(ledger, "mongo", "m1")
    assert depths(r) == {b: 1, c: 2, d: 3}
    assert r.store_counts == {"mongo": 1, "pgvector": 2}
    assert r.bindings[d] == [("pgvector", "v2")]
    assert {(e.parent_write_id, e.child_write_id) for e in r.edges} == {(a, b), (b, c), (c, d)}


def test_edge_tags_returned_and_filterable(ledger):
    a = put(ledger, "mem", "a")
    b = put(ledger, "pg", "b", [("mem", "a", "in_prompt")])
    c = put(ledger, "mem", "c", [("mem", "a", "retrieved_in_scope")])
    d = put(ledger, "mem", "d", [("pg", "b", "retrieved_in_scope"), ("mem", "c", "in_prompt")])
    e = put(ledger, "pg", "e", [("mem", "d", "in_prompt")])
    r = blast_radius(ledger, "mem", "a")
    assert set(r.edges) == {
        Edge(b, "mem", "a", "in_prompt", a),
        Edge(c, "mem", "a", "retrieved_in_scope", a),
        Edge(d, "pg", "b", "retrieved_in_scope", b),
        Edge(d, "mem", "c", "in_prompt", c),
        Edge(e, "mem", "d", "in_prompt", d),
    }
    assert {(e_.parent_write_id, e_.child_write_id) for e_ in r.edges_tagged("in_prompt")} == {
        (a, b),
        (c, d),
        (d, e),
    }
    assert {(e_.parent_write_id, e_.child_write_id) for e_ in r.edges_tagged("retrieved_in_scope")} == {
        (a, c),
        (b, d),
    }
    with pytest.raises(ValueError):
        r.edges_tagged("bogus")


def test_unknown_item_raises_keyerror(ledger):
    put(ledger, "mem", "a")
    with pytest.raises(KeyError):
        blast_radius(ledger, "mem", "missing")
    with pytest.raises(KeyError):
        blast_radius(ledger, "other", "a")


def test_item_nothing_derives_from_is_empty(ledger):
    put(ledger, "mem", "a")
    leaf = put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    r = blast_radius(ledger, "mem", "b")
    assert [w.write_id for w in r.items] == [leaf]
    assert r.descendants == [] and r.edges == [] and r.store_counts == {}
    assert not r.truncated


def test_diamond_lists_descendant_once_at_min_depth(ledger):
    # a -> b -> d, a -> b2 -> c -> d: d is at depth 2 and 3.
    put(ledger, "mem", "a")
    b = put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    b2 = put(ledger, "mem", "b2", [("mem", "a", "in_prompt")])
    c = put(ledger, "pg", "c", [("mem", "b2", "in_prompt")])
    d = put(ledger, "mem", "d", [("mem", "b", "in_prompt"), ("pg", "c", "in_prompt")])
    r = blast_radius(ledger, "mem", "a")
    assert [w.write_id for w in r.descendants].count(d) == 1
    assert depths(r) == {b: 1, b2: 1, c: 2, d: 2}
    # Both edges into d were walked.
    assert sum(e.child_write_id == d for e in r.edges) == 2


def test_cycle_terminates(path, ledger):
    a = put(ledger, "mem", "a")
    b = put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    c = put(ledger, "mem", "c", [("mem", "b", "in_prompt")])
    # Hand-insert c as a parent of a, closing a -> b -> c -> a.
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("INSERT INTO direct_parents VALUES (?, 'mem', 'c', 'in_prompt')", (a,))
    conn.close()
    r = blast_radius(ledger, "mem", "a")
    assert depths(r) == {b: 1, c: 2}
    assert Edge(a, "mem", "c", "in_prompt", c) in r.edges
    assert not r.truncated


def test_max_depth_truncates(ledger):
    ids = [put(ledger, "mem", "r0", source="user")]
    for i in range(1, 6):
        ids.append(put(ledger, "mem", f"r{i}", [("mem", f"r{i - 1}", "in_prompt")]))
    r = blast_radius(ledger, "mem", "r0", max_depth=2)
    assert depths(r) == {ids[1]: 1, ids[2]: 2}
    assert r.truncated
    assert all(e.parent_write_id in (ids[0], ids[1]) for e in r.edges)

    r = blast_radius(ledger, "mem", "r0", max_depth=5)
    assert len(r.descendants) == 5 and not r.truncated

    r = blast_radius(ledger, "mem", "r0", max_depth=0)
    assert r.descendants == [] and r.edges == [] and r.truncated

    with pytest.raises(ValueError):
        blast_radius(ledger, "mem", "r0", max_depth=-1)


def test_record_bound_to_several_writes_walks_from_all(ledger):
    # Over-include: an item rebound to a later write starts from both.
    a1 = put(ledger, "mem", "a")
    a2 = put(ledger, "mem", "a")
    b = put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    r = blast_radius(ledger, "mem", "a")
    assert [w.write_id for w in r.items] == [a1, a2]
    assert depths(r) == {b: 1}
    assert {e.parent_write_id for e in r.edges} == {a1, a2}


def test_consistent_with_trace_on_random_dag(ledger):
    """X in blast_radius(R) iff trace(X) reaches R, over a random DAG."""
    rng = random.Random(15)
    stores = ("mongo", "pgvector", "crm")
    items = []
    for i in range(500):
        parents = []
        for _ in range(rng.choice((0, 0, 1, 1, 2, 3)) if items else 0):
            pool = items[-60:] if rng.random() < 0.7 else items
            parents.append((*rng.choice(pool), rng.choice(("in_prompt", "retrieved_in_scope"))))
        if rng.random() < 0.05:
            parents.append(("docs", f"ext-{i}", "retrieved_in_scope"))  # unbound parent
        item = (rng.choice(stores), f"r{i}")
        put(ledger, *item, parents)
        items.append(item)

    write_of = {item: blast_radius(ledger, *item).items[0].write_id for item in items}
    ancestors = {
        item: {w.write_id for w in trace(ledger, *item).ancestors} for item in items
    }
    for root in items:
        blast = {w.write_id for w in blast_radius(ledger, *root).descendants}
        derived = {write_of[x] for x in items if write_of[root] in ancestors[x]}
        assert blast == derived, root
    assert any(len(blast_radius(ledger, *r).store_counts) > 1 for r in items[:20])


def test_deleted_items_reported_and_still_walked(ledger):
    a = put(ledger, "mem", "a", source="user")
    b = put(ledger, "pg", "b", [("mem", "a", "in_prompt")])
    c = put(ledger, "mem", "c", [("pg", "b", "in_prompt")])
    put(ledger, "mem", "unrelated")
    for store, rid, reason in (("pg", "b", "consolidation"), ("mem", "a", "forget"),
                               ("mem", "unrelated", "forget"), ("mem", "b", "forget")):
        ledger.append_tombstones([dict(store=store, record_id=rid, reason=reason)])
    r = blast_radius(ledger, "mem", "a")
    assert depths(r) == {b: 1, c: 2}  # a deleted item's descendants are still reached
    assert sorted(r.deleted) == [("mem", "a"), ("pg", "b")]
    assert r.deleted[("pg", "b")][0].reason == "consolidation"
    assert r.is_deleted("mem", "a") and not r.is_deleted("mem", "c")
    assert blast_radius(ledger, "mem", "c").deleted == {}


def test_blast_radius_10k_ledger_from_root_under_250ms(path):
    rng = random.Random(10_000)
    with Ledger(path) as ledger:
        put(ledger, "mem", "r0", source="user")
        for i in range(1, 10_000):
            parents = {
                ("mem" if j % 2 == 0 else "pg", f"r{j}", "in_prompt")
                for j in rng.sample(range(max(0, i - 200), i), min(i, 2))
            }
            put(ledger, "mem" if i % 2 == 0 else "pg", f"r{i}", parents)

        blast_radius(ledger, "mem", "r0")  # warm the page cache
        # Best of several runs: the walk is O(ledger), so the minimum is its
        # cost and the rest is scheduler noise. Measured ~65-100 ms by hardware.
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            r = blast_radius(ledger, "mem", "r0")
            samples.append((time.perf_counter() - start) * 1e3)
        elapsed_ms = min(samples)
        print(
            f"blast radius from root over 10k writes: {len(r.descendants)} descendants, "
            f"min {elapsed_ms:.2f} ms, median {sorted(samples)[2]:.2f} ms"
        )
    assert len(r.descendants) > 9_000
    assert elapsed_ms < 250
