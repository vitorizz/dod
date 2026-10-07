import sqlite3
import time

import pytest

from dod.ledger import Ledger
from dod.query import Edge, UnrecordedAncestor, trace


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
    return {w.write_id: w.depth for w in result.ancestors}


def test_chain_depths_and_root(ledger):
    a = put(ledger, "mem", "a", source="user")
    b = put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    c = put(ledger, "mem", "c", [("mem", "b", "in_prompt")])
    r = trace(ledger, "mem", "c")
    assert [w.write_id for w in r.items] == [c]
    assert depths(r) == {b: 1, a: 2}
    assert [w.write_id for w in r.roots] == [a]
    assert r.roots[0].source == "user"
    assert not r.unrecorded and not r.truncated


def test_cross_store_chain(ledger):
    a = put(ledger, "mongo", "m1", source="user")
    b = put(ledger, "pgvector", "v1", [("mongo", "m1", "in_prompt")])
    c = put(ledger, "mongo", "m2", [("pgvector", "v1", "retrieved_in_scope")])
    d = put(ledger, "pgvector", "v2", [("mongo", "m2", "in_prompt")])
    r = trace(ledger, "pgvector", "v2")
    assert depths(r) == {c: 1, b: 2, a: 3}
    assert {w.write_id: w.store for w in r.ancestors} == {c: "mongo", b: "pgvector", a: "mongo"}
    assert [w.write_id for w in r.roots] == [a]
    assert {(e.child_write_id, e.parent_write_id) for e in r.edges} == {(d, c), (c, b), (b, a)}


def test_edge_tags_returned_and_filterable(ledger):
    a = put(ledger, "mem", "a")
    x = put(ledger, "mem", "x")
    b = put(ledger, "pg", "b", [("mem", "a", "in_prompt"), ("mem", "x", "retrieved_in_scope")])
    c = put(ledger, "mem", "c", [("pg", "b", "retrieved_in_scope")])
    d = put(ledger, "mem", "d", [("mem", "c", "in_prompt")])
    r = trace(ledger, "mem", "d")
    assert set(r.edges) == {
        Edge(d, "mem", "c", "in_prompt", c),
        Edge(c, "pg", "b", "retrieved_in_scope", b),
        Edge(b, "mem", "a", "in_prompt", a),
        Edge(b, "mem", "x", "retrieved_in_scope", x),
    }
    assert {(e.child_write_id, e.parent_write_id) for e in r.edges_tagged("in_prompt")} == {
        (d, c),
        (b, a),
    }
    with pytest.raises(ValueError):
        r.edges_tagged("bogus")


def test_unknown_item_raises_keyerror(ledger):
    put(ledger, "mem", "a")
    with pytest.raises(KeyError):
        trace(ledger, "mem", "missing")
    with pytest.raises(KeyError):
        trace(ledger, "other", "a")


def test_item_without_parents_is_its_own_root(ledger):
    a = put(ledger, "mem", "a", source="user")
    r = trace(ledger, "mem", "a")
    assert r.ancestors == [] and r.edges == []
    assert [(w.write_id, w.depth) for w in r.roots] == [(a, 0)]


def test_diamond_lists_ancestor_once_at_min_depth(ledger):
    # d -> b -> a, d -> c -> b2 -> a: a is at depth 2 and 3.
    a = put(ledger, "mem", "a")
    b = put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    b2 = put(ledger, "mem", "b2", [("mem", "a", "in_prompt")])
    c = put(ledger, "pg", "c", [("mem", "b2", "in_prompt")])
    put(ledger, "mem", "d", [("mem", "b", "in_prompt"), ("pg", "c", "in_prompt")])
    r = trace(ledger, "mem", "d")
    assert [w.write_id for w in r.ancestors].count(a) == 1
    assert depths(r) == {b: 1, c: 1, b2: 2, a: 2}
    assert [w.write_id for w in r.roots] == [a]
    # Both edges into a were walked.
    assert sum(e.parent_write_id == a for e in r.edges) == 2


def test_cycle_terminates(path, ledger):
    a = put(ledger, "mem", "a")
    b = put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    c = put(ledger, "mem", "c", [("mem", "b", "in_prompt")])
    # Hand-insert a -> c, closing a -> c -> b -> a.
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("INSERT INTO parents VALUES (?, 'mem', 'c', 'in_prompt')", (a,))
    conn.close()
    r = trace(ledger, "mem", "c")
    assert depths(r) == {b: 1, a: 2}
    assert r.roots == []
    assert Edge(a, "mem", "c", "in_prompt", c) in r.edges


def test_max_depth_truncates(ledger):
    ids = [put(ledger, "mem", "r0", source="user")]
    for i in range(1, 6):
        ids.append(put(ledger, "mem", f"r{i}", [("mem", f"r{i - 1}", "in_prompt")]))
    r = trace(ledger, "mem", "r5", max_depth=2)
    assert depths(r) == {ids[4]: 1, ids[3]: 2}
    assert r.roots == [] and r.truncated
    assert all(e.child_write_id in (ids[5], ids[4]) for e in r.edges)

    r = trace(ledger, "mem", "r5", max_depth=5)
    assert [w.write_id for w in r.roots] == [ids[0]] and not r.truncated

    r = trace(ledger, "mem", "r5", max_depth=0)
    assert r.ancestors == [] and r.edges == [] and r.truncated

    with pytest.raises(ValueError):
        trace(ledger, "mem", "r5", max_depth=-1)


def test_unbound_parent_reported_as_unrecorded(ledger):
    a = put(ledger, "mem", "a")
    b = put(ledger, "mem", "b", [("mem", "a", "in_prompt"), ("docs", "ext-1", "retrieved_in_scope")])
    c = put(ledger, "mem", "c", [("mem", "b", "in_prompt"), ("docs", "ext-2", "in_prompt")])
    r = trace(ledger, "mem", "c")
    assert depths(r) == {b: 1, a: 2}
    assert r.unrecorded == [
        UnrecordedAncestor("docs", "ext-2", 1),
        UnrecordedAncestor("docs", "ext-1", 2),
    ]
    assert Edge(b, "docs", "ext-1", "retrieved_in_scope", None) in r.edges
    assert Edge(c, "docs", "ext-2", "in_prompt", None) in r.edges


def test_record_bound_to_several_writes_resolves_to_all(ledger):
    # Over-include: an item rebound to a later write resolves to both.
    a1 = put(ledger, "mem", "a")
    a2 = put(ledger, "mem", "a")
    b = put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    r = trace(ledger, "mem", "b")
    assert [w.write_id for w in r.items] == [b]
    assert depths(r) == {a1: 1, a2: 1}
    r = trace(ledger, "mem", "a")
    assert [w.write_id for w in r.items] == [a1, a2]


def test_trace_10k_ledger_depth_50_under_100ms(path):
    with Ledger(path) as ledger:
        # 50-deep chain of the traced item, each step also seeing noise records.
        prev = None
        for i in range(10_000 - 50):
            put(ledger, "noise", f"n{i}", [("noise", f"n{i - 1}", "retrieved_in_scope")] if i else ())
        for i in range(51):
            parents = [("noise", f"n{i * 7}", "retrieved_in_scope")]
            if prev:
                parents.append(("mem" if i % 2 else "pg", prev, "in_prompt"))
            store = "pg" if i % 2 else "mem"
            put(ledger, store, f"c{i}", parents)
            prev = f"c{i}"
        leaf_store = "mem" if 50 % 2 == 0 else "pg"

        trace(ledger, leaf_store, prev, max_depth=50)  # warm the page cache
        start = time.perf_counter()
        r = trace(ledger, leaf_store, prev, max_depth=50)
        elapsed_ms = (time.perf_counter() - start) * 1e3
        print(f"trace depth 50 over 10k writes: {elapsed_ms:.2f} ms")
    assert max(w.depth for w in r.ancestors) == 50
    assert elapsed_ms < 100
