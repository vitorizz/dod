import random
import sqlite3
import time

import pytest

from dod.ledger import TAGS, Ledger
from dod.query import Edge, UnrecordedAncestor, blast_radius, trace


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
    conn.execute("INSERT INTO direct_parents VALUES (?, 'mem', 'c', 'in_prompt')", (a,))
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


def _forget(ledger, store, record_id, reason="forget", write_id=None):
    ledger.append_tombstones(
        [dict(store=store, record_id=record_id, reason=reason, write_id=write_id)]
    )


def test_deleted_items_reported_and_still_walked(ledger):
    a = put(ledger, "mem", "a", source="user")
    b = put(ledger, "pg", "b", [("mem", "a", "in_prompt")])
    c = put(ledger, "mem", "c", [("pg", "b", "in_prompt"), ("mem", "gone", "retrieved_in_scope")])
    _forget(ledger, "mem", "a", "consolidation", write_id=b)
    _forget(ledger, "mem", "a", "excision")
    _forget(ledger, "mem", "gone")  # unrecorded, but its deletion was recorded
    _forget(ledger, "pg", "a")  # same record ID in another store: not in the result
    _forget(ledger, "mem", "c")
    r = trace(ledger, "mem", "c")
    assert depths(r) == {b: 1, a: 2}  # deletion doesn't cut lineage
    assert sorted(r.deleted) == [("mem", "a"), ("mem", "c"), ("mem", "gone")]
    assert [(t.reason, t.write_id) for t in r.deleted[("mem", "a")]] == [
        ("consolidation", b), ("excision", None),
    ]
    assert r.is_deleted("mem", "c") and not r.is_deleted("pg", "b")
    assert not r.is_deleted("pg", "a")


def test_no_tombstones_means_nothing_deleted(ledger):
    put(ledger, "mem", "a")
    put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    assert trace(ledger, "mem", "b").deleted == {}


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


# Equivalence of the two parent representations. "Old" stores every edge as a
# per-write row (the schema before context sets; what ``direct_parents``
# holds when nothing is passed as ``context``). "New" passes the scope's
# context as a shared, content-addressed context set. Queries must not be
# able to tell them apart.

STORES = ("mem", "pg")


def _random_ops(rng, n_writes=200, n_items=60):
    def item():
        return (rng.choice(STORES), f"r{rng.randrange(n_items)}", rng.choice(TAGS))

    scopes = [[] for _ in range(4)]
    batches = []
    while sum(len(b) for b in batches) < n_writes:
        ctx = scopes[rng.randrange(len(scopes))]
        if rng.random() < 0.1:
            ctx.clear()  # the scope ended; a new one starts empty
        ctx += [item() for _ in range(rng.choice([0, 0, 1, 2, 5]))]
        shared = list(ctx)  # one object per batch, as Capture.on_batch passes
        batch = []
        for _ in range(rng.choice([1, 1, 1, 3, 5])):
            batch.append(dict(
                replaced=[item() for _ in range(rng.choice([0, 0, 0, 1, 3]))],
                context=shared,
                store=rng.choice(STORES),
                record_id=None if rng.random() < 0.1 else f"r{rng.randrange(n_items)}",
            ))
        batches.append(batch)
    deletes = [
        dict(store=rng.choice(STORES), record_id=f"r{rng.randrange(n_items)}", reason="forget")
        for _ in range(10)
    ]
    return batches, deletes


def _build(ledger, batches, deletes, split):
    ids = []
    for batch in batches:
        rows = [
            dict(
                content=str(len(ids) + i),
                store=w["store"],
                source="agent_inference",
                record_id=w["record_id"],
                parents=w["replaced"] if split else w["replaced"] + w["context"],
                context=w["context"] if split else (),
            )
            for i, w in enumerate(batch)
        ]
        ids += ledger.append_batch(rows)
    ledger.append_tombstones(deletes)
    return {wid: i for i, wid in enumerate(ids)}


def _norm_write(w, idx):
    return (idx[w.write_id], w.depth, w.content_hash, w.store, w.source)


def _norm_edges(edges, idx):
    return sorted(
        (idx[e.child_write_id], e.parent_store, e.parent_record_id, e.tag,
         idx.get(e.parent_write_id, -1))
        for e in edges
    )


def _norm_deleted(deleted, idx):
    return {k: [(t.reason, idx.get(t.write_id)) for t in ts] for k, ts in deleted.items()}


def _norm_trace(r, idx):
    return (
        [_norm_write(w, idx) for w in r.items],
        [_norm_write(w, idx) for w in r.ancestors],
        [_norm_write(w, idx) for w in r.roots],
        _norm_edges(r.edges, idx),
        r.unrecorded,
        r.truncated,
        _norm_deleted(r.deleted, idx),
    )


def _norm_blast(r, idx):
    return (
        [_norm_write(w, idx) for w in r.items],
        [_norm_write(w, idx) for w in r.descendants],
        _norm_edges(r.edges, idx),
        {idx[w]: items for w, items in r.bindings.items()},
        r.store_counts,
        r.truncated,
        _norm_deleted(r.deleted, idx),
    )


def _edges_by_write(ledger, idx):
    rows = ledger._conn.execute(
        "SELECT write_id, parent_store, parent_record_id, tag FROM parents"
    ).fetchall()
    return sorted((idx[w], s, r, t) for w, s, r, t in rows)


def _count(ledger, table):
    return ledger._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.mark.parametrize("seed", range(5))
def test_context_sets_query_like_per_write_edges(tmp_path, seed):
    rng = random.Random(seed)
    batches, deletes = _random_ops(rng)
    with Ledger(tmp_path / "old.db") as old, Ledger(tmp_path / "new.db") as new:
        old_idx = _build(old, batches, deletes, split=False)
        new_idx = _build(new, batches, deletes, split=True)
        # The new ledger really shares context: fewer stored rows than edges.
        assert _count(old, "context_sets") == 0
        assert _count(new, "direct_parents") + _count(new, "context_members") < _count(
            old, "direct_parents"
        )

        assert _edges_by_write(old, old_idx) == _edges_by_write(new, new_idx)
        bound = sorted({(w["store"], w["record_id"]) for b in batches for w in b
                        if w["record_id"] is not None})
        for store, record_id in rng.sample(bound, 25):
            for max_depth in (None, 1):
                assert _norm_trace(trace(old, store, record_id, max_depth), old_idx) == \
                    _norm_trace(trace(new, store, record_id, max_depth), new_idx)
                assert _norm_blast(blast_radius(old, store, record_id, max_depth), old_idx) == \
                    _norm_blast(blast_radius(new, store, record_id, max_depth), new_idx)


def test_shared_context_with_different_direct_parents_in_one_batch(tmp_path):
    # Two writes in one batch share the context object but have their own
    # direct parents, one overlapping the context with the other tag.
    context = [("mem", "c1", "retrieved_in_scope"), ("pg", "c2", "in_prompt")]
    batches = [
        [dict(replaced=[], context=[], store="mem", record_id="c1")],  # write 0
        [dict(replaced=[], context=[], store="pg", record_id="c2")],  # write 1
        [
            dict(  # write 2
                replaced=[("mem", "c1", "in_prompt"), ("pg", "d0", "retrieved_in_scope")],
                context=context, store="mem", record_id="a",
            ),
            dict(  # write 3
                replaced=[("pg", "c2", "retrieved_in_scope"), ("mem", "d1", "in_prompt")],
                context=context, store="pg", record_id="b",
            ),
        ],
    ]
    expected = [
        # Write 2: c1 is in its direct parents as in_prompt, which beats the
        # context's retrieved_in_scope.
        (2, "mem", "c1", "in_prompt"),
        (2, "pg", "c2", "in_prompt"),
        (2, "pg", "d0", "retrieved_in_scope"),
        # Write 3: c2 stays in_prompt from the context; c1 keeps the
        # context's tag, unaffected by write 2's direct parent.
        (3, "mem", "c1", "retrieved_in_scope"),
        (3, "mem", "d1", "in_prompt"),
        (3, "pg", "c2", "in_prompt"),
    ]
    with Ledger(tmp_path / "old.db") as old, Ledger(tmp_path / "new.db") as new:
        old_idx = _build(old, batches, [], split=False)
        new_idx = _build(new, batches, [], split=True)
        assert _edges_by_write(old, old_idx) == expected
        assert _edges_by_write(new, new_idx) == expected
        assert (_count(new, "context_sets"), _count(new, "context_members")) == (1, 2)
        assert _count(new, "direct_parents") == 4
        for store, record_id in (("mem", "a"), ("pg", "b")):
            r = trace(new, store, record_id)
            assert _norm_trace(r, new_idx) == _norm_trace(trace(old, store, record_id), old_idx)
            assert sorted(new_idx[w.write_id] for w in r.ancestors) == [0, 1]
        for store, record_id in (("mem", "c1"), ("pg", "c2")):
            r = blast_radius(new, store, record_id)
            assert _norm_blast(r, new_idx) == _norm_blast(blast_radius(old, store, record_id), old_idx)
            assert sorted(new_idx[w.write_id] for w in r.descendants) == [2, 3]
