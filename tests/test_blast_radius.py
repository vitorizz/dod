import random
import sqlite3
import threading
import time
from collections import Counter

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


# --- Indexed walk (child_edges) -------------------------------------------


def full_scan_blast_radius(path, store, record_id, max_depth=None):
    """The pre-index algorithm: scan bindings, parents, writes and tombstones
    once, walk in memory. Kept as the reference the indexed walk must match."""
    conn = sqlite3.connect(str(path))
    try:
        start_ids, items_of = [], {}
        for wid, s, r in conn.execute("SELECT write_id, store, record_id FROM bindings"):
            bound = items_of.setdefault(wid, [])
            if (s, r) in bound:
                continue
            bound.append((s, r))
            if s == store and r == record_id:
                start_ids.append(wid)
        if not start_ids:
            raise KeyError((store, record_id))
        start_ids.sort()
        children = {}
        for child, pstore, prec, tag in conn.execute(
            "SELECT write_id, parent_store, parent_record_id, tag FROM parents"
        ):
            children.setdefault((pstore, prec), []).append((child, tag))
        seen = dict.fromkeys(start_ids, 0)
        edges, frontier, depth = [], start_ids, 0
        while frontier and (max_depth is None or depth < max_depth):
            depth += 1
            nxt = []
            for wid in frontier:
                for item in items_of.get(wid, ()):
                    for child, tag in children.get(item, ()):
                        edges.append(Edge(child, item[0], item[1], tag, wid))
                        if child not in seen:
                            seen[child] = depth
                            nxt.append(child)
            frontier = nxt
        truncated = any(
            child not in seen
            for wid in frontier
            for item in items_of.get(wid, ())
            for child, _ in children.get(item, ())
        )
        reached = [
            (r[1], seen[r[1]], r[0], *r[2:])
            for r in conn.execute(
                "SELECT seq, write_id, content_hash, ts, store, agent_id, session_id, source"
                " FROM writes ORDER BY seq"
            )
            if r[1] in seen
        ]
        descendants = sorted((w for w in reached if w[1]), key=lambda w: w[1])
        bindings = {w: sorted(items_of.get(w, ())) for w in seen}
        in_result = {(store, record_id)}.union(*bindings.values())
        deleted = {}
        for r in conn.execute(
            "SELECT seq, store, record_id, ts, reason, write_id, agent_id, session_id"
            " FROM tombstones ORDER BY seq"
        ):
            if (r[1], r[2]) in in_result:
                deleted.setdefault((r[1], r[2]), []).append(tuple(r))
        return dict(
            items=[w for w in reached if not w[1]],
            descendants=descendants,
            edges=sorted(edges),
            bindings=bindings,
            store_counts=dict(sorted(Counter(w[5] for w in descendants).items())),
            truncated=truncated,
            deleted=dict(sorted(deleted.items())),
        )
    finally:
        conn.close()


def comparable(r):
    """``r`` in ``full_scan_blast_radius``'s shape. Edges are compared as a
    multiset: their order within a level was never specified."""
    return dict(
        items=[tuple(w) for w in r.items],
        descendants=[tuple(w) for w in r.descendants],
        edges=sorted(r.edges),
        bindings=r.bindings,
        store_counts=r.store_counts,
        truncated=r.truncated,
        deleted={k: [tuple(t) for t in v] for k, v in r.deleted.items()},
    )


def scalar(path, sql):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


def watermark(path):
    return scalar(path, "SELECT coalesce(max(upto), 0) FROM child_edges_filled")


def max_seq(path):
    return scalar(path, "SELECT max(seq) FROM writes")


def edge_rows(path, table):
    conn = sqlite3.connect(str(path))
    try:
        return sorted(conn.execute(
            f"SELECT parent_store, parent_record_id, write_id, tag FROM {table}"
        ))
    finally:
        conn.close()


def fill_all(ledger, limit=100_000):
    while ledger._fill_child_edges(limit) is not None:
        pass


def random_ledger(ledger, rng, n):
    """Writes with direct parents, shared and overlapping context sets, items
    bound to several writes or twice, unbound and later-bound parents (which
    close cycles), and tombstones."""
    stores = ("mongo", "pgvector", "crm")
    tags = ("in_prompt", "retrieved_in_scope")
    items = [("mongo", "seed")]
    put(ledger, "mongo", "seed", source="user")
    context = []
    for i in range(1, n):
        def pick():
            if rng.random() < 0.05:
                return (rng.choice(stores), f"r{i + rng.randrange(1, 30)}")  # bound later
            if rng.random() < 0.03:
                return ("docs", f"ext-{rng.randrange(50)}")  # never bound
            return rng.choice(items[-40:] if rng.random() < 0.8 else items)

        if rng.random() < 0.2:  # the scope's context changes
            context = [(*pick(), rng.choice(tags)) for _ in range(rng.randrange(0, 6))]
        parents = [(*pick(), rng.choice(tags)) for _ in range(rng.choice((0, 1, 1, 2, 3)))]
        if context and rng.random() < 0.3:  # also a direct parent, maybe re-tagged
            parents.append((*rng.choice(context)[:2], rng.choice(tags)))
        item = (rng.choice(stores), f"r{i}") if rng.random() > 0.05 else rng.choice(items)
        wid = ledger.append_write(
            content=item[1], store=item[0], source="agent_inference",
            parents=parents, context=context if rng.random() < 0.7 else (),
        )
        ledger.bind(wid, *item)
        if rng.random() < 0.02:
            ledger.bind(wid, *item)  # the same binding twice
        if rng.random() < 0.03:
            ledger.append_tombstones([dict(store=item[0], record_id=item[1], reason="forget")])
        items.append(item)
    return items


def test_indexed_walk_matches_full_scan_unindexed_half_and_fully_indexed(path):
    rng = random.Random(34)
    with Ledger(path, checkpoint_interval=3600) as ledger:  # filled by hand here
        items = random_ledger(ledger, rng, 600)
        roots = sorted(set(rng.sample(items, 60)) | {("mongo", "seed")})
        top = max_seq(path)

        def check(stage):
            for root in roots:
                for max_depth in (None, 0, 1, 2, 4):
                    got = comparable(blast_radius(ledger, *root, max_depth=max_depth))
                    assert got == full_scan_blast_radius(path, *root, max_depth), (stage, root)

        assert watermark(path) == 0
        check("unindexed")
        while watermark(path) < top // 2:
            assert ledger._fill_child_edges(37) is not None
        assert 0 < watermark(path) < top
        check("half indexed")
        fill_all(ledger)
        assert watermark(path) == top
        check("fully indexed")
        random_ledger(ledger, rng, 50)  # a tail on top of a full index
        check("indexed plus tail")


def test_child_edges_hold_exactly_the_parents_view(path):
    with Ledger(path, checkpoint_interval=3600) as ledger:
        random_ledger(ledger, random.Random(7), 300)
        fill_all(ledger)
    view = edge_rows(path, "parents")
    assert view and edge_rows(path, "child_edges") == view
    # Each edge's seq is its child write's.
    assert scalar(
        path,
        "SELECT count(*) FROM child_edges c JOIN writes w ON w.seq = c.seq"
        " WHERE w.write_id != c.write_id",
    ) == 0


def test_child_edges_are_append_only(path, ledger):
    put(ledger, "mem", "a")
    put(ledger, "mem", "b", [("mem", "a", "in_prompt")])
    fill_all(ledger)
    conn = sqlite3.connect(str(path), isolation_level=None)
    for sql in (
        "UPDATE child_edges SET tag = 'retrieved_in_scope'",
        "DELETE FROM child_edges",
        "INSERT OR REPLACE INTO child_edges SELECT * FROM child_edges",
        "UPDATE child_edges_filled SET ts = 0",
        "DELETE FROM child_edges_filled",
        "INSERT OR REPLACE INTO child_edges_filled SELECT * FROM child_edges_filled",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(sql)
    assert conn.execute("SELECT count(*) FROM child_edges").fetchone() == (1,)
    conn.close()


def test_background_thread_fills_index(path):
    with Ledger(path, checkpoint_interval=0.02) as ledger:
        put(ledger, "mem", "a")
        b = put(ledger, "pg", "b", [("mem", "a", "in_prompt")])
        deadline = time.monotonic() + 10
        while watermark(path) < max_seq(path) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert watermark(path) == max_seq(path)
        assert edge_rows(path, "child_edges") == [("mem", "a", b, "in_prompt")]
        assert depths(blast_radius(ledger, "mem", "a")) == {b: 1}


def test_two_ledgers_filling_one_file_concurrently(path):
    with Ledger(path, checkpoint_interval=3600) as ledger:
        random_ledger(ledger, random.Random(2), 2_000)
    errors = []
    with Ledger(path, checkpoint_interval=3600) as one, \
            Ledger(path, checkpoint_interval=3600) as two:
        barrier = threading.Barrier(2)

        def fill(led):
            try:
                barrier.wait()
                while led._fill_child_edges(25) is not None:
                    pass
            except BaseException as e:  # reported below
                errors.append(e)

        threads = [threading.Thread(target=fill, args=(led,)) for led in (one, two)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # Nothing missing, nothing twice.
        assert edge_rows(path, "child_edges") == edge_rows(path, "parents")
        assert watermark(path) == max_seq(path)
        for led in (one, two):
            assert comparable(blast_radius(led, "mongo", "seed")) == \
                full_scan_blast_radius(path, "mongo", "seed")


def build_clusters(ledger, n_clusters, size, rng):
    """A root item ``r0``, then ``n_clusters`` random DAGs of ``size`` writes
    each, every cluster's first write a child of ``r0``. The clusters' writes
    are interleaved, so a cluster's edges are spread across the ledger."""
    batch = [dict(content="r0", store="mem", source="user", record_id="r0")]
    made = [0] * n_clusters
    order = list(range(n_clusters)) * size
    rng.shuffle(order)
    for c in order:
        i = made[c]
        made[c] += 1
        if i == 0:
            parents = [("mem", "r0", "in_prompt")]
        else:
            parents = [
                ("pg" if j % 2 else "mem", f"c{c}-{j}", "in_prompt")
                for j in {rng.randrange(max(0, i - 20), i) for _ in range(2)}
            ]
        batch.append(dict(content=f"c{c}-{i}", store="pg" if i % 2 else "mem",
                          source="agent_inference", parents=parents, record_id=f"c{c}-{i}"))
        if len(batch) == 1_000:
            ledger.append_batch(batch)
            batch = []
    ledger.append_batch(batch)


def best_ms(fn, runs):
    fn()  # warm the page cache
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        out = fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return min(samples), out


def test_blast_radius_100k_ledger_indexed(path):
    with Ledger(path, checkpoint_interval=3600) as ledger:
        build_clusters(ledger, 1_000, 100, random.Random(100))
        fill_all(ledger)

        small_ms, r = best_ms(lambda: blast_radius(ledger, "mem", "c500-0"), 5)
        print(f"blast radius over 100k writes, {len(r.descendants)} descendants:"
              f" min {small_ms:.2f} ms")
        assert len(r.descendants) == 99
        assert comparable(r) == full_scan_blast_radius(path, "mem", "c500-0")

        root_ms, r = best_ms(lambda: blast_radius(ledger, "mem", "r0"), 2)
        scan_ms, expected = best_ms(lambda: full_scan_blast_radius(path, "mem", "r0"), 2)
        print(f"blast radius from root over 100k writes: indexed min {root_ms:.0f} ms,"
              f" full scan min {scan_ms:.0f} ms")
        assert len(r.descendants) == 100_000
        assert comparable(r) == expected
    assert small_ms < 50
    assert root_ms < 2 * scan_ms
