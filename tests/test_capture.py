import sqlite3
import threading

import pytest

from dod.capture import Capture
from dod.ledger import Ledger


@pytest.fixture
def path(tmp_path):
    return tmp_path / "ledger.db"


@pytest.fixture
def ledger(path):
    with Ledger(path) as led:
        yield led


@pytest.fixture
def cap(ledger):
    return Capture(ledger)


def raw(path):
    return sqlite3.connect(str(path), isolation_level=None)


def parents(path, write_id):
    rows = raw(path).execute(
        "SELECT parent_store, parent_record_id, tag FROM parents WHERE write_id=?",
        (write_id,),
    )
    return {(s, r): t for s, r, t in rows}


def n_writes(path):
    return raw(path).execute("SELECT COUNT(*) FROM writes").fetchone()[0]


def write(cap, key="k", content="x", store="mem", source="agent_inference", **kw):
    return cap.on_write(key, content, store, source, **kw)


def test_write_parents_are_items_retrieved_under_key(path, cap):
    cap.on_retrieve("k", [("mem", "a"), ("mem", "b")])
    wid = write(cap, agent_id="a1", session_id="s1")
    assert parents(path, wid) == {
        ("mem", "a"): "retrieved_in_scope",
        ("mem", "b"): "retrieved_in_scope",
    }
    row = raw(path).execute(
        "SELECT store, source, agent_id, session_id FROM writes WHERE write_id=?", (wid,)
    ).fetchone()
    assert row == ("mem", "agent_inference", "a1", "s1")


def test_other_keys_never_included(path, cap):
    cap.on_retrieve("k1", [("mem", "a")])
    cap.on_retrieve("k2", [("mem", "b")])
    assert parents(path, write(cap, key="k1")) == {("mem", "a"): "retrieved_in_scope"}
    assert parents(path, write(cap, key="k2")) == {("mem", "b"): "retrieved_in_scope"}
    assert parents(path, write(cap, key="k3")) == {}


@pytest.mark.parametrize("order", [("in_prompt", "retrieved_in_scope"), ("retrieved_in_scope", "in_prompt")])
def test_duplicate_with_different_tags_keeps_in_prompt(path, cap, order):
    for tag in order:
        cap.on_retrieve("k", [("mem", "a")], tag=tag)
    assert parents(path, write(cap)) == {("mem", "a"): "in_prompt"}


def test_unknown_tag_rejected(cap):
    with pytest.raises(ValueError):
        cap.on_retrieve("k", [("mem", "a")], tag="maybe")


def test_over_includes_every_retrieval_since_reset(path, cap):
    cap.on_retrieve("k", [("mem", "a")])
    cap.on_retrieve("k", [("mem", "b")])
    cap.on_retrieve("k", [("vec", "c")])
    # Recording only the latest retrieval would miss a and b.
    assert set(parents(path, write(cap))) == {("mem", "a"), ("mem", "b"), ("vec", "c")}


def test_context_persists_across_writes_until_reset(path, cap):
    cap.on_retrieve("k", [("mem", "a")])
    write(cap)
    assert set(parents(path, write(cap))) == {("mem", "a")}
    cap.reset("k")
    assert parents(path, write(cap)) == {}


def test_write_without_retrieval_has_no_parents(path, cap):
    wid = write(cap)
    assert n_writes(path) == 1
    assert parents(path, wid) == {}


def test_retrieval_without_write_records_nothing(path, cap):
    cap.on_retrieve("k", [("mem", "a")])
    assert n_writes(path) == 0
    assert raw(path).execute("SELECT COUNT(*) FROM parents").fetchone()[0] == 0


def test_same_item_twice_is_one_edge(path, cap):
    cap.on_retrieve("k", [("mem", "a"), ("mem", "a")])
    cap.on_retrieve("k", [("mem", "a")])
    wid = write(cap)
    count = raw(path).execute("SELECT COUNT(*) FROM parents WHERE write_id=?", (wid,)).fetchone()[0]
    assert count == 1


def test_reset_between_retrieval_and_write(path, cap):
    cap.on_retrieve("k", [("mem", "a")])
    cap.reset("k")
    assert parents(path, write(cap)) == {}


def test_reset_unknown_key_is_noop(cap):
    cap.reset("nope")


def test_cross_store_parents(path, cap):
    cap.on_retrieve("k", [("mongo", "1")])
    cap.on_retrieve("k", [("pgvector", "1")])
    assert set(parents(path, write(cap, store="mongo"))) == {("mongo", "1"), ("pgvector", "1")}


def test_bind_uses_write_store(path, cap):
    wid = write(cap, store="pg")
    cap.bind(wid, "row-7")
    row = raw(path).execute("SELECT store, record_id FROM bindings WHERE write_id=?", (wid,)).fetchone()
    assert row == ("pg", "row-7")


def test_bind_unknown_write_raises(cap):
    with pytest.raises(KeyError):
        cap.bind("missing", "r")


def test_compaction_parents_and_trace(path, cap):
    # A, B, C are written and bound to their store IDs.
    for rid in ("A", "B", "C"):
        cap.bind(write(cap, key="setup", content=rid), rid)

    cap.on_retrieve("k", [("mem", "ctx")])
    s = cap.on_compaction("k", [("mem", "A"), ("mem", "B"), ("mem", "C")], "summary", "mem", "agent_inference")
    assert parents(path, s) == {
        ("mem", "A"): "in_prompt",
        ("mem", "B"): "in_prompt",
        ("mem", "C"): "in_prompt",
        ("mem", "ctx"): "retrieved_in_scope",
    }
    cap.bind(s, "S")

    # A later write derived from S traces back through S to A, B, C.
    cap.on_retrieve("later", [("mem", "S")])
    w = write(cap, key="later")
    conn = raw(path)
    hop1 = conn.execute(
        "SELECT b.write_id FROM parents p JOIN bindings b"
        " ON b.store = p.parent_store AND b.record_id = p.parent_record_id"
        " WHERE p.write_id = ?",
        (w,),
    ).fetchall()
    assert hop1 == [(s,)]
    assert {("mem", "A"), ("mem", "B"), ("mem", "C")} <= set(parents(path, s))


def test_compaction_replaced_overrides_context_tag(path, cap):
    cap.on_retrieve("k", [("mem", "A")])
    s = cap.on_compaction("k", [("mem", "A")], "summary", "mem", "agent_inference")
    assert parents(path, s) == {("mem", "A"): "in_prompt"}


def test_concurrent_keys_never_mix(path, cap):
    n_threads, rounds = 8, 50
    results = {}
    errors = []
    barrier = threading.Barrier(n_threads)

    def worker(i):
        try:
            key = f"k{i}"
            barrier.wait()
            for r in range(rounds):
                cap.reset(key)
                cap.on_retrieve(key, [(f"s{i}", f"{i}-{r}-a"), (f"s{i}", f"{i}-{r}-b")])
                wid = write(cap, key=key)
                results[wid] = {(f"s{i}", f"{i}-{r}-a"), (f"s{i}", f"{i}-{r}-b")}
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == n_threads * rounds
    for wid, expected in results.items():
        assert set(parents(path, wid)) == expected
