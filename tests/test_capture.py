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


class FailOnceLedger:
    def __init__(self, ledger):
        self._ledger = ledger
        self.failed = False

    def __getattr__(self, name):
        return getattr(self._ledger, name)

    def bind(self, *args):
        if not self.failed:
            self.failed = True
            raise sqlite3.OperationalError("database is locked")
        self._ledger.bind(*args)


def test_bind_retry_after_ledger_failure(path, ledger):
    cap = Capture(FailOnceLedger(ledger))
    wid = write(cap, store="pg")
    with pytest.raises(sqlite3.OperationalError):
        cap.bind(wid, "row-7")
    cap.bind(wid, "row-7")
    rows = raw(path).execute("SELECT store, record_id FROM bindings WHERE write_id=?", (wid,)).fetchall()
    assert rows == [("pg", "row-7")]
    # Once bound, the write is no longer pending.
    with pytest.raises(KeyError):
        cap.bind(wid, "row-7")


class BlockingLedger:
    def __init__(self, ledger):
        self._ledger = ledger
        self.entered = threading.Event()
        self.release = threading.Event()

    def __getattr__(self, name):
        return getattr(self._ledger, name)

    def bind(self, *args):
        self.entered.set()
        assert self.release.wait(5)
        self._ledger.bind(*args)


def test_overlapping_binds_same_write_bind_once(path, ledger):
    led = BlockingLedger(ledger)
    cap = Capture(led)
    wid = write(cap, store="pg")
    first = threading.Thread(target=cap.bind, args=(wid, "row-1"))
    first.start()
    assert led.entered.wait(5)
    # The first bind is in flight in the ledger; a second one must not race it.
    with pytest.raises(KeyError):
        cap.bind(wid, "row-2")
    led.release.set()
    first.join(5)
    rows = raw(path).execute("SELECT store, record_id FROM bindings WHERE write_id=?", (wid,)).fetchall()
    assert rows == [("pg", "row-1")]
    with pytest.raises(KeyError):
        cap.bind(wid, "row-1")


def test_pending_evicts_oldest_past_cap(path, ledger):
    cap = Capture(ledger, max_pending=2)
    w1, w2, w3 = (write(cap, content=c) for c in "123")
    with pytest.raises(KeyError):
        cap.bind(w1, "r1")
    cap.bind(w2, "r2")
    cap.bind(w3, "r3")
    # The evicted write stays in the ledger, unbound.
    assert n_writes(path) == 3
    bound = {w for (w,) in raw(path).execute("SELECT write_id FROM bindings")}
    assert bound == {w2, w3}


def test_bound_writes_free_pending_slots(ledger):
    cap = Capture(ledger, max_pending=1)
    for i in range(5):
        cap.bind(write(cap, content=str(i)), f"r{i}")


def test_max_pending_must_be_positive(ledger):
    with pytest.raises(ValueError):
        Capture(ledger, max_pending=0)


def tombstones(path):
    return raw(path).execute(
        "SELECT store, record_id, reason, write_id, agent_id, session_id"
        " FROM tombstones ORDER BY seq"
    ).fetchall()


def test_on_delete_records_tombstones(path, cap):
    cap.on_retrieve("task", [("mem", "ctx")])
    wid = cap.on_write("task", "x", "mem", "agent_inference")
    cap.on_delete("mem", ["r1", "r2"], "consolidation", wid, "a1", "s1")
    cap.on_delete("pg", ["r3"], "forget")
    assert tombstones(path) == [
        ("mem", "r1", "consolidation", wid, "a1", "s1"),
        ("mem", "r2", "consolidation", wid, "a1", "s1"),
        ("pg", "r3", "forget", None, None, None),
    ]
    assert n_writes(path) == 1  # a deletion is not a write
    assert cap._context["task"] == {("mem", "ctx"): "retrieved_in_scope"}


def test_on_delete_bad_reason_records_nothing(path, cap):
    with pytest.raises(ValueError):
        cap.on_delete("mem", ["r1", "r2"], "rumor")
    assert tombstones(path) == []


def test_on_batch_records_deletes_with_writes(path, cap):
    wid = cap.on_write("task", "x", "mem", "agent_inference")
    (cid,) = cap.on_batch(
        "task", "mem", [([("mem", "r1")], "merged", "agent_inference", "r0")],
        [(wid, "r0")], "a1", "s1", deletes=[("r1", "consolidation", wid)],
    )
    assert tombstones(path) == [("mem", "r1", "consolidation", wid, "a1", "s1")]
    assert parents(path, cid) == {("mem", "r1"): "in_prompt"}


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


def bindings(path):
    return raw(path).execute("SELECT write_id, store, record_id FROM bindings ORDER BY rowid").fetchall()


def test_on_batch_records_writes_compactions_and_binds(path, cap):
    earlier = write(cap, store="pg")
    cap.on_retrieve("k", [("mem", "ctx")])
    w_plain, w_merge, w_unbound = cap.on_batch(
        "k", "pg",
        [((), "a", "user", "rec-a"), ([("pg", "old")], "b", "agent_inference", "rec-b"),
         ((), "c", "tool", None)],
        binds=[(earlier, "rec-0")], agent_id="a1", session_id="s1",
    )
    assert parents(path, w_plain) == {("mem", "ctx"): "retrieved_in_scope"}
    assert parents(path, w_merge) == {
        ("pg", "old"): "in_prompt", ("mem", "ctx"): "retrieved_in_scope",
    }
    assert bindings(path) == [
        (w_plain, "pg", "rec-a"), (w_merge, "pg", "rec-b"), (earlier, "pg", "rec-0"),
    ]
    rows = raw(path).execute(
        "SELECT source, agent_id, session_id FROM writes ORDER BY seq"
    ).fetchall()[1:]
    assert rows == [("user", "a1", "s1"), ("agent_inference", "a1", "s1"), ("tool", "a1", "s1")]
    # Bound writes are no longer pending; the unbound one is.
    with pytest.raises(KeyError):
        cap.bind(earlier, "again")
    cap.bind(w_unbound, "rec-c")


def test_binds_carry_an_optional_content_hash(path, ledger):
    from dod.capture import BindEvent

    events = []

    class Rec:
        def on_bind(self, e):
            events.append(e)

    cap = Capture(ledger, observers=[Rec()])
    a, b, c = (write(cap) for _ in range(3))
    cap.bind(a, "ra", content_hash="ha")
    cap.on_batch("k", "mem", binds=[(b, "rb", "hb"), (c, "rc")])
    assert raw(path).execute(
        "SELECT write_id, record_id, content_hash FROM bindings ORDER BY rowid").fetchall() == [
        (a, "ra", "ha"), (b, "rb", "hb"), (c, "rc", None)]
    assert events == [BindEvent(a, "mem", "ra", "ha"), BindEvent(b, "mem", "rb", "hb"),
                      BindEvent(c, "mem", "rc", None)]


def test_on_batch_unknown_or_duplicate_bind_records_nothing(path, cap):
    wid = write(cap)
    for binds in ([("missing", "r")], [(wid, "r1"), (wid, "r2")]):
        with pytest.raises(KeyError):
            cap.on_batch("k", "mem", [((), "x", "user", None)], binds)
    assert n_writes(path) == 1
    cap.bind(wid, "r")  # still pending


class FailBatchLedger:
    def __init__(self, ledger):
        self._ledger = ledger

    def __getattr__(self, name):
        return getattr(self._ledger, name)

    def append_batch(self, *args):
        raise sqlite3.OperationalError("database is locked")


def test_on_batch_ledger_failure_keeps_binds_pending(path, ledger):
    cap = Capture(FailBatchLedger(ledger))
    wid = write(cap)
    with pytest.raises(sqlite3.OperationalError):
        cap.on_batch("k", "mem", [((), "x", "user", None)], [(wid, "r")])
    cap.bind(wid, "r")
    assert bindings(path) == [(wid, "mem", "r")]


def test_snapshot_keeps_context_after_reset(path, cap):
    cap.on_retrieve("k", [("mem", "a")], "in_prompt")
    snap = cap.snapshot("k")
    cap.reset("k")
    cap.on_retrieve("k", [("mem", "later")])
    wid = write(cap, key=snap)
    assert parents(path, wid) == {("mem", "a"): "in_prompt"}
    cap.reset(snap)
    assert snap not in cap._context


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
