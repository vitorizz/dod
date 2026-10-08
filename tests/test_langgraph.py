import asyncio
import hashlib
import json
import sqlite3
import threading

import pytest

pytest.importorskip("langgraph.store.memory")
pytestmark = pytest.mark.langgraph

from langgraph.store.base import BaseStore, GetOp, PutOp  # noqa: E402
from langgraph.store.base.batch import AsyncBatchedBaseStore  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.integrations.langgraph import instrument, record_id  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402

NS = ("users", "u1")


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


@pytest.fixture
def store(cap):
    return instrument(InMemoryStore(), cap)


def raw(path):
    return sqlite3.connect(str(path), isolation_level=None)


def writes(path):
    return raw(path).execute(
        "SELECT write_id, store, source, agent_id, session_id, content_hash"
        " FROM writes ORDER BY seq"
    ).fetchall()


def parents(path, write_id):
    return dict(raw(path).execute(
        "SELECT parent_record_id, tag FROM parents WHERE write_id=?", (write_id,)
    ))


def bound(path, write_id):
    return [r for (r,) in raw(path).execute(
        "SELECT record_id FROM bindings WHERE write_id=?", (write_id,)
    )]


def rid(key, ns=NS):
    return record_id(ns, key)


# --- Wrapping ----------------------------------------------------------------


def test_record_id_is_json_and_reversible():
    r = record_id(("a", 'b"/c'), "k,1")
    assert r == json.dumps(["a", 'b"/c', "k,1"])
    *ns, key = json.loads(r)
    assert (tuple(ns), key) == (("a", 'b"/c'), "k,1")
    # A separator inside a part can't collide with a deeper namespace.
    assert record_id(("a/b",), "c") != record_id(("a",), "b/c")


def test_return_values_unchanged_and_idempotent(cap):
    plain = InMemoryStore()
    s = instrument(plain, cap)
    assert s is plain and instrument(s, cap) is s
    with agent_task_scope("agent", "task"):
        s.put(NS, "k", {"text": "hi"})
        item = s.get(NS, "k")
        found = s.search(("users",))
        assert s.list_namespaces() == [NS]
    assert item.value == {"text": "hi"} and item.key == "k"
    assert [i.key for i in found] == ["k"]


def test_rejects_stores_that_bypass_batch(cap):
    class Bypass(InMemoryStore):
        def put(self, namespace, key, value, index=None, **kw):
            self._data[namespace][key] = value

    with pytest.raises(TypeError, match="overrides put"):
        instrument(Bypass(), cap)
    with pytest.raises(TypeError, match="BaseStore"):
        instrument(object(), cap)
    # AsyncBatchedBaseStore runs abatch in a background task's context.
    assert any(
        getattr(AsyncBatchedBaseStore, n) is not getattr(BaseStore, n)
        for n in ("get", "put", "aget", "aput")
    )


# --- Reads -------------------------------------------------------------------


def test_put_parents_are_items_retrieved_in_scope(path, store):
    store.put(NS, "a", {"text": "A"})  # outside scope: no parents
    store.put(NS, "b", {"text": "B"})
    with agent_task_scope("agent", "task"):
        store.get(NS, "a")
        store.search(("users",), query=None, limit=10)
        store.get(NS, "missing")
        store.put(NS, "c", {"text": "C"})
    w = writes(path)[-1]
    assert w[1:5] == ("langgraph", "agent_inference", "agent", "task")
    assert parents(path, w[0]) == {
        rid("a"): "retrieved_in_scope",
        rid("b"): "retrieved_in_scope",
    }
    assert bound(path, w[0]) == [rid("c")]


def test_reads_outside_scope_are_not_tracked(path, store, caplog):
    store.put(NS, "a", {"text": "A"})
    store.get(NS, "a")
    with agent_task_scope("agent", "task"):
        store.put(NS, "b", {"text": "B"})
    assert parents(path, writes(path)[-1][0]) == {}
    assert "outside agent_task_scope" in caplog.text


def test_reads_in_same_batch_as_put_are_not_its_parents(path, store):
    store.put(NS, "a", {"text": "A"})
    with agent_task_scope("agent", "task"):
        store.batch([GetOp(NS, "a"), PutOp(NS, "b", {"text": "B"})])
        assert parents(path, writes(path)[-1][0]) == {}
        store.put(NS, "c", {"text": "C"})
    assert parents(path, writes(path)[-1][0]) == {rid("a"): "retrieved_in_scope"}


# --- Writes ------------------------------------------------------------------


def test_put_content_hash_is_sorted_json(path, store):
    store.put(NS, "a", {"b": 1, "a": [1, 2]})
    expected = hashlib.sha256(b'{"a":[1,2],"b":1}').hexdigest()
    assert writes(path)[-1][5] == expected


class FlakyStore(InMemoryStore):
    """Raises on any batch that writes; records every batch it is given."""

    __slots__ = ("calls",)

    def __init__(self):
        super().__init__()
        self.calls = []

    def batch(self, ops):
        ops = list(ops)
        self.calls.append(ops)
        if any(isinstance(o, PutOp) for o in ops):
            raise RuntimeError("store down")
        return super().batch(ops)

    async def abatch(self, ops):
        ops = list(ops)
        if any(isinstance(o, PutOp) for o in ops):
            raise RuntimeError("store down")
        return await super().abatch(ops)


def test_failed_put_leaves_write_unbound(path, cap):
    s = instrument(FlakyStore(), cap)
    assert isinstance(s, FlakyStore)
    with agent_task_scope("agent", "task"), pytest.raises(RuntimeError):
        s.put(NS, "a", {"text": "A"})
    (w,) = writes(path)
    assert bound(path, w[0]) == []
    # The pre-read doesn't refresh TTLs.
    assert s.calls[0] == [GetOp(NS, "a", refresh_ttl=False)]


def test_overwrite_is_compaction_of_overwritten_item(path, ledger, store):
    with agent_task_scope("agent", "task"):
        store.put(NS, "a", {"text": "v1"})
        store.put(NS, "b", {"text": "B"})
        store.get(NS, "b")
        store.put(NS, "a", {"text": "v2"})
    w1, _, w2 = [w[0] for w in writes(path)]
    assert parents(path, w2) == {rid("a"): "in_prompt", rid("b"): "retrieved_in_scope"}
    assert bound(path, w2) == [rid("a")]
    t = trace(ledger, "langgraph", rid("a"))
    assert {w.write_id for w in t.items} == {w1, w2}


def tombstones(path):
    return raw(path).execute(
        "SELECT store, record_id, reason, write_id, agent_id, session_id"
        " FROM tombstones ORDER BY seq"
    ).fetchall()


def test_delete_records_a_tombstone_not_a_write(path, store):
    with agent_task_scope("agent", "task"):
        store.put(NS, "a", {"text": "A"})
        store.delete(NS, "a")
        store.delete(NS, "never-existed")  # deletes nothing: not recorded
    assert len(writes(path)) == 1
    assert tombstones(path) == [("langgraph", rid("a"), "forget", None, "agent", "task")]
    assert store.get(NS, "a") is None


def test_delete_outside_scope_records_tombstone_without_warning(path, store, caplog):
    store.put(NS, "a", {"text": "A"})
    caplog.clear()
    store.delete(NS, "a")
    assert tombstones(path) == [("langgraph", rid("a"), "forget", None, None, None)]
    assert "outside agent_task_scope" not in caplog.text


class FailingDeleteStore(InMemoryStore):
    __slots__ = ()

    def batch(self, ops):
        ops = list(ops)
        if any(isinstance(o, PutOp) and o.value is None for o in ops):
            raise RuntimeError("store down")
        return super().batch(ops)


def test_failed_delete_records_no_tombstone(path, cap):
    s = instrument(FailingDeleteStore(), cap)
    s.put(NS, "a", {"text": "A"})
    with pytest.raises(RuntimeError):
        s.delete(NS, "a")
    assert tombstones(path) == []


def test_batch_with_duplicate_key_records_the_last_put(path, store):
    with agent_task_scope("agent", "task"):
        store.batch([PutOp(NS, "a", {"v": 1}), PutOp(NS, "a", {"v": 2})])
    (w,) = writes(path)
    assert w[5] == hashlib.sha256(b'{"v":2}').hexdigest()
    assert bound(path, w[0]) == [rid("a")]


# --- Async -------------------------------------------------------------------


def test_async_variants_are_captured_identically(path, store):
    async def run():
        with agent_task_scope("agent", "task"):
            await store.aput(NS, "a", {"text": "A"})
            await store.aget(NS, "a")
            await store.asearch(("users",))
            await store.aput(NS, "b", {"text": "B"})
            await store.aput(NS, "a", {"text": "A2"})
            await store.adelete(NS, "b")

    asyncio.run(run())
    a, b, a2 = [w[0] for w in writes(path)]
    assert parents(path, b) == {rid("a"): "retrieved_in_scope"}
    assert bound(path, b) == [rid("b")]
    assert parents(path, a2) == {rid("a"): "in_prompt"}
    assert bound(path, a2) == [rid("a")]
    assert tombstones(path) == [("langgraph", rid("b"), "forget", None, "agent", "task")]


def test_failed_async_put_leaves_write_unbound(path, cap):
    s = instrument(FlakyStore(), cap)

    async def run():
        with agent_task_scope("agent", "task"):
            await s.aput(NS, "a", {"text": "A"})

    with pytest.raises(RuntimeError):
        asyncio.run(run())
    (w,) = writes(path)
    assert bound(path, w[0]) == []


# --- Concurrency -------------------------------------------------------------


def test_concurrent_scopes_never_cross_link(path, store):
    for i in range(4):
        store.put(("src",), f"s{i}", {"text": f"seed {i}"})
    barrier = threading.Barrier(2)

    def thread_work(i):
        with agent_task_scope(f"agent-t{i}", f"task-t{i}"):
            store.get(("src",), f"s{i}")
            barrier.wait()
            store.put(("out",), f"t{i}", {"text": "derived"})

    threads = [threading.Thread(target=thread_work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    async def coro_work(i):
        with agent_task_scope(f"agent-c{i}", f"task-c{i}"):
            await store.aget(("src",), f"s{i}")
            await asyncio.sleep(0)  # interleave with the other coroutine
            await store.aput(("out",), f"c{i}", {"text": "derived"})

    async def both():
        await asyncio.gather(coro_work(2), coro_work(3))

    asyncio.run(both())

    by_record = {
        r: wid for wid, r in raw(path).execute("SELECT write_id, record_id FROM bindings")
    }
    for name, i in (("t0", 0), ("t1", 1), ("c2", 2), ("c3", 3)):
        wid = by_record[rid(name, ("out",))]
        assert parents(path, wid) == {rid(f"s{i}", ("src",)): "retrieved_in_scope"}


# --- Queries -----------------------------------------------------------------


def test_trace_and_blast_radius_within_store(path, ledger, store):
    with agent_task_scope("agent", "task"):
        store.put(NS, "root", {"text": "false fact"})
    with agent_task_scope("agent", "task2"):
        store.get(NS, "root")
        store.put(NS, "child", {"text": "derived"})
    t = trace(ledger, "langgraph", rid("child"))
    assert [r.write_id for r in t.roots] == [writes(path)[0][0]]
    b = blast_radius(ledger, "langgraph", rid("root"))
    assert b.store_counts == {"langgraph": 1}


def test_search_items_map_to_put_record_ids(path, store):
    with agent_task_scope("agent", "task"):
        store.put(("users", "u1"), "k", {"text": "A"})
        (item,) = store.search(("users",))
    assert record_id(item.namespace, item.key) == bound(path, writes(path)[0][0])[0]


# --- Quarantine --------------------------------------------------------------


def test_quarantined_items_are_not_returned_or_recorded_as_parents(path, ledger, store):
    from dod.quarantine import _blocked, blocked_retrievals

    _blocked.clear()
    store.put(NS, "good", {"text": "G"})
    store.put(NS, "bad", {"text": "B"})
    ledger.quarantine([("langgraph", rid("bad"))])
    with agent_task_scope("agent", "task"):
        assert store.get(NS, "bad") is None
        assert [i.key for i in store.search(NS)] == ["good"]
        store.put(NS, "c", {"text": "C"})
    assert parents(path, writes(path)[-1][0]) == {rid("good"): "retrieved_in_scope"}
    assert blocked_retrievals() == {("langgraph", rid("bad")): 2}

    # Overwriting a quarantined key is still seen as a compaction.
    with agent_task_scope("agent", "task2"):
        store.put(NS, "bad", {"text": "B2"})
    assert parents(path, writes(path)[-1][0]) == {rid("bad"): "in_prompt"}

    ledger.release([("langgraph", rid("bad"))])
    assert store.get(NS, "bad").value == {"text": "B2"}

    async def run():
        ledger.quarantine([("langgraph", rid("good"))])
        assert await store.aget(NS, "good") is None
        assert [i.key for i in await store.asearch(NS)] == ["bad", "c"]

    asyncio.run(run())
    _blocked.clear()
