import asyncio
import contextvars
import hashlib
import inspect
import json
import logging
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from dod.capture import Capture
from dod.integrations import crewai as integ
from dod.integrations.crewai import agent_task_scope, instrument
from dod.ledger import SOURCES, Ledger

# Keep crewai (when installed) from phoning home during tests.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")


# --- Stub of crewai.Memory 1.15.22 -------------------------------------------
# recall / remember / remember_many mirror the real signatures (checked by
# test_stub_signatures_match_crewai). remember mirrors the real save path: it
# runs on a one-worker pool through contextvars.copy_context().run, searches
# storage for similar records, then applies a consolidation plan with
# storage.delete / update / save. remember_many submits
# _background_encode_batch to the same pool and returns [] without waiting.


class StubRecord:
    def __init__(self, content, id=None, scope="/"):
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.scope = scope


class StubMatch:
    def __init__(self, record):
        self.record = record


class StubStorage:
    def __init__(self):
        self.records = {}
        self.similar = []  # record IDs the next search returns

    def search(self, query_embedding, scope_prefix=None, categories=None,
               metadata_filter=None, limit=10, min_score=0.0):
        return [(self.records[rid], 0.9) for rid in self.similar][:limit]

    def save(self, records):
        for r in records:
            self.records[r.id] = r

    def update(self, record):
        self.records[record.id] = record

    def delete(self, scope_prefix=None, categories=None, record_ids=None,
               older_than=None, metadata_filter=None):
        # Like crewai's backends: IDs alone delete exactly those (scope is
        # ignored); otherwise filter by scope (other filters aren't modeled).
        if record_ids and not (categories or older_than or metadata_filter):
            doomed = [r for r in dict.fromkeys(record_ids) if r in self.records]
        else:
            doomed = [rid for rid, r in self.records.items()
                      if r.scope.startswith(scope_prefix or "/")]
        for rid in doomed:
            del self.records[rid]
        return len(doomed)

    def reset(self, scope_prefix=None):
        self.delete(scope_prefix=scope_prefix or "/")

    def get_record(self, record_id):
        return self.records.get(record_id)

    def count(self, scope_prefix=None):
        return len(self.records)

    def list_records(self, scope_prefix=None, limit=200, offset=0):
        records = [r for r in self.records.values() if r.scope.startswith(scope_prefix or "/")]
        return records[offset:offset + limit]


class StubMemory:
    def __init__(self, read_only=False, pool=True):
        self.read_only = read_only
        self._storage = StubStorage()
        self._save_pool = ThreadPoolExecutor(max_workers=1) if pool else None
        self._pending = []  # remember_many futures, waited on by drain_writes
        self.results = {}  # query -> [StubRecord]
        # Consolidation plan for the next save: {"update": {id: content},
        # "delete": [id], "insert": bool}. None inserts a new record.
        self.plan = None
        self.fail = None

    def add(self, *ids):
        records = [StubRecord(f"content of {i}", id=i) for i in ids]
        self._storage.save(records)
        return records

    def recall(self, query, scope=None, categories=None, limit=10,
               depth="deep", source=None, include_private=False):
        return [StubMatch(r) for r in self.results.get(query, [])][:limit]

    def remember(self, content, scope=None, categories=None, metadata=None,
                 importance=None, source=None, private=False, agent_role=None,
                 root_scope=None):
        if self.read_only:
            return None
        if self._save_pool is None:
            return self._encode(content)
        ctx = contextvars.copy_context()
        return self._save_pool.submit(ctx.run, self._encode, content).result()

    def remember_many(self, contents, scope=None, categories=None,
                      metadata=None, importance=None, source=None,
                      private=False, agent_role=None, root_scope=None):
        if not contents or self.read_only:
            return []
        if self._save_pool is None:
            self._background_encode_batch(contents)
            return []
        ctx = contextvars.copy_context()
        self._pending.append(
            self._save_pool.submit(ctx.run, self._background_encode_batch, contents)
        )
        return []

    def _background_encode_batch(self, contents):
        return self._encode_many(contents)

    def update(self, record_id, content=None, scope=None, categories=None,
               metadata=None, importance=None):
        existing = self._storage.get_record(record_id)
        if existing is None:
            raise ValueError(f"Record not found: {record_id}")
        if self.read_only:
            return existing
        rec = StubRecord(existing.content if content is None else content, id=record_id)
        self._storage.update(rec)
        return rec

    def forget(self, scope=None, categories=None, older_than=None,
               metadata_filter=None, record_ids=None):
        return self._storage.delete(scope_prefix=scope, categories=categories,
                                    record_ids=record_ids, older_than=older_than,
                                    metadata_filter=metadata_filter)

    def reset(self, scope=None):
        self.drain_writes()
        self._storage.reset(scope_prefix=scope)

    def reset_all(self):
        self.drain_writes()
        self._storage.reset(scope_prefix=None)

    def drain_writes(self):
        pending, self._pending = self._pending, []
        for f in pending:
            f.exception()

    def _encode(self, content):
        records = self._encode_many([content])
        return records[0] if records else None

    def _encode_many(self, contents):
        similar = [r for r, _ in self._storage.search([1.0], limit=5)]
        plan, self.plan = self.plan or {"insert": True}, None
        if plan.get("delete"):
            self._storage.delete(record_ids=list(plan["delete"]))
        updated = []
        for rid, new_content in plan.get("update", {}).items():
            rec = StubRecord(new_content, id=rid)
            self._storage.update(rec)
            updated.append(rec)
        if self.fail:
            raise self.fail
        if plan.get("insert"):
            records = [StubRecord(c) for c in contents]
            self._storage.save(records)
            return records
        return updated[:1] or similar[:1]


# --- Fixtures and helpers ----------------------------------------------------


@pytest.fixture
def path(tmp_path):
    return tmp_path / "ledger.db"


@pytest.fixture
def cap(path):
    with Ledger(path) as ledger:
        yield Capture(ledger)


@pytest.fixture
def mem(cap):
    return instrument(StubMemory(), cap)


def raw(path):
    return sqlite3.connect(str(path), isolation_level=None)


def writes(path):
    return raw(path).execute(
        "SELECT write_id, store, source, agent_id, session_id FROM writes ORDER BY seq"
    ).fetchall()


def parents(path, write_id):
    rows = raw(path).execute(
        "SELECT parent_record_id, tag FROM parents WHERE write_id=?", (write_id,)
    )
    return dict(rows)


def bound(path, write_id):
    return [r for (r,) in raw(path).execute(
        "SELECT record_id FROM bindings WHERE write_id=?", (write_id,)
    )]


def write_for(path, record_id):
    (wid,) = raw(path).execute(
        "SELECT write_id FROM bindings WHERE record_id=?", (record_id,)
    ).fetchone()
    return wid


def last_write(path):
    return writes(path)[-1][0]


def tombstones(path):
    return raw(path).execute(
        "SELECT record_id, reason, write_id, agent_id, session_id FROM tombstones ORDER BY seq"
    ).fetchall()


# --- Wrapping ----------------------------------------------------------------


def test_signatures_and_return_values_unchanged(cap):
    plain = StubMemory()
    before = {n: inspect.signature(getattr(plain, n)) for n in ("recall", "remember", "remember_many")}
    a, b = plain.add("A", "B")
    plain.results["q"] = [a, b]
    m = instrument(plain, cap)
    for name, sig in before.items():
        assert inspect.signature(getattr(m, name)) == sig
    with agent_task_scope("agent", "task"):
        res = m.recall("q", limit=1)
        rec = m.remember("C")
    assert [x.record for x in res] == [a]
    assert isinstance(rec, StubRecord) and rec.content == "C"
    assert m._storage.records[rec.id] is rec


def test_instrument_twice_is_noop(path, cap):
    m = StubMemory()
    assert instrument(m, cap) is m
    wrapped = (m.recall, m.remember, m.remember_many, m._storage.save)
    assert instrument(m, cap) is m
    assert instrument(m, Capture(cap._ledger)) is m
    assert (m.recall, m.remember, m.remember_many, m._storage.save) == wrapped
    with agent_task_scope("agent", "task"):
        m.remember("C")
    assert len(writes(path)) == 1


def test_remember_records_before_save_then_binds(path, mem):
    seen = []
    orig = mem._storage.save
    mem._storage.save = lambda records: (seen.append(len(writes(path))), orig(records))
    with agent_task_scope("agent", "task"):
        rec = mem.remember("C")
    assert seen == [1]  # the write was in the ledger before the store saved
    ((wid, store, source, agent_id, session_id),) = writes(path)
    assert (store, source, agent_id, session_id) == ("crewai", "agent_inference", "agent", "task")
    assert bound(path, wid) == [rec.id]


def test_read_only_remember_records_unbound_write(path, cap):
    m = instrument(StubMemory(read_only=True), cap)
    with agent_task_scope("agent", "task"):
        assert m.remember("C") is None
    ((wid, *_),) = writes(path)
    assert bound(path, wid) == []


def test_failed_remember_records_unbound_write_and_propagates(path, mem):
    err = RuntimeError("storage down")
    mem.fail = err
    with agent_task_scope("agent", "task"):
        with pytest.raises(RuntimeError) as info:
            mem.remember("C")
    assert info.value is err
    ((wid, *_),) = writes(path)
    assert bound(path, wid) == []


def test_recall_exception_propagates(cap):
    err = ValueError("bad query")

    def boom(*a, **k):
        raise err

    m = StubMemory()
    m.recall = boom
    instrument(m, cap)
    with agent_task_scope("agent", "task"), pytest.raises(ValueError) as info:
        m.recall("q")
    assert info.value is err


def test_store_name_used_for_parents_and_bindings(path, cap):
    m = instrument(StubMemory(), cap, store="crew-mem")
    (a,) = m.add("A")
    m.results["q"] = [a]
    with agent_task_scope("agent", "task"):
        m.recall("q")
        rec = m.remember("C")
    rows = raw(path).execute("SELECT parent_store FROM parents").fetchall()
    assert rows == [("crew-mem",)]
    assert raw(path).execute("SELECT store, record_id FROM bindings").fetchall() == [("crew-mem", rec.id)]


@pytest.mark.parametrize("value", SOURCES)
def test_source_kept_when_ledger_value(path, mem, value):
    with agent_task_scope("agent", "task"):
        mem.remember("C", source=value)
    assert writes(path)[0][2] == value


@pytest.mark.parametrize("value", [None, "user-123", "session-9", "USER"])
def test_source_otherwise_agent_inference(path, mem, value):
    with agent_task_scope("agent", "task"):
        mem.remember("C", "/", None, None, None, value)  # positional source
    assert writes(path)[0][2] == "agent_inference"


# --- remember_many -----------------------------------------------------------


def _hold_save_pool(mem):
    """Block the stub's save pool until the returned event is set."""
    release = threading.Event()
    mem._save_pool.submit(release.wait, 5)
    return release


def test_remember_many_records_at_call_and_binds_after_save(path, cap, mem):
    (a,) = mem.add("A")
    mem.results["q"] = [a]
    release = _hold_save_pool(mem)
    with agent_task_scope("agent", "task"):
        mem.recall("q")
        assert mem.remember_many(["C", "D"], source="user") == []
    # Recorded in the caller's scope before the background save ran.
    (w_c, *_), (w_d, *_) = rows = writes(path)
    assert [r[2:] for r in rows] == [("user", "agent", "task")] * 2
    for wid in (w_c, w_d):
        assert parents(path, wid) == {"A": "retrieved_in_scope"}
        assert bound(path, wid) == []
    release.set()
    mem.drain_writes()
    by_content = {r.content: r.id for r in mem._storage.records.values()}
    assert bound(path, w_c) == [by_content["C"]]
    assert bound(path, w_d) == [by_content["D"]]
    assert cap._context == {}  # the batch's context snapshot was released


def test_remember_many_consolidation_keeps_callers_context(path, cap, mem):
    mem.add("A", "K")
    mem.results["k"] = [mem._storage.records["K"]]
    mem._storage.similar = ["A"]
    mem.plan = {"update": {"A": "A merged with C"}, "insert": True}
    release = _hold_save_pool(mem)
    with agent_task_scope("agent", "task"):
        mem.recall("k")
        mem.remember_many(["C", "D"])
    release.set()  # the scope ended (and was reset) before the save runs
    mem.drain_writes()
    w_c, w_d, *compactions = (w for w, *_ in writes(path))
    assert len(compactions) == 3  # A rewritten, C and D inserted
    for wid in compactions:
        assert parents(path, wid) == {"A": "in_prompt", "K": "retrieved_in_scope"}
    assert bound(path, compactions[0]) == ["A"]
    assert cap._context == {}


def test_remember_many_one_item_merged_binds_to_returned_record(path, mem):
    mem.add("A")
    mem._storage.similar = ["A"]
    mem.plan = {"update": {"A": "A merged with C"}, "insert": False}
    with agent_task_scope("agent", "task"):
        mem.remember_many(["C"])
    mem.drain_writes()
    (w_c, *_), (w_merge, *_) = writes(path)
    assert bound(path, w_c) == ["A"] and bound(path, w_merge) == ["A"]


def test_remember_many_read_only_records_unbound_writes(path, cap):
    m = instrument(StubMemory(read_only=True), cap)
    with agent_task_scope("agent", "task"):
        assert m.remember_many(["C", "D"]) == []
    assert [bound(path, w) for w, *_ in writes(path)] == [[], []]
    assert cap._context == {}


def test_remember_many_empty_records_nothing(path, mem):
    with agent_task_scope("agent", "task"):
        assert mem.remember_many([]) == []
    assert writes(path) == []


def test_remember_many_outside_scope_warns_and_records_no_parents(path, mem, caplog):
    (a,) = mem.add("A")
    mem.results["q"] = [a]
    with caplog.at_level(logging.WARNING, logger="dod.integrations.crewai"):
        mem.recall("q")
        mem.remember_many(["C"])
    mem.drain_writes()
    ((wid, *_rest, agent_id, session_id),) = writes(path)
    assert parents(path, wid) == {} and (agent_id, session_id) == (None, None)
    assert any("remember_many() outside agent_task_scope" in r.message for r in caplog.records)


def test_remember_many_failed_save_leaves_writes_unbound(path, cap, mem):
    mem.fail = RuntimeError("storage down")
    with agent_task_scope("agent", "task"):
        mem.remember_many(["C"])
    mem.drain_writes()
    ((wid, *_),) = writes(path)
    assert bound(path, wid) == []
    assert cap._context == {}


# --- Parents -----------------------------------------------------------------


def test_every_recall_since_scope_entry_is_a_parent(path, mem):
    a, b = mem.add("A", "B")
    mem.results = {"qa": [a], "qb": [b]}
    with agent_task_scope("agent", "task"):
        mem.recall("qa")
        mem.recall("qb")
        mem.remember("C")
    assert set(parents(path, last_write(path))) == {"A", "B"}


def test_end_to_end_two_tasks(path, mem):
    a, b = mem.add("A", "B")
    mem.results = {"both": [a, b], "only b": [b]}
    with agent_task_scope("agent", "task-1"):
        mem.recall("both")
        c = mem.remember("C")
    with agent_task_scope("agent", "task-2"):
        mem.recall("only b")
        d = mem.remember("D")
    wc, wd = write_for(path, c.id), write_for(path, d.id)
    assert parents(path, wc) == {"A": "retrieved_in_scope", "B": "retrieved_in_scope"}
    assert bound(path, wc) == [c.id]
    assert set(parents(path, wd)) == {"B"}
    assert bound(path, wd) == [d.id]


def test_scope_entry_resets_and_nested_scopes_restore(path, mem):
    a, b = mem.add("A", "B")
    mem.results = {"qa": [a], "qb": [b]}
    with agent_task_scope("agent", "task"):
        mem.recall("qa")
        with agent_task_scope("agent", "sub"):
            mem.recall("qb")
            mem.remember("inner")
            assert set(parents(path, last_write(path))) == {"B"}
        assert integ._key.get() == ("agent", "task")
        mem.remember("outer")
        assert set(parents(path, last_write(path))) == {"A"}
    assert integ._key.get() is None
    with agent_task_scope("agent", "task"):  # re-entry starts empty
        mem.remember("again")
    assert parents(path, last_write(path)) == {}


def test_recall_outside_scope_not_tracked(path, mem):
    (a,) = mem.add("A")
    mem.results["q"] = [a]
    mem.recall("q")
    with agent_task_scope("agent", "task"):
        mem.remember("C")
    assert parents(path, last_write(path)) == {}


def test_write_outside_scope_has_no_parents_and_warns_once(path, mem, caplog):
    (a,) = mem.add("A")
    mem.results["q"] = [a]
    with caplog.at_level(logging.WARNING, logger="dod.integrations.crewai"):
        mem.recall("q")
        mem.remember("C")
        mem.remember("D")
    for wid, *_rest, agent_id, session_id in writes(path):
        assert parents(path, wid) == {}
        assert (agent_id, session_id) == (None, None)
    assert len([r for r in caplog.records if "outside agent_task_scope" in r.message]) == 1


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_finished_scopes_hold_no_context(cap, mem):
    (a,) = mem.add("A")
    mem.results["q"] = [a]
    for i in range(3):
        with agent_task_scope("agent", f"task-{i}"):
            mem.recall("q")
            assert cap._context
    assert cap._context == {}
    # Same through the PRE_STEP / POST_STEP hooks, including nested tasks.
    agent = _Obj(id="agent")
    outer, inner = _Obj(id="outer"), _Obj(id="inner")
    integ._enter_task_scope(_Obj(kind="task", agent=agent, task=outer))
    mem.recall("q")
    integ._enter_task_scope(_Obj(kind="task", agent=agent, task=inner))
    mem.recall("q")
    integ._exit_task_scope(_Obj(kind="task", task=inner))
    assert integ._key.get() == ("agent", "outer")
    assert set(cap._context) == {("agent", "outer")}
    integ._exit_task_scope(_Obj(kind="task", task=outer))
    assert integ._key.get() is None
    assert cap._context == {}


# --- Concurrency -------------------------------------------------------------


def _run_agent(mem, agent_id, task_id, own, rounds, barrier, out):
    with agent_task_scope(agent_id, task_id):
        barrier.wait()
        for r in range(rounds):
            mem.recall(own)
            rec = mem.remember(f"{agent_id}/{task_id}/{r}")
            out.append((rec.id, own))


@pytest.mark.parametrize("agents", [("a1", "t1", "a2", "t2"), ("a1", "t1", "a1", "t2")],
                         ids=["two-agents", "same-agent-two-tasks"])
def test_threads_never_cross_link(path, mem, agents):
    a, b = mem.add("A", "B")
    mem.results = {"A": [a], "B": [b]}
    (ag1, t1, ag2, t2), out = agents, []
    barrier = threading.Barrier(2)
    threads = [
        threading.Thread(target=_run_agent, args=(mem, ag1, t1, "A", 50, barrier, out)),
        threading.Thread(target=_run_agent, args=(mem, ag2, t2, "B", 50, barrier, out)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 100
    for record_id, own in out:
        assert set(parents(path, write_for(path, record_id))) == {own}


def test_asyncio_tasks_never_cross_link(path, mem):
    a, b = mem.add("A", "B")
    mem.results = {"A": [a], "B": [b]}
    out = []

    async def agent(agent_id, own):
        with agent_task_scope(agent_id, "task"):
            for r in range(20):
                mem.recall(own)
                await asyncio.sleep(0)  # let the other agent run in between
                rec = mem.remember(f"{agent_id}/{r}")
                out.append((rec.id, own))
                await asyncio.sleep(0)

    async def main():
        await asyncio.gather(agent("a1", "A"), agent("a2", "B"))

    asyncio.run(main())
    assert len(out) == 40
    for record_id, own in out:
        assert set(parents(path, write_for(path, record_id))) == {own}


def test_executor_worker_without_scope_gets_no_parents(path, mem, caplog):
    (a,) = mem.add("A")
    mem.results["q"] = [a]
    with caplog.at_level(logging.WARNING, logger="dod.integrations.crewai"):
        with agent_task_scope("agent", "task"):
            mem.recall("q")
            with ThreadPoolExecutor(max_workers=1) as pool:
                rec = pool.submit(mem.remember, "C").result()
            # Through copy_context().run, the worker is in the parent's scope.
            ctx = contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                rec2 = pool.submit(ctx.run, mem.remember, "D").result()
    assert parents(path, write_for(path, rec.id)) == {}
    assert set(parents(path, write_for(path, rec2.id))) == {"A"}
    assert any("outside agent_task_scope" in r.message for r in caplog.records)


# --- Tags --------------------------------------------------------------------


def _injects_into_prompt(mem, q):
    return mem.recall(q)


class _ScopeView:
    """Like crewai's MemoryScope: forwards recall to the memory."""

    def __init__(self, mem):
        self.mem = mem

    def recall(self, q):
        return self.mem.recall(q)


def _injects_via_scope(view, q):
    return view.recall(q)


def test_tags_in_prompt_only_for_injecting_callers(path, mem, monkeypatch):
    monkeypatch.setattr(
        integ, "_in_prompt_callers",
        {_injects_into_prompt.__code__, _injects_via_scope.__code__},
    )
    a, b, c = mem.add("A", "B", "C")
    mem.results = {"a": [a], "b": [b], "c": [c]}
    with agent_task_scope("agent", "task"):
        _injects_into_prompt(mem, "a")
        _injects_via_scope(_ScopeView(mem), "b")
        mem.recall("c")
        mem.remember("D")
    assert parents(path, last_write(path)) == {
        "A": "in_prompt", "B": "in_prompt", "C": "retrieved_in_scope",
    }


# --- Compaction --------------------------------------------------------------


def test_consolidation_recorded_as_compaction(path, mem):
    a, b, x, ctx_item = mem.add("A", "B", "X", "K")
    mem.results["k"] = [ctx_item]
    mem._storage.similar = ["A", "B", "X"]  # all shown to the consolidation
    mem.plan = {"update": {"A": "A merged with C"}, "delete": ["B"], "insert": False}
    with agent_task_scope("agent", "task"):
        mem.recall("k")
        rec = mem.remember("C")
    assert rec.id == "A" and rec.content == "A merged with C"
    (w_c, *_), (w_merge, _, src, *_) = writes(path)
    # The write is recorded and bound to the returned record as usual.
    assert parents(path, w_c) == {"K": "retrieved_in_scope"}
    assert bound(path, w_c) == ["A"]
    # The rewrite is a compaction: every consulted/replaced record is an
    # in_prompt parent, plus the scope's context.
    assert parents(path, w_merge) == {
        "A": "in_prompt", "B": "in_prompt", "X": "in_prompt", "K": "retrieved_in_scope",
    }
    assert bound(path, w_merge) == ["A"]
    assert src == "agent_inference"
    # The deleted record gets a tombstone naming the write whose save caused it.
    assert tombstones(path) == [("B", "consolidation", w_c, "agent", "task")]


def test_delete_then_insert_is_compaction_of_new_record(path, mem):
    mem.add("A")
    mem._storage.similar = ["A"]
    mem.plan = {"delete": ["A"], "insert": True}
    with agent_task_scope("agent", "task"):
        rec = mem.remember("C", source="user")
    (w_c, *_), (w_merge, _, src, *_) = writes(path)
    assert bound(path, w_c) == [rec.id]
    assert parents(path, w_merge) == {"A": "in_prompt"}
    assert bound(path, w_merge) == [rec.id]
    assert src == "user"
    assert tombstones(path) == [("A", "consolidation", w_c, "agent", "task")]


def test_delete_only_records_tombstones_not_compaction(path, mem):
    mem.add("A", "B")
    mem._storage.similar = ["A", "B"]
    mem.plan = {"delete": ["B"], "insert": False}
    with agent_task_scope("agent", "task"):
        rec = mem.remember("C")
    assert rec.id == "A" and "B" not in mem._storage.records
    ((w_c, *_),) = writes(path)
    assert bound(path, w_c) == ["A"]
    assert tombstones(path) == [("B", "consolidation", w_c, "agent", "task")]


def test_delete_only_then_failure_still_recorded(path, mem):
    mem.add("A")
    mem._storage.similar = ["A"]
    mem.plan = {"delete": ["A"], "insert": False}
    mem.fail = RuntimeError("after delete")
    with agent_task_scope("agent", "task"), pytest.raises(RuntimeError):
        mem.remember("C")
    ((w_c, *_),) = writes(path)
    assert bound(path, w_c) == []
    assert tombstones(path) == [("A", "consolidation", w_c, "agent", "task")]


def test_consolidation_tombstones_and_binds_in_one_commit(path, cap):
    m = instrument(StubMemory(pool=False), cap)
    m.add("A", "B")
    m._storage.similar = ["A", "B"]
    m.plan = {"delete": ["A", "B"], "insert": True}
    commits = []
    cap._ledger._conn.set_trace_callback(lambda sql: sql == "COMMIT" and commits.append(sql))
    with agent_task_scope("agent", "task"):
        m.remember("C")
    cap._ledger._conn.set_trace_callback(None)
    assert [r for r, *_ in tombstones(path)] == ["A", "B"]
    assert len(commits) == 2  # the write before the save; the rest after it


@pytest.mark.parametrize("items, caused", [(["C"], True), (["C", "D"], False)],
                         ids=["one-item", "two-items"])
def test_remember_many_consolidation_tombstones(path, mem, items, caused):
    mem.add("A")
    mem._storage.similar = ["A"]
    mem.plan = {"delete": ["A"], "insert": True}
    with agent_task_scope("agent", "task"):
        mem.remember_many(items)
    mem.drain_writes()
    first = writes(path)[0][0]
    # Which item's save deleted the record is only known for one item.
    assert tombstones(path) == [("A", "consolidation", first if caused else None,
                                 "agent", "task")]


def test_consolidating_remember_commits_twice(path, cap):
    # One commit records the write before the save; one records every
    # compaction and bind after it.
    m = instrument(StubMemory(pool=False), cap)
    m.add("A", "B", "X")
    m._storage.similar = ["A", "B", "X"]
    m.plan = {"update": {"A": "a2", "X": "x2"}, "delete": ["B"], "insert": True}
    commits = []
    cap._ledger._conn.set_trace_callback(lambda sql: sql == "COMMIT" and commits.append(sql))
    with agent_task_scope("agent", "task"):
        m.remember("C")
    cap._ledger._conn.set_trace_callback(None)
    assert len(writes(path)) == 4  # the write plus three compactions
    assert len(commits) == 2


def test_no_compaction_without_rewrite_or_delete(path, mem):
    mem.add("A")
    mem._storage.similar = ["A"]  # consulted, but left alone
    with agent_task_scope("agent", "task"):
        mem.remember("C")
    assert len(writes(path)) == 1
    assert parents(path, last_write(path)) == {}


def test_compaction_recorded_even_if_save_fails(path, mem):
    mem.add("A")
    mem._storage.similar = ["A"]
    mem.plan = {"update": {"A": "rewritten"}, "insert": True}
    mem.fail = RuntimeError("insert failed")
    with agent_task_scope("agent", "task"), pytest.raises(RuntimeError):
        mem.remember("C")
    (w_c, *_), (w_merge, *_) = writes(path)
    assert bound(path, w_c) == []
    assert bound(path, w_merge) == ["A"]


def test_storage_calls_outside_remember_not_observed(path, mem):
    mem.add("A")
    mem._storage.update(StubRecord("direct", id="A"))
    mem._storage.delete(record_ids=["A"])
    mem._storage.reset()
    assert writes(path) == [] and tombstones(path) == []


# --- Direct edits and deletions ----------------------------------------------


def content_hash(path, write_id):
    (h,) = raw(path).execute(
        "SELECT content_hash FROM writes WHERE write_id=?", (write_id,)
    ).fetchone()
    return h


def test_update_is_compaction_recorded_before_store_write(path, mem):
    a, k = mem.add("A", "K")
    mem.results["k"] = [k]
    seen = []
    orig = mem._storage.update
    mem._storage.update = lambda record: (seen.append(len(writes(path))), orig(record))
    with agent_task_scope("agent", "task"):
        mem.recall("k")
        rec = mem.update("A", "corrected A")
    assert rec.content == "corrected A" and seen == [1]
    ((wid, store, src, agent_id, session_id),) = writes(path)
    assert (store, src, agent_id, session_id) == ("crewai", "agent_inference", "agent", "task")
    assert parents(path, wid) == {"A": "in_prompt", "K": "retrieved_in_scope"}
    assert bound(path, wid) == ["A"]
    assert content_hash(path, wid) == hashlib.sha256(b"corrected A").hexdigest()


def test_metadata_only_update_hashes_unchanged_content(path, mem):
    mem.add("A")
    with agent_task_scope("agent", "task"):
        mem.update("A", importance=0.9)
    ((wid, *_),) = writes(path)
    assert content_hash(path, wid) == hashlib.sha256(b"content of A").hexdigest()
    assert bound(path, wid) == ["A"]


def test_update_missing_record_raises_and_records_nothing(path, mem):
    with agent_task_scope("agent", "task"), pytest.raises(ValueError):
        mem.update("nope")
    assert writes(path) == []


def test_failed_update_leaves_compaction_unbound(path, mem):
    mem.add("A")

    def boom(record):
        raise RuntimeError("store down")

    mem._storage.update = boom
    with agent_task_scope("agent", "task"), pytest.raises(RuntimeError):
        mem.update("A", "new")
    ((wid, *_),) = writes(path)
    assert bound(path, wid) == []


def test_read_only_update_records_nothing(path, cap):
    m = instrument(StubMemory(read_only=True), cap)
    m.add("A")
    with agent_task_scope("agent", "task"):
        assert m.update("A", "new").content == "content of A"
    assert writes(path) == []


def test_forget_by_ids_records_tombstones(path, mem):
    _, _, k = mem.add("A", "B", "K")
    mem.results["k"] = [k]
    with agent_task_scope("agent", "task"):
        mem.recall("k")
        # "nope" never existed and "A" is repeated: neither gets an extra tombstone.
        assert mem.forget(record_ids=["A", "B", "nope", "A"]) == 2
    assert set(mem._storage.records) == {"K"}
    assert writes(path) == []
    assert tombstones(path) == [
        ("A", "forget", None, "agent", "task"), ("B", "forget", None, "agent", "task"),
    ]


def test_forget_by_ids_checks_only_those_ids(mem):
    mem.add("A", "B")
    mem._storage.list_records = None  # a full listing would fail
    mem.forget(record_ids=["A"])
    assert set(mem._storage.records) == {"B"}


def _scoped(mem, *specs):
    for rid, scope in specs:
        mem._storage.records[rid] = StubRecord(f"content of {rid}", id=rid, scope=scope)


@pytest.mark.parametrize("call, deleted", [
    (lambda m: m.forget(scope="/x"), ["X1", "X2"]),
    (lambda m: m.forget(record_ids=[]), ["R", "X1", "X2", "Y"]),
    (lambda m: m.forget(record_ids=["Y"], categories=["c"]), ["R", "X1", "X2", "Y"]),
    (lambda m: m.reset("/x"), ["X1", "X2"]),
    (lambda m: m.reset(), ["R", "X1", "X2", "Y"]),
    (lambda m: m.reset_all(), ["R", "X1", "X2", "Y"]),
], ids=["forget-filter", "forget-empty-ids", "forget-ids-and-filter", "reset-scope",
        "reset", "reset_all"])
def test_filter_deletes_record_a_tombstone_per_deleted_record(path, mem, call, deleted):
    _scoped(mem, ("R", "/"), ("X1", "/x"), ("X2", "/x/sub"), ("Y", "/y"))
    with agent_task_scope("agent", "task"):
        call(mem)
        call(mem)  # nothing left to delete the second time
    assert sorted(set(["R", "X1", "X2", "Y"]) - set(mem._storage.records)) == deleted
    assert writes(path) == []
    assert tombstones(path) == [(r, "forget", None, "agent", "task") for r in deleted]


def test_delete_outside_scope_records_tombstone_without_actor(path, mem, caplog):
    mem.add("A")
    with caplog.at_level(logging.WARNING, logger="dod.integrations.crewai"):
        mem.reset_all()
    assert tombstones(path) == [("A", "forget", None, None, None)]
    assert caplog.records == []


def test_reset_covers_saves_drained_inside_it(path, mem):
    release = _hold_save_pool(mem)
    with agent_task_scope("agent", "task"):
        mem.remember_many(["C"])
        threading.Timer(0.05, release.set).start()
        mem.reset()  # drains the pending save, then deletes what it saved
    assert mem._storage.records == {}
    ((w_c, *_),) = writes(path)
    (rid,) = bound(path, w_c)
    assert tombstones(path) == [(rid, "forget", None, "agent", "task")]


def test_failed_delete_records_what_was_deleted(path, mem):
    mem.add("A", "B")
    orig = mem._storage.delete

    def partial(*args, **kwargs):
        orig(record_ids=["A"])
        raise RuntimeError("store down")

    mem._storage.delete = partial
    integ._observe_deletes(mem._storage)
    with agent_task_scope("agent", "task"), pytest.raises(RuntimeError):
        mem.forget(scope="/")
    assert tombstones(path) == [("A", "forget", None, "agent", "task")]


def test_failed_delete_with_nothing_deleted_records_nothing(path, mem):
    mem.add("A")

    def boom(*args, **kwargs):
        raise RuntimeError("store down")

    mem._storage.delete = boom
    integ._observe_deletes(mem._storage)
    with pytest.raises(RuntimeError):
        mem.forget(record_ids=["A"])
    assert tombstones(path) == []


# --- Real crewai 1.15.22 -----------------------------------------------------


def _params(fn):
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


@pytest.mark.crewai
def test_stub_signatures_match_crewai():
    pytest.importorskip("crewai")
    from crewai.memory.unified_memory import Memory

    for name in ("recall", "remember", "remember_many", "update", "forget", "reset", "reset_all"):
        assert _params(getattr(StubMemory, name)) == _params(getattr(Memory, name)), name
    # remember_many's background save, wrapped to bind its writes.
    assert callable(getattr(Memory, "_background_encode_batch"))
    assert "self._background_encode_batch" in inspect.getsource(Memory.remember_many)


class _MemStorage:
    """Minimal in-memory crewai StorageBackend. By default scores stay below
    the consolidation threshold, so saves never need an LLM; raise ``score``
    to make EncodingFlow consolidate."""

    def __init__(self):
        self.records = {}
        self.score = 0.5

    def save(self, records):
        for r in records:
            self.records[r.id] = r

    def search(self, query_embedding, scope_prefix=None, categories=None,
               metadata_filter=None, limit=10, min_score=0.0):
        return [(r, self.score) for r in list(self.records.values())[:limit]]

    def update(self, record):
        self.records[record.id] = record

    def delete(self, scope_prefix=None, categories=None, record_ids=None,
               older_than=None, metadata_filter=None):
        for rid in record_ids or ():
            self.records.pop(rid, None)
        return len(record_ids or ())

    def get_record(self, record_id):
        return self.records.get(record_id)

    def count(self, scope_prefix=None):
        return len(self.records)

    def list_records(self, scope_prefix=None, limit=200, offset=0):
        return list(self.records.values())[offset:offset + limit]

    def reset(self, scope_prefix=None):
        for rid, r in list(self.records.items()):
            if scope_prefix is None or r.scope.startswith(scope_prefix):
                del self.records[rid]

    def list_scopes(self, path="/"):
        return []

    def list_categories(self, scope_prefix=None):
        return {}

    def touch_records(self, record_ids):
        pass


class _NoLLM:
    def call(self, *args, **kwargs):
        raise AssertionError("test must not reach the LLM")


class _PlanLLM:
    """Answers crewai's consolidation call with a scripted ConsolidationPlan."""

    def __init__(self):
        self.plan = None
        self.calls = 0

    def call(self, messages, **kwargs):
        self.calls += 1
        return json.dumps(self.plan)


def _one_hot(texts):
    # Distinct texts get orthogonal vectors, so in-batch dedup keeps them.
    return [[float(i == sum(map(ord, t)) % 16) for i in range(16)] for t in texts]


def _real_memory(llm=None, embedder=None):
    from crewai.memory.unified_memory import Memory

    return Memory(
        storage=_MemStorage(),
        llm=llm or _NoLLM(),
        embedder=embedder or (lambda texts: [[1.0, 0.0, 0.0] for _ in texts]),
        exploration_budget=0,
    )


def _save(m, content):
    # Explicit fields skip the LLM field-resolution call.
    return m.remember(content, scope="/", categories=[], importance=0.5)


@pytest.mark.crewai
def test_real_memory_end_to_end(path, cap):
    pytest.importorskip("crewai")
    m = _real_memory(embedder=_one_hot)
    before = {n: inspect.signature(getattr(m, n)) for n in ("recall", "remember", "remember_many")}
    assert instrument(m, cap) is m
    assert {n: inspect.signature(getattr(m, n)) for n in before} == before
    m.model_dump(exclude={"embedder", "llm", "storage"})  # pydantic still happy

    with agent_task_scope("agent", "task-0"):
        a, b = _save(m, "A"), _save(m, "B")
    with agent_task_scope("agent", "task-1"):
        found = m.recall("anything", depth="shallow")
        c = _save(m, "C")
    assert {x.record.id for x in found} == {a.id, b.id}
    wc = write_for(path, c.id)
    assert parents(path, wc) == {a.id: "retrieved_in_scope", b.id: "retrieved_in_scope"}

    with agent_task_scope("agent", "task-2"):
        m.recall("anything", depth="shallow")
        assert m.remember_many(["D", "E"], scope="/", categories=[], importance=0.5) == []
    m.drain_writes()
    w_d, w_e = (w for w, *_ in writes(path)[-2:])
    by_content = {r.content: r.id for r in m._storage.records.values()}
    assert bound(path, w_d) == [by_content["D"]] and bound(path, w_e) == [by_content["E"]]
    assert set(parents(path, w_d)) == {a.id, b.id, c.id}
    assert cap._context == {}


@pytest.mark.crewai
def test_real_update_and_forget_recorded(path, cap):
    pytest.importorskip("crewai")
    m = instrument(_real_memory(embedder=_one_hot), cap)
    with agent_task_scope("setup", "setup"):
        a, b = _save(m, "A"), _save(m, "B")
        e = m.remember("E", scope="/elsewhere", categories=[], importance=0.5)
    with agent_task_scope("agent", "task"):
        m.recall("anything", depth="shallow")
        rec = m.update(a.id, "corrected A")
        assert m.forget(record_ids=[b.id]) == 1
        # MemoryScope delegates to the instance, so its deletes are seen too.
        m.scope("/elsewhere").reset()
    assert m._storage.records[a.id].content == "corrected A"
    assert set(m._storage.records) == {a.id}
    (w_upd,) = (w for w, *_ in writes(path)[3:])
    assert bound(path, w_upd) == [rec.id] == [a.id]
    assert parents(path, w_upd) == {
        a.id: "in_prompt", b.id: "retrieved_in_scope", e.id: "retrieved_in_scope",
    }
    assert tombstones(path) == [
        (b.id, "forget", None, "agent", "task"), (e.id, "forget", None, "agent", "task"),
    ]


@pytest.mark.crewai
def test_real_lancedb_filter_deletes_recorded(path, cap, tmp_path):
    # crewai's default backend, so the listing and the deletes are its own.
    pytest.importorskip("crewai")
    from crewai.memory.storage.lancedb_storage import LanceDBStorage
    from crewai.memory.unified_memory import Memory

    m = instrument(Memory(
        storage=LanceDBStorage(path=tmp_path / "lance", vector_dim=16),
        llm=_NoLLM(), embedder=_one_hot, exploration_budget=0,
    ), cap)
    with agent_task_scope("setup", "setup"):
        recs = {c: m.remember(c, scope=s, categories=[], importance=0.5)
                for c, s in (("A", "/a"), ("B", "/b/x"), ("C", "/b"), ("D", "/d"))}
    with agent_task_scope("agent", "task"):
        m.forget(record_ids=[recs["A"].id])
        m.reset(scope="/b")
        m.reset_all()
    actor = ("agent", "task")
    assert tombstones(path)[0] == (recs["A"].id, "forget", None, *actor)
    assert sorted(tombstones(path)[1:3]) == sorted(
        (recs[c].id, "forget", None, *actor) for c in "BC"
    )
    assert tombstones(path)[3:] == [(recs["D"].id, "forget", None, *actor)]


@pytest.mark.crewai
def test_real_delete_only_consolidation(path, cap):
    pytest.importorskip("crewai")
    llm = _PlanLLM()
    m = instrument(_real_memory(llm), cap)
    with agent_task_scope("setup", "setup"):
        a, b = _save(m, "A"), _save(m, "B")
    m._storage.score = 0.95
    llm.plan = {
        "actions": [
            {"action": "keep", "record_id": a.id},
            {"action": "delete", "record_id": b.id},
        ],
        "insert_new": False,
    }
    with agent_task_scope("agent", "task"):
        rec = m.remember("C", scope="/", categories=[], importance=0.5)
    assert llm.calls == 1 and b.id not in m._storage.records
    (w_c,) = (w for w, *_ in writes(path)[2:])
    # crewai returns the first similar record; the write is bound to it.
    assert bound(path, w_c) == [rec.id]
    assert tombstones(path) == [(b.id, "consolidation", w_c, "agent", "task")]


@pytest.mark.crewai
def test_real_consolidation_recorded_as_compaction(path, cap):
    # Drives crewai's own EncodingFlow through update + delete + insert; only
    # the LLM's plan is scripted.
    pytest.importorskip("crewai")
    llm = _PlanLLM()
    m = instrument(_real_memory(llm), cap)
    with agent_task_scope("setup", "setup"):
        a, b, x = _save(m, "A"), _save(m, "B"), _save(m, "X")
    assert llm.calls == 0

    m._storage.score = 0.95  # above consolidation_threshold
    llm.plan = {
        "actions": [
            {"action": "update", "record_id": a.id, "new_content": "A merged with C"},
            {"action": "delete", "record_id": b.id},
            {"action": "keep", "record_id": x.id},
        ],
        "insert_new": True,
    }
    with agent_task_scope("agent", "task"):
        c = m.remember("C", scope="/", categories=[], importance=0.5, source="user")
    assert llm.calls == 1
    store = m._storage.records
    assert store[a.id].content == "A merged with C" and b.id not in store and c.id in store

    w_c, w_a, w_new = (w for w, *_ in writes(path)[3:])
    assert bound(path, w_c) == [c.id]
    consulted = {a.id: "in_prompt", b.id: "in_prompt", x.id: "in_prompt"}
    # The rewritten record comes from the consolidation LLM.
    assert parents(path, w_a) == consulted and bound(path, w_a) == [a.id]
    # The newly inserted record keeps the caller's source.
    assert parents(path, w_new) == consulted and bound(path, w_new) == [c.id]
    sources = {w: s for w, _, s, *_ in writes(path)}
    assert (sources[w_c], sources[w_a], sources[w_new]) == ("user", "agent_inference", "user")


@pytest.mark.crewai
def test_real_in_prompt_paths(path, cap):
    pytest.importorskip("crewai")
    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.memory_tools import RecallMemoryTool

    m = instrument(_real_memory(), cap)
    with agent_task_scope("setup", "setup"):
        a = _save(m, "A")

    agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini", memory=m)
    task = Task(description="short query", expected_output="x", agent=agent)
    with agent_task_scope("agent", "prompt"):
        prompt = agent._retrieve_memory_context(task, "base prompt")
        assert "Relevant memories" in prompt
        _save(m, "from prompt")
    assert parents(path, last_write(path)) == {a.id: "in_prompt"}

    # Later recalls also return the records saved above; check every tag.
    with agent_task_scope("agent", "tool"):
        RecallMemoryTool(memory=m)._run("query")
        _save(m, "from tool")
    tags = parents(path, last_write(path))
    assert a.id in tags and set(tags.values()) == {"in_prompt"}

    with agent_task_scope("agent", "user"):
        m.recall("query")  # user code: deep recall, not injected by crewai
        _save(m, "from user")
    tags = parents(path, last_write(path))
    assert a.id in tags and set(tags.values()) == {"retrieved_in_scope"}


@pytest.mark.crewai
def test_install_task_scopes_enters_scope_in_real_task(path, cap):
    pytest.importorskip("crewai")
    from crewai.agent.core import Agent
    from crewai.hooks.dispatch import InterceptionPoint, unregister
    from crewai.task import Task

    m = instrument(_real_memory(), cap)
    with agent_task_scope("setup", "setup"):
        a = _save(m, "A")
    seen = []

    class ProbeAgent(Agent):
        def execute_task(self, task, context=None, tools=None):
            seen.append(integ._key.get())
            m.recall("query", depth="shallow")
            _save(m, "derived")
            return "done"

    integ.install_task_scopes()
    integ.install_task_scopes()  # idempotent
    try:
        agent = ProbeAgent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
        task = Task(description="d", expected_output="x", agent=agent)
        task.execute_sync()
    finally:
        unregister(InterceptionPoint.PRE_STEP, integ._enter_task_scope)
        unregister(InterceptionPoint.POST_STEP, integ._exit_task_scope)
    assert seen == [(str(agent.id), str(task.id))]
    assert integ._key.get() is None  # POST_STEP restored the previous key
    wid = last_write(path)
    assert parents(path, wid) == {a.id: "retrieved_in_scope"}
    assert writes(path)[-1][3:] == (str(agent.id), str(task.id))
