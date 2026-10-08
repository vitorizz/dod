"""``dod.reconcile`` against each real integration: writes made outside capture
(a second, uninstrumented client of the same store; direct storage calls;
files edited on disk) are found, and reconciling records no retrievals."""

import asyncio
import hashlib
import os
import sqlite3

import pytest

from dod.capture import Capture
from dod.ledger import Ledger
from dod.reconcile import reconcile
from dod.scope import agent_task_scope

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")


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


def kinds(report):
    return sorted((f.store, f.record_id, f.kind) for f in report.findings)


def parents_of_last_write(path):
    conn = sqlite3.connect(str(path))
    (wid,) = conn.execute("SELECT write_id FROM writes ORDER BY seq DESC LIMIT 1").fetchone()
    return set(conn.execute(
        "SELECT parent_store, parent_record_id FROM parents WHERE write_id = ?", (wid,)))


# LangGraph -----------------------------------------------------------------


@pytest.mark.langgraph
def test_langgraph_finds_writes_outside_capture(path, ledger, cap):
    pytest.importorskip("langgraph.store.memory")
    from langgraph.store.base import PutOp
    from langgraph.store.memory import InMemoryStore

    from dod.integrations import langgraph

    store = langgraph.instrument(InMemoryStore(), cap)
    adapter = langgraph.ReconcileAdapter(store, page=2)  # several pages
    with agent_task_scope("agent", "t1"):
        for k in ("a", "b", "c", "d"):
            store.put(("ns",), k, {"v": k})
        store.put(("ns", "deep"), "e", {"v": "e"})
    assert reconcile(ledger, {"langgraph": adapter}, grace=0).ok

    # Outside capture: the store's own batch, as another process would.
    raw = type(store)._dod_base.batch
    raw(store, [PutOp(("ns",), "x", {"v": "planted"})])   # insert
    raw(store, [PutOp(("ns",), "b", {"v": "changed"})])   # edit
    raw(store, [PutOp(("ns",), "c", None)])               # delete
    with agent_task_scope("agent", "t2"):
        store.delete(("ns",), "d")                        # recorded deletion...
    raw(store, [PutOp(("ns",), "d", {"v": "back"})])      # ...undone outside capture
    with agent_task_scope("agent", "t3"):
        store.get(("ns",), "x")                           # the planted item is read
        store.put(("ns",), "derived", {"v": "from x"})    # and derived from

    rid = langgraph.record_id
    report = reconcile(ledger, {"langgraph": adapter}, grace=0)
    assert kinds(report) == [
        ("langgraph", rid(("ns",), "b"), "modified"),
        ("langgraph", rid(("ns",), "c"), "missing"),
        ("langgraph", rid(("ns",), "d"), "resurrected"),
        ("langgraph", rid(("ns",), "x"), "unrecorded"),
    ]
    by = {f.record_id: f for f in report.findings}
    assert len(by[rid(("ns",), "x")].read_into) == 1
    assert report.content_checked == {"langgraph": True}

    # Reconciling inside a scope adds nothing to its context.
    with agent_task_scope("agent", "t4"):
        reconcile(ledger, {"langgraph": adapter}, grace=0)
        store.put(("ns",), "after", {"v": 1})
    assert parents_of_last_write(path) == set()


@pytest.mark.langgraph
def test_langgraph_adapter_on_uninstrumented_store_and_bad_args():
    pytest.importorskip("langgraph.store.memory")
    from langgraph.store.memory import InMemoryStore

    from dod.integrations import langgraph

    plain = InMemoryStore()
    plain.put(("a",), "k", {"v": 1})
    adapter = langgraph.ReconcileAdapter(plain)
    ((rid, h),) = adapter.scan()
    assert rid == langgraph.record_id(("a",), "k")
    assert h == hashlib.sha256(b'{"v":1}').hexdigest()
    assert adapter.get(rid) == (True, h)
    assert adapter.get(langgraph.record_id(("a",), "nope")) == (False, None)
    with pytest.raises(TypeError):
        langgraph.ReconcileAdapter(object())
    with pytest.raises(ValueError):
        langgraph.ReconcileAdapter(plain, page=0)


# CrewAI ----------------------------------------------------------------------


def _toy_embed(texts):
    out = []
    for text in texts:
        v = [0.0] * 64
        for word in text.lower().split():
            v[hashlib.sha256(word.encode()).digest()[0] % 64] += 1.0
        out.append(v)
    return out


def _memory(storage_dir):
    from crewai.memory.unified_memory import Memory

    return Memory(storage=str(storage_dir), embedder=_toy_embed, consolidation_threshold=1.0)


def _remember(memory, content):
    return memory.remember(content, scope="/notes", categories=[], importance=0.5)


@pytest.mark.crewai
def test_crewai_finds_writes_outside_capture(path, ledger, cap, tmp_path):
    pytest.importorskip("crewai")
    from crewai.events.event_listener import event_listener

    from dod.integrations import crewai

    event_listener.formatter.verbose = False
    storage_dir = tmp_path / "crewai"
    memory = crewai.instrument(_memory(storage_dir), cap)
    adapter = crewai.ReconcileAdapter(memory)
    with agent_task_scope("agent", "t1"):
        kept = _remember(memory, "The user prefers window seats.")
        edited = _remember(memory, "The user's home airport is JFK.")
        dropped = _remember(memory, "The user is vegetarian.")
    # Stored content is exactly what capture hashed: nothing to report.
    report = reconcile(ledger, {"crewai": adapter}, grace=0)
    assert report.ok, report.findings
    assert report.scanned == {"crewai": 3} and report.content_checked == {"crewai": True}

    # Outside capture: a second, uninstrumented Memory on the same storage,
    # and direct storage calls.
    other = _memory(storage_dir)
    planted = _remember(other, "The user's loyalty number is 12345.")
    record = memory._storage.get_record(edited.id)
    memory._storage.update(record.model_copy(update={"content": "The user's home airport is EWR."}))
    memory._storage.delete(record_ids=[dropped.id])

    report = reconcile(ledger, {"crewai": adapter}, grace=0)
    assert kinds(report) == sorted([
        ("crewai", planted.id, "unrecorded"),
        ("crewai", edited.id, "modified"),
        ("crewai", dropped.id, "missing"),
    ])
    assert kept.id not in {f.record_id for f in report.findings}

    # Reconciling inside a scope adds nothing to its context.
    with agent_task_scope("agent", "t2"):
        reconcile(ledger, {"crewai": adapter}, grace=0)
        _remember(memory, "Unrelated.")
    assert parents_of_last_write(path) == set()


# Anthropic memory tool -------------------------------------------------------


@pytest.mark.anthropic
@pytest.mark.parametrize("asynchronous", [False, True])
def test_memory_tool_finds_files_outside_capture(path, ledger, cap, tmp_path, asynchronous):
    pytest.importorskip("anthropic.tools.memory")
    from anthropic.tools.memory import (
        BetaAsyncLocalFilesystemMemoryTool,
        BetaLocalFilesystemMemoryTool,
    )

    from dod.integrations import anthropic_memory

    base = tmp_path / "mem"
    cls = BetaAsyncLocalFilesystemMemoryTool if asynchronous else BetaLocalFilesystemMemoryTool
    tool = anthropic_memory.instrument(cls(base_path=str(base)), cap)

    def call(command):
        result = tool.call(command)
        return asyncio.run(result) if asynchronous else result

    adapter = anthropic_memory.ReconcileAdapter(tool)
    with agent_task_scope("agent", "t1"):
        for name in ("a.md", "b.md", "deep/x/y/z.md"):
            call({"command": "create", "path": f"/memories/{name}", "file_text": name})
    report = reconcile(ledger, {"anthropic_memory": adapter}, grace=0)
    assert report.ok and report.scanned == {"anthropic_memory": 3}
    assert report.content_checked == {"anthropic_memory": False}

    files = base / "memories"
    (files / "planted.md").write_text("written by hand")
    (files / "b.md").unlink()
    (files / "a.md").write_text("edited by hand")  # existence only: not detected
    report = reconcile(ledger, {"anthropic_memory": adapter}, grace=0)
    assert kinds(report) == [
        ("anthropic_memory", "/memories/b.md", "missing"),
        ("anthropic_memory", "/memories/planted.md", "unrecorded"),
    ]

    # Reconciling inside a scope adds nothing to its context.
    with agent_task_scope("agent", "t2"):
        reconcile(ledger, {"anthropic_memory": adapter}, grace=0)
        call({"command": "create", "path": "/memories/after.md", "file_text": "x"})
    assert parents_of_last_write(path) == set()


@pytest.mark.anthropic
def test_memory_tool_empty_store_and_read_errors(ledger, tmp_path):
    pytest.importorskip("anthropic.tools.memory")
    from anthropic.tools.memory import BetaLocalFilesystemMemoryTool

    from dod.integrations import anthropic_memory

    tool = BetaLocalFilesystemMemoryTool(base_path=str(tmp_path / "mem"))
    adapter = anthropic_memory.ReconcileAdapter(tool)  # uninstrumented works too
    assert adapter.scan() == []
    assert adapter.get("/memories/none.md") == (False, None)

    class Failing(BetaLocalFilesystemMemoryTool):
        def view(self, command):
            raise OSError("disk unreadable")

    with pytest.raises(OSError):
        reconcile(ledger, {"m": anthropic_memory.ReconcileAdapter(
            Failing(base_path=str(tmp_path / "mem")))}, grace=0)
    with pytest.raises(TypeError):
        anthropic_memory.ReconcileAdapter(object())


# Across stores ---------------------------------------------------------------


@pytest.mark.langgraph
@pytest.mark.anthropic
def test_one_pass_over_several_stores(path, ledger, cap, tmp_path):
    pytest.importorskip("langgraph.store.memory")
    pytest.importorskip("anthropic.tools.memory")
    from anthropic.tools.memory import BetaLocalFilesystemMemoryTool
    from langgraph.store.memory import InMemoryStore

    from dod.integrations import anthropic_memory, langgraph

    store = langgraph.instrument(InMemoryStore(), cap)
    tool = anthropic_memory.instrument(BetaLocalFilesystemMemoryTool(base_path=str(tmp_path)), cap)
    with agent_task_scope("agent", "t1"):
        store.put(("ns",), "a", {"v": 1})
        tool.call({"command": "create", "path": "/memories/n.md", "file_text": "n"})
    (tmp_path / "memories" / "planted.md").write_text("by hand")
    with agent_task_scope("agent", "t2"):
        tool.call({"command": "view", "path": "/memories/planted.md"})
        store.put(("ns",), "derived", {"v": 2})  # a LangGraph write derived from it
    stores = {"langgraph": langgraph.ReconcileAdapter(store),
              "anthropic_memory": anthropic_memory.ReconcileAdapter(tool)}
    report = reconcile(ledger, stores, grace=0)
    (f,) = report.findings
    assert (f.store, f.record_id, f.kind) == ("anthropic_memory", "/memories/planted.md",
                                             "unrecorded")
    # Trace breaks at the planted file: the LangGraph write that used it.
    assert len(f.read_into) == 1
    assert report.scanned == {"langgraph": 2, "anthropic_memory": 2}
