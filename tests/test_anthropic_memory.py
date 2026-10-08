import asyncio
import hashlib
import sqlite3

import pytest

pytest.importorskip("anthropic.tools.memory")
pytestmark = pytest.mark.anthropic

from anthropic.lib.tools import ToolError  # noqa: E402
from anthropic.tools.memory import (  # noqa: E402
    BetaAsyncLocalFilesystemMemoryTool,
    BetaLocalFilesystemMemoryTool,
)

from dod.capture import Capture  # noqa: E402
from dod.excise import excise  # noqa: E402
from dod.integrations.anthropic_memory import DeleteAdapter, instrument, record_id  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.quarantine import blocked_retrievals  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402

STORE = "anthropic_memory"


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
def mem_dir(tmp_path):
    return tmp_path / "mem"


@pytest.fixture
def tool(cap, mem_dir):
    return instrument(BetaLocalFilesystemMemoryTool(base_path=str(mem_dir)), cap)


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


def tombstones(path):
    return raw(path).execute(
        "SELECT store, record_id, reason, write_id, agent_id, session_id"
        " FROM tombstones ORDER BY seq"
    ).fetchall()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def create(tool, p, text="x\n"):
    return tool.call({"command": "create", "path": p, "file_text": text})


def view(tool, p, **kw):
    return tool.call({"command": "view", "path": p, **kw})


# --- Wrapping ----------------------------------------------------------------


def test_record_id_is_normalized_path():
    assert record_id("/memories/notes/user.md") == "/memories/notes/user.md"
    assert record_id("/memories//notes/./a/../user.md") == "/memories/notes/user.md"
    assert record_id("/memories/notes/") == "/memories/notes"


def test_return_values_unchanged_and_idempotent(cap, mem_dir):
    plain = BetaLocalFilesystemMemoryTool(base_path=str(mem_dir))
    t = instrument(plain, cap)
    assert t is plain and instrument(t, cap) is t
    reference = BetaLocalFilesystemMemoryTool(base_path=str(mem_dir.parent / "ref"))
    with agent_task_scope("agent", "task"):
        for tl in (t, reference):
            assert create(tl, "/memories/a.md", "hello\n") == (
                "File created successfully at: /memories/a.md")
        assert view(t, "/memories/a.md") == view(reference, "/memories/a.md")
        assert view(t, "/memories/a.md", view_range=[1, 1]) == view(
            reference, "/memories/a.md", view_range=[1, 1])
        listing = view(t, "/memories")
    assert listing.split("\n")[2].endswith("\t/memories/a.md")


def test_rejects_tools_that_bypass_execute(cap, mem_dir):
    class Bypass(BetaLocalFilesystemMemoryTool):
        def call(self, input):
            return self.view(input)

    with pytest.raises(TypeError, match="overrides call"):
        instrument(Bypass(base_path=str(mem_dir)), cap)
    with pytest.raises(TypeError, match="not an Anthropic memory tool"):
        instrument(object(), cap)


def test_tool_errors_propagate_unchanged(path, tool):
    with pytest.raises(ToolError, match="does not exist"):
        view(tool, "/memories/missing.md")
    with pytest.raises(ToolError, match="must start with /memories"):
        view(tool, "/etc/passwd")
    create(tool, "/memories/a.md")
    with pytest.raises(ToolError, match="did not appear verbatim"):
        tool.call({"command": "str_replace", "path": "/memories/a.md",
                   "old_str": "nope", "new_str": "y"})
    with pytest.raises(ToolError, match="does not exist"):
        tool.call({"command": "delete", "path": "/memories/missing.md"})
    # The failed str_replace was recorded first and stays unbound.
    assert bound(path, writes(path)[-1][0]) == []
    assert tombstones(path) == []


# --- Reads -------------------------------------------------------------------


def test_view_of_file_is_in_prompt_parent(path, tool):
    create(tool, "/memories/a.md")
    create(tool, "/memories/b.md")
    with agent_task_scope("agent", "task"):
        view(tool, "/memories/./a.md")
        create(tool, "/memories/c.md", "derived\n")
    w = writes(path)[-1]
    assert w[1:6] == (STORE, "agent_inference", "agent", "task", sha("derived\n"))
    assert parents(path, w[0]) == {"/memories/a.md": "in_prompt"}
    assert bound(path, w[0]) == ["/memories/c.md"]


def test_view_of_directory_lists_files_retrieved_in_scope(path, tool):
    create(tool, "/memories/a.md")
    create(tool, "/memories/notes/b.md")
    create(tool, "/memories/x/y/z/deep.md")  # past the listing's 2 levels
    with agent_task_scope("agent", "task"):
        view(tool, "/memories")
        create(tool, "/memories/c.md")
    assert parents(path, writes(path)[-1][0]) == {
        "/memories/a.md": "retrieved_in_scope",
        "/memories/notes/b.md": "retrieved_in_scope",
    }


def test_view_of_file_after_listing_upgrades_to_in_prompt(path, tool):
    create(tool, "/memories/a.md")
    with agent_task_scope("agent", "task"):
        view(tool, "/memories/")
        view(tool, "/memories/a.md")
        create(tool, "/memories/c.md")
    assert parents(path, writes(path)[-1][0]) == {"/memories/a.md": "in_prompt"}


def test_failed_view_and_views_outside_scope_record_nothing(path, tool, caplog):
    create(tool, "/memories/a.md")
    view(tool, "/memories/a.md")
    with agent_task_scope("agent", "task"):
        with pytest.raises(ToolError):
            view(tool, "/memories/missing.md")
        create(tool, "/memories/c.md")
    assert parents(path, writes(path)[-1][0]) == {}
    assert "outside agent_task_scope" in caplog.text  # the first create


# --- Writes ------------------------------------------------------------------


def test_str_replace_and_insert_are_compactions_of_the_file(path, tool):
    create(tool, "/memories/a.md", "line one\n")
    create(tool, "/memories/b.md")
    with agent_task_scope("agent", "task"):
        view(tool, "/memories/b.md")
        tool.call({"command": "str_replace", "path": "/memories/a.md",
                   "old_str": "one", "new_str": "uno"})
        tool.call({"command": "insert", "path": "/memories/a.md",
                   "insert_line": 1, "insert_text": "line two\n"})
    *_, rep, ins = writes(path)
    assert rep[5] == sha("uno") and ins[5] == sha("line two\n")
    for w in (rep, ins):
        assert parents(path, w[0]) == {"/memories/a.md": "in_prompt",
                                       "/memories/b.md": "in_prompt"}
        assert bound(path, w[0]) == ["/memories/a.md"]


class Overwriting(BetaLocalFilesystemMemoryTool):
    """A backend whose ``create`` overwrites an existing file."""

    def create(self, command):
        full = self._validate_path(command.path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(command.file_text)
        return f"File created successfully at: {command.path}"


def test_create_over_existing_file_is_compaction(path, cap, mem_dir):
    tool = instrument(Overwriting(base_path=str(mem_dir)), cap)
    create(tool, "/memories/a.md", "v1\n")
    with agent_task_scope("agent", "task"):
        create(tool, "/memories/a.md", "v2\n")
    w = writes(path)[-1]
    assert parents(path, w[0]) == {"/memories/a.md": "in_prompt"}
    assert bound(path, w[0]) == ["/memories/a.md"]
    assert w[5] == sha("v2\n")


def test_create_of_existing_file_on_sdk_tool_fails_unbound(path, tool):
    create(tool, "/memories/a.md")
    with pytest.raises(ToolError, match="already exists"):
        create(tool, "/memories/a.md")
    w = writes(path)[-1]
    assert parents(path, w[0]) == {"/memories/a.md": "in_prompt"}
    assert bound(path, w[0]) == []


def test_rename_file_is_write_with_old_parent_plus_tombstone(path, tool):
    create(tool, "/memories/a.md")
    with agent_task_scope("agent", "task"):
        tool.call({"command": "rename", "old_path": "/memories/a.md",
                   "new_path": "/memories/archive/a.md"})
    w = writes(path)[-1]
    assert parents(path, w[0]) == {"/memories/a.md": "in_prompt"}
    assert bound(path, w[0]) == ["/memories/archive/a.md"]
    assert w[5] == sha("")
    assert tombstones(path) == [(STORE, "/memories/a.md", "other", w[0], "agent", "task")]


def test_rename_directory_moves_every_file(path, tool):
    create(tool, "/memories/d/a.md")
    create(tool, "/memories/d/x/y/z/deep.md")
    tool.call({"command": "rename", "old_path": "/memories/d", "new_path": "/memories/e"})
    moved = writes(path)[-2:]
    got = {bound(path, w[0])[0]: parents(path, w[0]) for w in moved}
    assert got == {
        "/memories/e/a.md": {"/memories/d/a.md": "in_prompt"},
        "/memories/e/x/y/z/deep.md": {"/memories/d/x/y/z/deep.md": "in_prompt"},
    }
    assert {t[1] for t in tombstones(path)} == {"/memories/d/a.md", "/memories/d/x/y/z/deep.md"}


def test_delete_file_and_directory_record_tombstones(path, tool):
    create(tool, "/memories/a.md")
    create(tool, "/memories/d/b.md")
    create(tool, "/memories/d/x/y/z/deep.md")
    with agent_task_scope("agent", "task"):
        tool.call({"command": "delete", "path": "/memories/a.md"})
        tool.call({"command": "delete", "path": "/memories/d"})
    assert len(writes(path)) == 3
    assert sorted(tombstones(path)) == sorted(
        (STORE, r, "forget", None, "agent", "task")
        for r in ("/memories/a.md", "/memories/d/b.md", "/memories/d/x/y/z/deep.md")
    )


def test_clear_all_memory_tombstones_every_file(path, tool):
    create(tool, "/memories/a.md")
    create(tool, "/memories/d/b.md")
    assert tool.clear_all_memory() == "All memory cleared"
    assert sorted(t[1] for t in tombstones(path)) == ["/memories/a.md", "/memories/d/b.md"]
    assert tool.clear_all_memory() == "All memory cleared"  # nothing left
    assert len(tombstones(path)) == 2


class FailingDelete(BetaLocalFilesystemMemoryTool):
    def delete(self, command):
        raise ToolError("disk full")


def test_failed_delete_records_no_tombstone(path, cap, mem_dir):
    tool = instrument(FailingDelete(base_path=str(mem_dir)), cap)
    create(tool, "/memories/a.md")
    with pytest.raises(ToolError, match="disk full"):
        tool.call({"command": "delete", "path": "/memories/a.md"})
    assert tombstones(path) == []


def test_ledger_failure_before_command_blocks_it(cap, tool, mem_dir, monkeypatch):
    def broken(*a, **kw):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(cap, "on_batch", broken)
    with pytest.raises(sqlite3.OperationalError):
        create(tool, "/memories/a.md")
    assert not (mem_dir / "memories" / "a.md").exists()


def _fail_after_command(cap, monkeypatch):
    real = cap.on_batch

    def flaky(*args, **kwargs):
        if kwargs.get("binds") or kwargs.get("deletes"):
            raise sqlite3.OperationalError("disk I/O error")
        return real(*args, **kwargs)

    monkeypatch.setattr(cap, "on_batch", flaky)


@pytest.mark.parametrize("op", ["create", "delete"])
def test_ledger_failure_after_command_never_reaches_caller(cap, tool, monkeypatch, caplog, op):
    create(tool, "/memories/a.md", "v1\n")
    _fail_after_command(cap, monkeypatch)
    with caplog.at_level("ERROR"):
        if op == "create":
            assert create(tool, "/memories/b.md").startswith("File created")
        else:
            assert tool.call({"command": "delete", "path": "/memories/a.md"}).startswith(
                "Successfully deleted")
    assert "lineage for these files is incomplete" in caplog.text


# --- Async -------------------------------------------------------------------


def test_async_tool_is_captured_identically(path, cap, mem_dir):
    tool = instrument(BetaAsyncLocalFilesystemMemoryTool(base_path=str(mem_dir)), cap)

    async def run():
        await tool.call({"command": "create", "path": "/memories/a.md", "file_text": "A\n"})
        with agent_task_scope("agent", "task"):
            await tool.call({"command": "view", "path": "/memories/a.md"})
            await tool.call({"command": "create", "path": "/memories/b.md", "file_text": "B\n"})
            await tool.call({"command": "delete", "path": "/memories/a.md"})
        await tool.clear_all_memory()

    asyncio.run(run())
    w = writes(path)[-1]
    assert parents(path, w[0]) == {"/memories/a.md": "in_prompt"}
    assert bound(path, w[0]) == ["/memories/b.md"]
    assert [t[1:3] for t in tombstones(path)] == [
        ("/memories/a.md", "forget"), ("/memories/b.md", "forget")]


# --- Quarantine and excision -------------------------------------------------


def test_quarantined_file_is_hidden_and_never_a_parent(path, ledger, tool):
    create(tool, "/memories/bad.md", "false fact\n")
    create(tool, "/memories/good.md")
    ledger.quarantine([(STORE, "/memories/bad.md")])
    before = blocked_retrievals().get((STORE, "/memories/bad.md"), 0)
    with agent_task_scope("agent", "task"):
        with pytest.raises(ToolError, match="/memories/bad.md does not exist"):
            view(tool, "/memories/bad.md")
        listing = view(tool, "/memories")
        create(tool, "/memories/c.md")
    assert "bad.md" not in listing and "good.md" in listing
    assert parents(path, writes(path)[-1][0]) == {"/memories/good.md": "retrieved_in_scope"}
    assert blocked_retrievals()[(STORE, "/memories/bad.md")] == before + 2
    ledger.release([(STORE, "/memories/bad.md")])
    assert "false fact" in view(tool, "/memories/bad.md")


def test_excise_delete_removes_file_and_derived(path, ledger, tool, mem_dir):
    create(tool, "/memories/bad.md")
    create(tool, "/memories/other.md")
    with agent_task_scope("agent", "task"):
        view(tool, "/memories/bad.md")
        create(tool, "/memories/derived.md")
    result = excise(ledger, STORE, "/memories/bad.md", mode="delete",
                    stores={STORE: DeleteAdapter(tool)})
    assert sorted(result.done[STORE]) == ["/memories/bad.md", "/memories/derived.md"]
    assert {t[1:3] for t in tombstones(path)} == {
        ("/memories/bad.md", "excision"), ("/memories/derived.md", "excision")}
    root = mem_dir / "memories"
    assert not (root / "bad.md").exists() and (root / "other.md").exists()
    again = excise(ledger, STORE, "/memories/bad.md", mode="delete",
                   stores={STORE: DeleteAdapter(tool)})
    assert again.done == {} and {s.reason for s in again.skipped} == {"not in store"}


def test_delete_adapter_requires_instrumented_tool(mem_dir):
    with pytest.raises(TypeError, match="instrumented"):
        DeleteAdapter(BetaLocalFilesystemMemoryTool(base_path=str(mem_dir)))


def test_delete_adapter_async_tool(path, cap, mem_dir):
    tool = instrument(BetaAsyncLocalFilesystemMemoryTool(base_path=str(mem_dir)), cap)
    asyncio.run(tool.call({"command": "create", "path": "/memories/a.md", "file_text": "A"}))
    adapter = DeleteAdapter(tool)
    assert adapter.delete("/memories/a.md") is True
    assert adapter.delete("/memories/a.md") is False
    assert [t[1:3] for t in tombstones(path)] == [("/memories/a.md", "forget")]


# --- Cross-store -------------------------------------------------------------


@pytest.mark.langgraph
def test_trace_and_blast_radius_cross_store_with_langgraph(path, ledger, cap, tool):
    pytest.importorskip("langgraph.store.memory")
    from langgraph.store.memory import InMemoryStore

    from dod.integrations import langgraph

    store = langgraph.instrument(InMemoryStore(), cap)
    with agent_task_scope("agent", "plant"):
        create(tool, "/memories/user.md", "allergic to peanuts\n")  # false

    with agent_task_scope("agent", "plan"):
        view(tool, "/memories/user.md")
        store.put(("users", "u1"), "diet", {"text": "no peanuts in meal plans"})

    # And back: read from LangGraph, write a memory file.
    with agent_task_scope("agent", "shop"):
        store.get(("users", "u1"), "diet")
        create(tool, "/memories/groceries.md", "no peanut butter\n")

    lg_id = langgraph.record_id(("users", "u1"), "diet")
    t = trace(ledger, "langgraph", lg_id)
    (root,) = trace(ledger, STORE, "/memories/user.md").items
    assert [w.write_id for w in t.roots] == [root.write_id]
    assert [(e.parent_store, e.parent_record_id, e.tag) for e in t.edges] == [
        (STORE, "/memories/user.md", "in_prompt")]

    b = blast_radius(ledger, STORE, "/memories/user.md")
    assert b.store_counts == {STORE: 1, "langgraph": 1}
    reached = {item for items in b.bindings.values() for item in items}
    assert {("langgraph", lg_id), (STORE, "/memories/groceries.md")} <= reached


# --- File hashes (opt-in) ------------------------------------------------------


def binding_hashes(path):
    return raw(path).execute(
        "SELECT record_id, content_hash FROM bindings ORDER BY rowid").fetchall()


def file_sha(mem_dir, p):
    return hashlib.sha256((mem_dir / p.lstrip("/")).read_bytes()).hexdigest()


def test_file_hashes_are_off_by_default(path, tool):
    with agent_task_scope("agent", "task"):
        create(tool, "/memories/a.md", "a\n")
    assert binding_hashes(path) == [("/memories/a.md", None)]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_file_hashes_record_each_written_file(path, cap, mem_dir, asynchronous):
    cls = BetaAsyncLocalFilesystemMemoryTool if asynchronous else BetaLocalFilesystemMemoryTool
    tool = instrument(cls(base_path=str(mem_dir)), cap, hash_files=True)

    def call(command):
        result = tool.call(command)
        return asyncio.run(result) if asynchronous else result

    # Texts the view's line numbering must round-trip exactly: a trailing
    # newline, none, CRLF, tabs, and an empty file.
    texts = {"/memories/a.md": "one\ntwo\n", "/memories/b.md": "no newline",
             "/memories/c.md": "crlf\r\nline\tx\ty\r\n", "/memories/e.md": ""}
    with agent_task_scope("agent", "task"):
        for p, text in texts.items():
            call({"command": "create", "path": p, "file_text": text})
        call({"command": "str_replace", "path": "/memories/a.md",
              "old_str": "two", "new_str": "2"})
        call({"command": "insert", "path": "/memories/b.md", "insert_line": 0,
              "insert_text": "first\n"})
        call({"command": "create", "path": "/memories/dir/x.md", "file_text": "x\n"})
        call({"command": "rename", "old_path": "/memories/dir", "new_path": "/memories/moved"})
    on_disk = {p: file_sha(mem_dir, p) for p in
               ("/memories/a.md", "/memories/b.md", "/memories/c.md", "/memories/e.md",
                "/memories/moved/x.md")}
    latest = dict(binding_hashes(path))
    assert {p: latest[p] for p in on_disk} == on_disk
    assert latest["/memories/e.md"] == sha("")


def test_unreadable_file_is_bound_without_hash(path, cap, mem_dir, caplog):
    class NoViews(BetaLocalFilesystemMemoryTool):
        def view(self, command):
            if command.path.endswith("a.md"):
                raise OSError("read failed")
            return super().view(command)

    tool = instrument(NoViews(base_path=str(mem_dir)), cap, hash_files=True)
    with agent_task_scope("agent", "task"):
        result = create(tool, "/memories/a.md", "a\n")  # the probe fails too: treated as new
    assert "created" in str(result).lower()
    assert binding_hashes(path) == [("/memories/a.md", None)]
    assert "for its hash failed" in caplog.text


def test_file_text_rejects_other_formats():
    from dod.integrations.anthropic_memory import _file_text

    head = "Here's the content of /memories/a.md with line numbers:\n"
    assert _file_text("/memories/a.md", head + "     1\ta\n     2\tb") == "a\nb"
    assert _file_text("/memories/a.md", head + "     1\ta\n     3\tb") is None  # gap
    assert _file_text("/memories/a.md", head + "a\nb") is None  # unnumbered
    assert _file_text("/memories/b.md", head + "     1\ta") is None  # other path
    assert _file_text("/memories/a.md", None) is None
