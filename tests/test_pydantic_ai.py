import json
import os

import pytest

pytest.importorskip("pydantic_ai")
pytestmark = pytest.mark.pydantic_ai
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai import Agent, ModelRetry  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402
from pydantic_ai.toolsets import FunctionToolset  # noqa: E402
from test_crewai import StubMemory  # noqa: E402

import dod  # noqa: E402
from dod.capture import Capture  # noqa: E402
from dod.integrations import crewai  # noqa: E402
from dod.integrations.pydantic_ai import CaptureToolset, call_record_id  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.scope import current_key  # noqa: E402

S = "pydantic_ai"


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        yield led


def scripted(*calls):
    """A model that makes each tool call in turn, then answers."""
    counter = {"n": 0, "run": 0}

    def fn(messages, info):
        n = counter["n"]
        counter["n"] += 1
        if n < len(calls):
            name, args = calls[n]
            return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f"c{n}")])
        return ModelResponse(parts=[TextPart("done")])

    return FunctionModel(fn), counter


def tools(memory=None, seen=None):
    ts = FunctionToolset()
    tickets = {}

    @ts.tool_plain
    def lookup(name: str) -> str:
        return f"{name}: refund window 90 days"

    @ts.tool_plain
    def create_ticket(text: str) -> dict:
        tid = f"T{len(tickets) + 1}"
        tickets[tid] = text
        return {"id": tid}

    @ts.tool_plain
    def close_ticket(id: str) -> str:
        tickets.pop(id, None)
        return "closed"

    @ts.tool_plain
    def flaky() -> str:
        raise ModelRetry("try again")

    @ts.tool_plain
    def remember(text: str) -> str:
        if seen is not None:
            seen.append(current_key())
        return memory.remember(text).id

    return ts


def ticket_id(tool, args, result):
    return result["id"] if tool == "create_ticket" else None


def bound(ledger):
    with ledger._lock:
        return ledger._conn.execute(
            "SELECT b.record_id, w.source, w.agent_id FROM bindings b"
            " JOIN writes w ON w.write_id = b.write_id ORDER BY b.rowid").fetchall()


def test_write_traces_to_the_read_result_it_came_from(ledger):
    model, _ = scripted(("lookup", {"name": "ana"}), ("create_ticket", {"text": "refund ok"}))
    agent = Agent(model, name="support", toolsets=[
        CaptureToolset(tools(), Capture(ledger), read_only=["lookup"], record_id=ticket_id)])
    result = agent.run_sync("can ana get a refund?")
    assert result.output == "done"
    (read, write) = bound(ledger)
    assert read[0].startswith("call:") and read[0].endswith(":c0") and read[1] == "tool"
    assert write == ("T1", "agent_inference", "support")
    t = trace(ledger, S, "T1")
    assert [(e.parent_record_id, e.tag) for e in t.edges] == [(read[0], "in_prompt")]


def test_history_from_an_earlier_run_is_in_context(ledger):
    cap = Capture(ledger)
    model, counter = scripted(("lookup", {"name": "ana"}))
    ts = CaptureToolset(tools(), cap, read_only=["lookup"], record_id=ticket_id)
    agent = Agent(model, toolsets=[ts])
    first = agent.run_sync("look ana up")
    (read,) = bound(ledger)

    model2, _ = scripted(("create_ticket", {"text": "refund"}))
    agent2 = Agent(model2, toolsets=[ts])
    agent2.run_sync("now file it", message_history=first.all_messages())
    t = trace(ledger, S, "T1")
    assert [(e.parent_record_id, e.tag) for e in t.edges] == [(read[0], "in_prompt")]
    assert read[0] == call_record_id(first.run_id, "c0")
    assert blast_radius(ledger, S, read[0]).store_counts == {S: 1}


def test_runs_are_isolated(ledger):
    cap = Capture(ledger)
    ts = CaptureToolset(tools(), cap, read_only=["lookup"], record_id=ticket_id)
    Agent(scripted(("lookup", {"name": "ana"}))[0], toolsets=[ts]).run_sync("a")
    Agent(scripted(("create_ticket", {"text": "x"}))[0], toolsets=[ts]).run_sync("b")
    assert trace(ledger, S, "T1").edges == []
    assert cap._context == {}  # cleared once no run is active


def test_unlisted_tools_are_writes_and_failures_stay_unbound(ledger):
    model, _ = scripted(("flaky", {}), ("close_ticket", {"id": "T9"}))
    agent = Agent(model, toolsets=[CaptureToolset(tools(), Capture(ledger))])
    agent.run_sync("go")
    with ledger._lock:
        rows = ledger._conn.execute("SELECT source FROM writes").fetchall()
    assert [s for (s,) in rows] == ["agent_inference", "agent_inference"]
    (only,) = bound(ledger)
    assert only[0].endswith(":c1")  # close_ticket bound to its call ID


def test_deleted_records_are_tombstoned(ledger):
    def deleted(tool, args, result):
        return [args["id"]] if tool == "close_ticket" else []

    model, _ = scripted(("create_ticket", {"text": "x"}), ("close_ticket", {"id": "T1"}))
    agent = Agent(model, toolsets=[CaptureToolset(tools(), Capture(ledger), record_id=ticket_id,
                                                  deleted=deleted)])
    agent.run_sync("go")
    assert [d.reason for d in trace(ledger, S, "T1").deleted[(S, "T1")]] == ["forget"]


def test_passthrough_tools_scope_instrumented_stores(ledger):
    cap = Capture(ledger)
    mem = crewai.instrument(StubMemory(pool=False), cap)
    seen = []
    model, _ = scripted(("lookup", {"name": "ana"}), ("remember", {"text": "ana: 90 days"}))
    agent = Agent(model, name="support", toolsets=[CaptureToolset(
        tools(mem, seen), cap, read_only=["lookup"], passthrough=["remember"])])
    agent.run_sync("note it")
    assert seen[0][0] == "support"
    rows = bound(ledger)
    assert [s for _, s, _ in rows] == ["tool", "agent_inference"]
    crewai_id = rows[1][0]
    t = trace(ledger, "crewai", crewai_id)
    assert [e.parent_store for e in t.edges] == [S]


def test_quarantined_results_leave_the_context(ledger):
    cap = Capture(ledger)
    ts = CaptureToolset(tools(), cap, read_only=["lookup"], record_id=ticket_id)
    first = Agent(scripted(("lookup", {"name": "ana"}))[0], toolsets=[ts]).run_sync("a")
    (read,) = bound(ledger)
    dod.excise(ledger, S, read[0])
    Agent(scripted(("create_ticket", {"text": "x"}))[0], toolsets=[ts]).run_sync(
        "b", message_history=first.all_messages())
    assert trace(ledger, S, "T1").edges == []


def test_bad_arguments(ledger):
    with pytest.raises(TypeError):
        CaptureToolset(tools(), object())
    with pytest.raises(ValueError, match="both"):
        CaptureToolset(tools(), Capture(ledger), read_only=["a"], passthrough=["a"])


def test_content_is_never_stored(tmp_path):
    path = tmp_path / "l.db"
    with Ledger(path) as ledger:
        model, _ = scripted(("create_ticket", {"text": "SECRET-PAYLOAD"}))
        Agent(model, toolsets=[CaptureToolset(tools(), Capture(ledger))]).run_sync("go")
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.is_file())
    assert b"SECRET-PAYLOAD" not in raw and json.dumps("SECRET").encode() not in raw
