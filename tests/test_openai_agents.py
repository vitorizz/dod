import asyncio
import gc
import json

import pytest

pytest.importorskip("agents")
pytestmark = pytest.mark.openai_agents

from agents import (Agent, ModelResponse, RunConfig, RunContextWrapper, Runner,  # noqa: E402
                    Usage, function_tool)
from agents.models.interface import Model  # noqa: E402
from openai.types.responses import (ResponseFunctionToolCall,  # noqa: E402
                                    ResponseOutputMessage, ResponseOutputText)

from dod.capture import Capture  # noqa: E402
from dod.integrations.openai_agents import call_record_id, instrument_tools  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402

CONFIG = RunConfig(tracing_disabled=True)


class Scripted(Model):
    """Each step: ("call", tool, args, call_id) or ("say", text)."""

    def __init__(self, steps):
        self.steps = list(steps)

    async def get_response(self, system_instructions, input, model_settings, tools,
                           output_schema, handoffs, tracing, **kwargs):
        step = self.steps.pop(0)
        if step[0] == "call":
            out = [ResponseFunctionToolCall(type="function_call", call_id=step[3], name=step[1],
                                            arguments=json.dumps(step[2]), id="fc_" + step[3],
                                            status="completed")]
        else:
            out = [ResponseOutputMessage(id="msg", type="message", role="assistant",
                                         status="completed",
                                         content=[ResponseOutputText(type="output_text",
                                                                     text=step[1],
                                                                     annotations=[])])]
        return ModelResponse(output=out, usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


DB = {}


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        yield led


def make_tools():
    """Fresh tools: instrumenting wraps them in place."""

    @function_tool
    def lookup(topic: str) -> str:
        """Look up a policy."""
        return DB.get(topic, "unknown")

    @function_tool
    def save_note(note: str) -> str:
        """Save a note."""
        DB["note"] = note
        return "saved"

    @function_tool(failure_error_function=None)
    def broken(x: str) -> str:
        """Always fails."""
        raise RuntimeError("down")

    @function_tool
    def forget(topic: str) -> str:
        """Forget a topic."""
        DB.pop(topic, None)
        return "ok"

    return [lookup, save_note, broken, forget]


@pytest.fixture
def tools(ledger):
    DB.clear()
    DB["refunds"] = "refund window: 90 days"
    return instrument_tools(make_tools(), Capture(ledger),
                            read_only=["lookup"], record_id=lambda n, a, r: a.get("note"),
                            deleted=lambda n, a, r: [a["topic"]] if n == "forget" else [])


def run(tools, steps, input="hi"):
    agent = Agent(name="support", model=Scripted(steps), tools=tools)
    return asyncio.run(Runner.run(agent, input, run_config=CONFIG))


def derived(ledger, rid):
    return {i for its in blast_radius(ledger, "openai_agents", rid).bindings.values()
            for i in its} - {("openai_agents", rid)}


def test_a_read_in_the_run_is_a_parent_of_its_writes(ledger, tools):
    run(tools, [("call", "lookup", {"topic": "refunds"}, "c1"),
                ("call", "save_note", {"note": "told Ben 90 days"}, "c2"), ("say", "done")])
    assert derived(ledger, call_record_id("c1")) == {("openai_agents", "told Ben 90 days")}
    t = trace(ledger, "openai_agents", "told Ben 90 days")
    assert [r.source for r in t.roots] == ["tool"]
    assert DB["note"] == "told Ben 90 days"


def test_separate_runs_dont_share_context(ledger, tools):
    run(tools, [("call", "lookup", {"topic": "refunds"}, "c1"), ("say", "done")])
    gc.collect()
    run(tools, [("call", "save_note", {"note": "unrelated"}, "c2"), ("say", "done")])
    assert derived(ledger, call_record_id("c1")) == set()


def test_history_carries_reads_into_a_later_run(ledger, tools):
    first = run(tools, [("call", "lookup", {"topic": "refunds"}, "c1"), ("say", "90 days")])
    gc.collect()
    later = first.to_input_list() + [{"role": "user", "content": "note it"}]
    run(tools, [("call", "save_note", {"note": "from history"}, "c9"), ("say", "ok")], later)
    assert derived(ledger, call_record_id("c1")) == {("openai_agents", "from history")}


def test_scope_is_used_when_active(ledger, tools):
    with agent_task_scope("agent", "task"):
        run(tools, [("call", "lookup", {"topic": "refunds"}, "c1"), ("say", "done")])
        run(tools, [("call", "save_note", {"note": "same task"}, "c2"), ("say", "done")])
    assert derived(ledger, call_record_id("c1")) == {("openai_agents", "same task")}
    (agent_id,) = {a for (a,) in ledger._read_conn.execute("SELECT agent_id FROM writes")}
    assert agent_id == "agent"


def test_failed_write_stays_unbound(ledger, tools):
    with pytest.raises(Exception):
        run(tools, [("call", "broken", {"x": "1"}, "c1"), ("say", "done")])
    assert ledger._read_conn.execute("SELECT count(*) FROM bindings").fetchone() == (0,)
    assert ledger._read_conn.execute("SELECT count(*) FROM writes").fetchone() == (1,)


def test_deleted_records_tombstones(ledger, tools):
    run(tools, [("call", "forget", {"topic": "refunds"}, "c1"), ("say", "done")])
    assert ledger._read_conn.execute(
        "SELECT record_id, reason FROM tombstones").fetchall() == [("refunds", "forget")]


def test_quarantined_reads_leave_the_context(ledger, tools):
    first = run(tools, [("call", "lookup", {"topic": "refunds"}, "c1"), ("say", "90 days")])
    ledger.quarantine([("openai_agents", call_record_id("c1"))])
    gc.collect()
    run(tools, [("call", "save_note", {"note": "clean"}, "c2"), ("say", "ok")],
        first.to_input_list())
    assert derived(ledger, call_record_id("c1")) == set()


def test_passthrough_tools_are_only_scoped(ledger):
    from dod.scope import current_key
    seen = []

    @function_tool
    def inner(x: str) -> str:
        """Records the scope it ran in."""
        seen.append(current_key())
        return x

    instrument_tools([inner], Capture(ledger), passthrough=["inner"])
    run([inner], [("call", "inner", {"x": "1"}, "c1"), ("say", "done")])
    assert seen and seen[0][0] == "support"
    assert ledger._read_conn.execute("SELECT count(*) FROM writes").fetchone() == (0,)


def test_validation(ledger):
    with pytest.raises(ValueError):
        instrument_tools([], Capture(ledger), read_only=["a"], passthrough=["a"])
    with pytest.raises(TypeError):
        instrument_tools([], object())
    ctx_tool = RunContextWrapper  # non-FunctionTools pass through unchanged
    assert instrument_tools([ctx_tool], Capture(ledger)) == [ctx_tool]
