"""dod's overhead per tool call, invoking the wrapped tool directly (no model)."""

import asyncio
import itertools
import types

import pytest

pytest.importorskip("agents")
pytestmark = pytest.mark.openai_agents

from agents import Usage, function_tool  # noqa: E402
from perf import p99_ms  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.integrations.openai_agents import instrument_tools  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402


def tools():
    @function_tool
    def lookup(topic: str) -> str:
        """Look up."""
        return "refund window: 90 days"

    @function_tool
    def save(note: str) -> str:
        """Save."""
        return "ok"

    return lookup, save


@pytest.mark.parametrize("which", ["read", "write"])
def test_tool_call_overhead_p99_under_1ms(tmp_path, which):
    # A run of 20 tool calls per scope, so each write carries a run-sized
    # context (every earlier read result in the run is in the prompt).
    loop = asyncio.new_event_loop()
    agent = types.SimpleNamespace(name="support")
    raw_tool = tools()[0 if which == "read" else 1]
    args = '{"topic": "refunds"}' if which == "read" else '{"note": "told Ben"}'

    def caller(tool):
        calls = itertools.count()

        def call():
            i = next(calls)
            ctx = types.SimpleNamespace(usage=Usage(), agent=agent, turn_input=[],
                                        tool_call_id=f"c{i}", tool_name=tool.name, context=None)
            with agent_task_scope("agent", f"run-{i // 20}"):
                loop.run_until_complete(tool.on_invoke_tool(ctx, args))

        return call

    base = p99_ms(caller(raw_tool), n=5_000, rounds=3, label=f"raw {which} tool call")
    with Ledger(tmp_path / "ledger.db") as ledger:
        lookup, save = instrument_tools(tools(), Capture(ledger), read_only=["lookup"])
        p99 = p99_ms(caller(lookup if which == "read" else save), n=5_000, rounds=3,
                     label=f"instrumented {which} tool call (runs of 20 calls, on-disk)")
    loop.close()
    print(f"{which} tool call overhead p99 ~ {p99 - base:.4f} ms")
    assert p99 - base < 1.0
