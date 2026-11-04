from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")
pytestmark = pytest.mark.mcp

from mcp import types  # noqa: E402
from perf import p99_ms  # noqa: E402

from dod import scope  # noqa: E402
from dod.capture import Capture  # noqa: E402
from dod.integrations import mcp as dmcp  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402

RESULT = types.CallToolResult(content=[types.TextContent(type="text", text="ok")])


class Fake:
    """A session whose calls complete without awaiting, so a call is timed
    as capture's own work."""

    async def call_tool(self, name, arguments=None):
        return RESULT

    async def read_resource(self, uri):
        return None

    async def list_tools(self):
        return types.ListToolsResult(tools=[types.Tool(
            name="lookup", inputSchema={"type": "object"},
            annotations=types.ToolAnnotations(readOnlyHint=True))])


def drive(coro):
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("coroutine awaited")


def _session(ledger):
    s = dmcp.instrument(Fake(), Capture(ledger), "crm")
    drive(s.list_tools())
    return s


@pytest.mark.parametrize("tool", ["write", "lookup"])
def test_call_tool_overhead_p99_under_1ms(tmp_path, tool):
    # A write: record before, bind after (two commits). A read: one commit
    # (the call item, bound in the same transaction). 10 context parents.
    # The absolute p99 is asserted (the raw fake call is microseconds), not
    # a difference of two p99s taken at different times.
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger)
        s = dmcp.instrument(Fake(), cap, "crm")
        drive(s.list_tools())
        docs = [("mcp:crm", f"doc://{i}") for i in range(10)]

        def call():
            # Each call adds its own item to the context; start every call
            # from the same 10 parents, as a bounded unit of work would.
            cap.reset(key)
            cap.on_retrieve(key, docs)
            drive(s.call_tool(tool, {"x": 1}))

        with agent_task_scope("agent", "task") as key:
            p99 = p99_ms(call, n=5_000, rounds=3, label=f"instrumented {tool} call (on-disk)")
    assert p99 < 1.0


class _Conn:
    """Stands in for a client connection's server session (weakref-able)."""


# 500 matches the ledger's growing-context benchmark. Cost grows linearly
# with the context: ~0.9 ms p99 at 1,000 on a dev machine, too close to the
# bar to assert on shared CI runners (see docs/decisions.md).
@pytest.mark.parametrize("size", [500])
def test_gateway_long_connection_growing_context_p99_under_1ms(tmp_path, size):
    # The gateway path without the MCP transport: Gateway._scope (the
    # connection's key, via use_key, never reset between calls) around the
    # instrumented upstream call. Every call adds its result to the
    # connection's context, so each write's context is one item larger than
    # the last, up to ``size`` items; then the connection ends (its key is
    # reset, as when its session is collected) and a new one starts empty.
    # Alternate writes and read-only calls, as an agent would.
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger)
        gw = dmcp.Gateway(cap, {"crm": Fake()})
        upstream = gw.upstreams["crm"]
        drive(upstream.list_tools())
        state = {"conn": _Conn(), "calls": 0}

        def call():
            if state["calls"] == size:
                scope.reset(gw._keys[state["conn"]])
                state["conn"], state["calls"] = _Conn(), 0
            gw.server = SimpleNamespace(request_context=SimpleNamespace(
                meta=None, session=state["conn"]))
            tool = "write" if state["calls"] % 2 else "lookup"
            with gw._scope():
                drive(upstream.call_tool(tool, {"x": state["calls"]}))
            state["calls"] += 1

        p99 = p99_ms(call, n=5_000, warmup=size, rounds=3,
                     label=f"gateway call (connection context growing to {size}, on-disk)")
        # The context really grew: the last write had the connection's items.
        key = gw._keys[state["conn"]]
        assert len(cap._context[key]) == state["calls"]
    assert p99 < 1.0


def test_read_resource_overhead_p99_under_1ms(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        s = _session(ledger)
        with agent_task_scope("agent", "task"):
            p99 = p99_ms(lambda: drive(s.read_resource("doc://1")), rounds=3,
                         label="instrumented read_resource")
    assert p99 < 1.0
