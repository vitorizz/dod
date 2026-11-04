import pytest

pytest.importorskip("mcp")
pytestmark = pytest.mark.mcp

from mcp import types  # noqa: E402
from perf import p99_ms  # noqa: E402

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
    base = p99_ms(lambda: drive(Fake().call_tool(tool, {"x": 1})), n=5_000, label="raw")
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
    assert p99 - base < 1.0


def test_read_resource_overhead_p99_under_1ms(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        s = _session(ledger)
        with agent_task_scope("agent", "task"):
            p99 = p99_ms(lambda: drive(s.read_resource("doc://1")), rounds=3,
                         label="instrumented read_resource")
    assert p99 < 1.0
