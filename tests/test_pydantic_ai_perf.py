import itertools
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic_ai")
pytestmark = pytest.mark.pydantic_ai
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from perf import p99_ms  # noqa: E402
from pydantic_ai.messages import ModelRequest, ToolReturnPart  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.integrations.pydantic_ai import CaptureToolset  # noqa: E402
from dod.ledger import Ledger  # noqa: E402


class Fake:
    """A toolset whose calls complete without awaiting, so a call is timed
    as capture's own work."""

    async def call_tool(self, name, tool_args, ctx, tool):
        return {"id": "T1"}


def drive(coro):
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("coroutine awaited")


# A run 10 read results deep: every call scans them into the context.
HISTORY = [
    ModelRequest(parts=[ToolReturnPart("lookup", f"result {i}", tool_call_id=f"h{i}")],
                 run_id="earlier")
    for i in range(10)
]


@pytest.mark.parametrize("tool", ["write", "lookup"])
def test_call_tool_overhead_p99_under_1ms(tmp_path, tool):
    ids = itertools.count()

    def ctx():
        return SimpleNamespace(run_id="run-1", tool_call_id=f"c{next(ids)}",
                               messages=HISTORY, agent=None)

    base = p99_ms(lambda: drive(Fake().call_tool(tool, {"x": 1}, ctx(), None)), n=5_000,
                  label="raw")
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger)
        ts = CaptureToolset(Fake(), cap, read_only=["lookup"])

        def call():
            # Start each call from the same context, as one bounded run would.
            cap.reset(("pydantic_ai", "run-1"))
            drive(ts.call_tool(tool, {"x": 1}, ctx(), None))

        p99 = p99_ms(call, n=5_000, rounds=3,
                     label=f"instrumented {tool} call (10 history parents, on-disk)")
    assert p99 - base < 1.0
