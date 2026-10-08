import itertools
import time

import pytest

pytest.importorskip("anthropic.tools.memory")
pytestmark = pytest.mark.anthropic

from anthropic.tools.memory import BetaLocalFilesystemMemoryTool  # noqa: E402
from perf import split_p99_ms  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.integrations.anthropic_memory import instrument  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402

# The tool's own work (file I/O, an fsync per edit) varies far more than
# capture does, so each call is split: the raw tool's time is measured inside
# the instrumented call, and the rest is the overhead capture adds.


class Timed(BetaLocalFilesystemMemoryTool):
    """Records the nanoseconds each view/str_replace spends in the raw tool."""

    def __init__(self, base_path, inner):
        super().__init__(base_path=base_path)
        self.inner = inner
        self.timing = True

    def view(self, command):
        start = time.perf_counter_ns()
        try:
            return super().view(command)
        finally:
            if self.timing:
                self.inner.append(time.perf_counter_ns() - start)

    def str_replace(self, command):
        start = time.perf_counter_ns()
        try:
            return super().str_replace(command)
        finally:
            self.inner.append(time.perf_counter_ns() - start)


def _seeded(tool):
    tool.timing = False
    for i in range(10):
        tool.call({"command": "create", "path": f"/memories/facts/f{i}.md",
                   "file_text": f"fact {i}\n" * 20})
    tool.call({"command": "create", "path": "/memories/edit.md", "file_text": "v0\n"})
    tool.timing = True
    return tool


def test_view_overhead_p99_under_1ms(tmp_path):
    inner = []
    with Ledger(tmp_path / "ledger.db") as ledger:
        tool = instrument(_seeded(Timed(str(tmp_path / "mem"), inner)), Capture(ledger))
        with agent_task_scope("agent", "task"):
            overhead, raw = split_p99_ms(
                lambda: tool.call({"command": "view", "path": "/memories/facts/f3.md"}),
                inner, rounds=3, labels=("capture overhead (view, on-disk)", "raw view"),
            )
    print(f"view overhead p99 ~ {overhead:.4f} ms (raw view p99 {raw:.4f} ms)")
    assert overhead < 1.0


def test_str_replace_overhead_p99_under_1ms(tmp_path):
    # Each call: a compaction commit before the edit (the file plus 10 viewed
    # files as parents), then a bind commit after it.
    inner = []
    with Ledger(tmp_path / "ledger.db") as ledger:
        tool = instrument(_seeded(Timed(str(tmp_path / "mem"), inner)), Capture(ledger))
        versions = itertools.count()

        def run():
            i = next(versions)
            tool.call({"command": "str_replace", "path": "/memories/edit.md",
                       "old_str": f"v{i}\n", "new_str": f"v{i + 1}\n"})

        with agent_task_scope("agent", "task"):
            tool.timing = False
            for i in range(10):
                tool.call({"command": "view", "path": f"/memories/facts/f{i}.md"})
            overhead, raw = split_p99_ms(
                run, inner, n=2_000, warmup=200, rounds=3,
                labels=("capture overhead (str_replace, on-disk)", "raw str_replace"),
            )
    print(f"str_replace overhead p99 ~ {overhead:.4f} ms (raw str_replace p99 {raw:.4f} ms)")
    assert overhead < 1.0
