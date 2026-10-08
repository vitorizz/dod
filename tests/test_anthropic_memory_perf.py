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


class TimedWithRead(Timed):
    """Adds the view that follows each str_replace (the hash read) to that
    call's raw time, so one raw time is recorded per call."""

    def view(self, command):
        start = time.perf_counter_ns()
        try:
            return BetaLocalFilesystemMemoryTool.view(self, command)
        finally:
            if self.timing:
                self.inner[-1] += time.perf_counter_ns() - start


def test_str_replace_overhead_with_file_hashes_p99_under_1ms(tmp_path):
    # As above, plus hash_files=True: after each edit the file is read back
    # through the tool's view and hashed. The read is the store's own work
    # (I/O that varies like the edit's), so it is subtracted with the raw
    # edit; what remains is capture's cost, parsing and hashing included.
    # The opt-in's full price is that plus one view, printed below.
    inner = []
    with Ledger(tmp_path / "ledger.db") as ledger:
        tool = instrument(_seeded(TimedWithRead(str(tmp_path / "mem"), inner)),
                          Capture(ledger), hash_files=True)
        versions = itertools.count()

        def run():
            i = next(versions)
            tool.call({"command": "str_replace", "path": "/memories/edit.md",
                       "old_str": f"v{i}\n", "new_str": f"v{i + 1}\n"})

        with agent_task_scope("agent", "task"):
            tool.timing = False
            for i in range(10):
                tool.call({"command": "view", "path": f"/memories/facts/f{i}.md"})
            tool.timing = True  # the hash read counts as the tool's work
            overhead, raw = split_p99_ms(
                run, inner, n=2_000, warmup=200, rounds=3,
                labels=("capture overhead (str_replace + file hash, on-disk)",
                        "raw str_replace + hash read"),
            )
    print(f"str_replace + hash overhead p99 ~ {overhead:.4f} ms"
          f" (raw edit + read p99 {raw:.4f} ms)")
    assert overhead < 1.0
