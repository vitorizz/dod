import itertools
import uuid

import pytest

pytest.importorskip("langgraph.graph")
pytestmark = pytest.mark.langgraph

from perf import p99_ms  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.integrations.langgraph_graph import _StateRecorder  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402


def test_node_update_capture_p99_under_1ms(tmp_path):
    # What a node finishing costs: its update (2 channels, one overwriting a
    # loaded channel) recorded and bound in one commit, with the thread's
    # 10 loaded channels as context.
    with Ledger(tmp_path / "ledger.db") as ledger:
        cap = Capture(ledger)
        key = ("bot", "t1")
        rec = _StateRecorder(cap, "langgraph_state", key, "t1", "")
        loaded = [rec.item(f"c{i}") for i in range(9)] + [rec.item("messages")]
        root = uuid.uuid4()
        rec.root = root
        steps = itertools.count()

        def node():
            cap.reset(key)
            cap.on_retrieve(key, loaded)
            run = uuid.uuid4()
            rec.on_chain_start({}, {}, run_id=run, parent_run_id=root,
                               metadata={"langgraph_node": "agent"})
            rec.on_chain_end({"messages": [f"reply {next(steps)}"], "note": "n"}, run_id=run)

        with agent_task_scope(*key):
            p99 = p99_ms(node, n=5_000, rounds=3, label="node update capture (on-disk)")
    assert p99 < 1.0
