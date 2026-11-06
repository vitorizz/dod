"""Capture latency with ``dod ship`` copying to Postgres from its own
process (the recommended deployment: no contention for the interpreter)."""

import os
import subprocess
import sys
import time
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(
    not os.environ.get("DOD_TEST_POSTGRES_URI"), reason="DOD_TEST_POSTGRES_URI not set")]

from perf import p99_ms  # noqa: E402
from test_ship import prefix  # noqa: E402,F401

from dod.ledger import Ledger  # noqa: E402
from dod.ship import PgLineage  # noqa: E402

DSN = os.environ.get("DOD_TEST_POSTGRES_URI", "")


def test_append_while_shipping_p99_under_1ms(tmp_path, prefix):  # noqa: F811
    path = tmp_path / "ledger.db"
    edges = [("mem", f"rec-{i}", "retrieved_in_scope") for i in range(10)]
    with Ledger(path) as ledger:
        w = ledger.append_write(content="seed", store="mem", source="user")
        ledger.bind(w, "mem", "seed")
        proc = subprocess.Popen(
            [sys.executable, "-m", "dod", "ship", "--ledger", str(path), "--dsn", DSN,
             "--node", "perf", "--prefix", prefix, "--interval", "0.1"],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        try:
            def append():
                ledger.append_write(content=str(uuid.uuid4()), store="mem",
                                    source="agent_inference", agent_id="a", session_id="s",
                                    context=edges)

            p99 = p99_ms(append, n=5_000, rounds=3,
                         label="append_write (10 context) while `dod ship` runs every 100 ms")
            deadline = time.time() + 10
            while time.time() < deadline:  # it really shipped
                try:
                    PgLineage(DSN, prefix).trace("mem", "seed")
                    break
                except (KeyError, psycopg.errors.UndefinedTable):  # not yet
                    time.sleep(0.1)
            else:
                pytest.fail("dod ship shipped nothing")
        finally:
            proc.terminate()
            assert proc.wait(timeout=30) == 0
    assert p99 < 1.0
