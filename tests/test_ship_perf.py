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

# Another node: quarantine an item and ship it, every 200 ms, until killed.
OTHER_NODE = """
import sys, time
from dod.ledger import Ledger
from dod.ship import Shipper
led = Ledger(sys.argv[1])
s = Shipper(led, sys.argv[2], "other", prefix=sys.argv[3], pull=False)
n = 0
while True:
    n += 1
    led.quarantine([("mem", f"rec-{n}")])
    s.ship_once()
    time.sleep(0.2)
"""


def test_shipping_adds_under_half_ms_to_capture_p99(tmp_path, prefix):  # noqa: F811
    # Both measurements run while another node (its own process) quarantines
    # items and ships them, so Postgres is busy in both: on a 2-core CI
    # runner, Postgres itself competes for the CPU, which is load from the
    # test machine, not from shipping. The difference is what this node's
    # `dod ship` adds, while it ships capture's writes and pulls the other
    # node's quarantines into this ledger.
    path = tmp_path / "ledger.db"
    edges = [("mem", f"rec-{i}", "retrieved_in_scope") for i in range(10)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    other = subprocess.Popen([sys.executable, "-c", OTHER_NODE, str(tmp_path / "other.db"),
                              DSN, prefix], env=env)
    try:
        with Ledger(path) as ledger:
            def append():
                ledger.append_write(content=str(uuid.uuid4()), store="mem",
                                    source="agent_inference", agent_id="a", session_id="s",
                                    context=edges)

            base = p99_ms(append, n=5_000, rounds=3,
                          label="append_write (10 context), no shipper on this node")
            w = ledger.append_write(content="seed", store="mem", source="user")
            ledger.bind(w, "mem", "seed")
            proc = subprocess.Popen(
                [sys.executable, "-m", "dod", "ship", "--ledger", str(path), "--dsn", DSN,
                 "--node", "perf", "--prefix", prefix],  # the default 1 s interval
                env=env, preexec_fn=lambda: os.nice(19))  # low priority, as recommended
            try:
                # Steady state: its start-up (imports, schema DDL) isn't measured.
                deadline = time.time() + 30
                while time.time() < deadline:
                    try:
                        PgLineage(DSN, prefix).trace("mem", "seed")
                        break
                    except (KeyError, psycopg.errors.UndefinedTable):  # not yet
                        time.sleep(0.1)
                else:
                    pytest.fail("dod ship shipped nothing")
                p99 = p99_ms(append, n=5_000, rounds=3,
                             label="append_write (10 context) while `dod ship` (nice 19)"
                                   " ships and pulls")
                deadline = time.time() + 30  # its next pass pulls the quarantines
                while not ledger.quarantined() and time.time() < deadline:
                    time.sleep(0.1)
                assert ledger.quarantined(), "no quarantine was pulled"
            finally:
                proc.terminate()
                assert proc.wait(timeout=30) == 0
    finally:
        other.terminate()
        other.wait(timeout=30)
    print(f"shipping adds p99 ~ {p99 - base:.4f} ms")
    assert p99 - base < 0.5
