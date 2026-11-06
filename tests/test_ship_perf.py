"""Capture latency with ``dod ship`` copying to Postgres from its own
process (the recommended deployment: no contention for the interpreter)."""

import os
import subprocess
import sys
import threading
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


def test_shipping_adds_under_half_ms_to_capture_p99(tmp_path, prefix):  # noqa: F811
    path = tmp_path / "ledger.db"
    edges = [("mem", f"rec-{i}", "retrieved_in_scope") for i in range(10)]
    with Ledger(path) as ledger:
        def append():
            ledger.append_write(content=str(uuid.uuid4()), store="mem",
                                source="agent_inference", agent_id="a", session_id="s",
                                context=edges)

        base = p99_ms(append, n=5_000, rounds=3, label="append_write (10 context), no shipper")
        w = ledger.append_write(content="seed", store="mem", source="user")
        ledger.bind(w, "mem", "seed")
        proc = subprocess.Popen(
            [sys.executable, "-m", "dod", "ship", "--ledger", str(path), "--dsn", DSN,
             "--node", "perf", "--prefix", prefix, "--interval", "0.25"],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            # Low priority, as recommended: on a 2-core CI runner a pass at
            # normal priority takes the CPU the agent's writes need.
            preexec_fn=lambda: os.nice(19))
        try:
            # Steady state: the shipper has started and shipped (its start-up,
            # imports and schema DDL, isn't what this measures).
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    PgLineage(DSN, prefix).trace("mem", "seed")
                    break
                except (KeyError, psycopg.errors.UndefinedTable):  # not yet
                    time.sleep(0.1)
            else:
                pytest.fail("dod ship shipped nothing")
            # Meanwhile another node quarantines items, so this node's shipper
            # also appends pulled quarantine rows to the ledger during capture.
            from dod.ship import Shipper
            other = Ledger(tmp_path / "other.db")
            remote = Shipper(other, DSN, "other", prefix=prefix, pull=False)
            stop = threading.Event()

            def quarantine_elsewhere():
                n = 0
                while not stop.wait(0.2):
                    n += 1
                    other.quarantine([("mem", f"rec-{n}")])
                    remote.ship_once()

            t = threading.Thread(target=quarantine_elsewhere, daemon=True)
            t.start()
            try:
                p99 = p99_ms(append, n=5_000, rounds=3,
                             label="append_write (10 context) while `dod ship` (nice 19) runs"
                                   " every 250 ms and pulls quarantines")
            finally:
                stop.set()
                t.join()
                other.close()
            deadline = time.time() + 30  # the shipper's next pass pulls them
            while not ledger.quarantined() and time.time() < deadline:
                time.sleep(0.1)
            assert ledger.quarantined(), "no quarantine was pulled"
        finally:
            proc.terminate()
            assert proc.wait(timeout=30) == 0
    # What the shipper adds to capture, on the same machine: on a 2-core CI
    # runner with Postgres also running, the absolute p99 swings with load.
    print(f"shipping adds p99 ~ {p99 - base:.4f} ms")
    assert p99 - base < 0.5
