import json
import subprocess
import sys

import pytest

from dod.cli import main
from dod.ledger import Ledger


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "ledger.db"
    with Ledger(p) as led:
        def put(rid, parents=(), source="agent_inference"):
            w = led.append_write(content=rid, store="mem", source=source, parents=parents)
            led.bind(w, "mem", rid)
        put("bad", source="user")
        put("good", source="document")
        put("d1", [("mem", "bad", "in_prompt"), ("mem", "good", "retrieved_in_scope")])
        put("d2", [("mem", "d1", "retrieved_in_scope")])
    return str(p)


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_trace_text_and_json(path, capsys):
    code, out = run(capsys, "trace", "mem", "d2", "--ledger", path)
    assert code == 0 and "3 ancestor write(s)" in out and "source: user" in out
    _, out = run(capsys, "trace", "mem", "d2", "--ledger", path, "--json")
    data = json.loads(out)
    assert {w["source"] for w in data["roots"]} == {"user", "document"}


def test_blast_radius_and_in_prompt_only(path, capsys):
    _, out = run(capsys, "blast-radius", "mem", "bad", "--ledger", path, "--json")
    assert json.loads(out)["items"] == [["mem", "d1"], ["mem", "d2"]]
    _, out = run(capsys, "blast-radius", "mem", "bad", "--ledger", path, "--json",
                 "--in-prompt-only")
    assert json.loads(out)["items"] == [["mem", "d1"]]  # d2 only retrieved d1


def test_excise_quarantine_release(path, capsys, monkeypatch):
    monkeypatch.setenv("DOD_LEDGER", path)
    _, out = run(capsys, "excise", "mem", "bad", "--dry-run")
    assert "would quarantine 3 item(s)" in out
    _, out = run(capsys, "quarantined", "--json")
    assert json.loads(out)["quarantined"] == []
    _, out = run(capsys, "excise", "mem", "bad")
    assert "quarantined 3 item(s)" in out and "undo with" in out
    _, out = run(capsys, "quarantined", "--json")
    assert json.loads(out)["quarantined"] == [["mem", "bad"], ["mem", "d1"], ["mem", "d2"]]
    run(capsys, "release", "mem", "d1", "d2", "--reason", "false alarm")
    _, out = run(capsys, "quarantined", "--json")
    assert json.loads(out)["quarantined"] == [["mem", "bad"]]


def test_stats(path, capsys):
    _, out = run(capsys, "stats", "--ledger", path, "--json")
    data = json.loads(out)
    assert data["writes"] == {"mem": 4} and data["items"] == {"mem": 4}
    assert data["sources"]["user"] == 1


def test_errors(path, capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("DOD_LEDGER", raising=False)
    with pytest.raises(SystemExit, match="no ledger given"):
        main(["stats"])
    with pytest.raises(SystemExit, match="no ledger at"):
        main(["stats", "--ledger", str(tmp_path / "nope.db")])
    with pytest.raises(SystemExit, match="no recorded write"):
        main(["trace", "mem", "missing", "--ledger", path])


def test_python_dash_m(path):
    out = subprocess.run([sys.executable, "-m", "dod", "stats", "--ledger", path],
                         capture_output=True, text=True, check=True).stdout
    assert "writes: 4" in out


def test_release_refuses_items_not_quarantined(path, capsys):
    run(capsys, "excise", "mem", "d2", "--ledger", path)
    with pytest.raises(SystemExit, match="not quarantined: mem typo"):
        main(["release", "mem", "d2", "typo", "--ledger", path])
    _, out = run(capsys, "quarantined", "--ledger", path, "--json")
    assert json.loads(out)["quarantined"] == [["mem", "d2"]]  # nothing released


def test_stats_does_not_write(path, capsys):
    import os
    before = {f: os.stat(f).st_mtime_ns for f in (path, path + "-lineage") if os.path.exists(f)}
    run(capsys, "stats", "--ledger", path)
    assert {f: os.stat(f).st_mtime_ns for f in before} == before


def test_max_depth_and_unknown_store(path, capsys):
    _, out = run(capsys, "trace", "mem", "d2", "--ledger", path, "--max-depth", "1", "--json")
    assert json.loads(out)["truncated"]
    _, out = run(capsys, "blast-radius", "mem", "bad", "--ledger", path, "--max-depth", "1",
                 "--json")
    data = json.loads(out)
    assert data["truncated"] and data["items"] == [["mem", "d1"]]
    with pytest.raises(SystemExit):
        main(["excise", "nostore", "bad", "--ledger", path])


@pytest.fixture
def audited(tmp_path):
    p = tmp_path / "audited.db"
    with Ledger(p, audit=True) as led:
        w = led.append_write(content="bad", store="mem", source="user")
        led.bind(w, "mem", "bad")
        w = led.append_write(content="d1", store="mem", source="agent_inference",
                             parents=[("mem", "bad", "in_prompt")])
        led.bind(w, "mem", "d1")
    return str(p)


def test_verify_and_report(audited, capsys):
    run(capsys, "excise", "mem", "bad", "--ledger", audited)
    code, out = run(capsys, "verify", "--ledger", audited, "--json")
    data = json.loads(out)
    assert code == 0 and data["ok"] and data["entries"] == 5  # CLI writes stay chained
    code, out = run(capsys, "verify", "--ledger", audited, "--expect-head", data["head"])
    assert code == 0 and out.startswith("verified")
    _, out = run(capsys, "report", "mem", "bad", "--ledger", audited)
    assert out.startswith("# Excision report: mem bad") and "quarantined 2" in out
    _, out = run(capsys, "report", "mem", "bad", "--ledger", audited, "--json")
    assert json.loads(out)["summary"]["quarantined"] == 2
    with pytest.raises(SystemExit, match="no recorded write"):
        main(["report", "mem", "nope", "--ledger", audited])


def test_verify_fails_on_tampering(audited, capsys):
    import sqlite3
    conn = sqlite3.connect(audited, isolation_level=None)
    conn.execute("DROP TRIGGER writes_no_update")
    conn.execute("UPDATE writes SET source = 'document' WHERE seq = 1")
    conn.close()
    code, out = run(capsys, "verify", "--ledger", audited)
    assert code == 1 and out.startswith("NOT VERIFIED") and "digest mismatch" in out


def test_ship_final_sync_failure_exits_1_without_traceback(path, capsys, monkeypatch):
    """The loop is interrupted, then the final sync fails: message on stderr, exit 1."""
    import signal

    import dod.ship

    calls = []

    class FailingShipper:
        def __init__(self, *a, **kw):
            pass

        def sync(self):
            calls.append(1)
            if len(calls) == 1:
                raise KeyboardInterrupt  # stops the loop, like Ctrl-C
            raise RuntimeError("connection refused")

    monkeypatch.setattr(dod.ship, "Shipper", FailingShipper)
    old = signal.getsignal(signal.SIGTERM)
    try:
        code = main(["ship", "--ledger", path, "--dsn", "postgresql://unused", "--node", "n",
                     "--interval", "0.01"])
    finally:
        signal.signal(signal.SIGTERM, old)
    err = capsys.readouterr().err
    assert code == 1
    assert len(calls) == 2
    assert "dod: final ship failed: connection refused; the next run ships it" in err
    assert "Traceback" not in err
def test_precise_flag(path, capsys):
    code, out = run(capsys, "blast-radius", "mem", "bad", "--ledger", path, "--precise", "--json")
    assert code == 0 and "items" in json.loads(out)
    code, _ = run(capsys, "excise", "mem", "bad", "--ledger", path, "--precise", "--dry-run")
    assert code == 0
