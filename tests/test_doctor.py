import json
import sqlite3

import pytest

from dod.cli import main
from dod.doctor import check
from dod.ledger import Ledger


def levels(findings):
    return {f.check: f.level for f in findings}


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "ledger.db"
    with Ledger(p) as led:
        w = led.append_write(content="a", store="mem", source="user")
        led.bind(w, "mem", "a")
        led.append_write(content="b", store="crm", source="agent_inference")  # never bound
        led.quarantine([("mem", "a")])
    return str(p)


def test_healthy_but_unbound(path):
    f = check(path, unbound_after=0)
    assert levels(f) == {"schema": "ok", "wal": "ok", "index": "ok", "unbound": "warn",
                         "audit": "ok", "quarantine": "ok"}
    (u,) = [x for x in f if x.check == "unbound"]
    assert u.value == {"crm": 1}
    (q,) = [x for x in f if x.check == "quarantine"]
    assert q.value == 1
    # Recent unbound writes may still be in flight: not counted by default.
    assert levels(check(path))["unbound"] == "ok"


def test_audited_and_tampered(tmp_path):
    p = tmp_path / "a.db"
    with Ledger(p, audit=True) as led:
        led.append_write(content="a", store="mem", source="user")
    assert levels(check(str(p)))["audit"] == "ok"
    conn = sqlite3.connect(p, isolation_level=None)
    conn.execute("DROP TRIGGER writes_no_update")
    conn.execute("UPDATE writes SET source = 'tool'")
    conn.close()
    f = check(str(p))
    assert levels(f)["audit"] == "error"


def test_schema_problems(tmp_path):
    assert levels(check(str(tmp_path / "missing.db"))) == {"schema": "error"}
    bad = tmp_path / "bad.db"
    conn = sqlite3.connect(bad)
    conn.execute("CREATE TABLE t (x)")
    conn.execute("PRAGMA user_version=99")
    conn.close()
    assert levels(check(str(bad))) == {"schema": "error"}
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a database at all" * 100)
    assert levels(check(str(junk))) == {"schema": "error"}


def test_cli(path, capsys):
    assert main(["doctor", "--ledger", path, "--unbound-after", "0"]) == 0
    out = capsys.readouterr().out
    assert "WARN  unbound: 1 write(s) never bound" in out and "ok    schema" in out
    assert main(["doctor", "--ledger", path, "--unbound-after", "0", "--strict"]) == 1
    capsys.readouterr()
    assert main(["doctor", "--ledger", path, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {f["check"] for f in data["findings"]} >= {"schema", "wal", "index", "audit"}


def test_old_schema_needs_migration(path):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("DROP TABLE audit_chain")
    conn.execute("PRAGMA user_version=5")
    conn.close()
    f = check(path)
    assert [(x.check, x.level) for x in f] == [("schema", "warn")]
    assert "migrates it to 6" in f[0].message
