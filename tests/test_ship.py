import os
import random
import uuid

import pytest

pytest.importorskip("psycopg")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(
    not os.environ.get("DOD_TEST_POSTGRES_URI"), reason="DOD_TEST_POSTGRES_URI not set")]

from dod.ledger import Ledger  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.ship import PgLineage, Shipper  # noqa: E402

DSN = os.environ.get("DOD_TEST_POSTGRES_URI", "")


@pytest.fixture
def prefix():
    """Each test gets its own tables (CI shares one database)."""
    p = f"t{uuid.uuid4().hex[:10]}_"
    yield p
    import psycopg
    with psycopg.connect(DSN, autocommit=True) as pg:
        tables = [r[0] for r in pg.execute(
            "SELECT tablename FROM pg_tables WHERE tablename LIKE %s", (p + "%",))]
        pg.execute(f"DROP VIEW IF EXISTS {p}parents")
        for t in tables:
            pg.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
        pg.execute(f"DROP FUNCTION IF EXISTS {p}append_only() CASCADE")


def write(led, store, rid, parents=(), context=()):
    w = led.append_write(content=f"{store}:{rid}:{uuid.uuid4()}", store=store,
                         source="agent_inference", parents=list(parents),
                         context=list(context))
    led.bind(w, store, rid)
    return w


def test_lineage_spans_nodes(tmp_path, prefix):
    a, b = Ledger(tmp_path / "a.db"), Ledger(tmp_path / "b.db")
    sa = Shipper(a, DSN, "host-a", prefix=prefix)
    sb = Shipper(b, DSN, "host-b", prefix=prefix)
    x = write(a, "mem", "x")
    y = write(b, "crm", "y", context=[("mem", "x", "in_prompt")])  # host b read x
    z = write(a, "mem", "z", context=[("crm", "y", "retrieved_in_scope")])
    unrelated = write(b, "crm", "u")
    assert sa.sync() > 0 and sb.sync() > 0
    lin = PgLineage(DSN, prefix)
    blast = lin.blast_radius("mem", "x")
    assert blast["items"] == {("crm", "y"), ("mem", "z")}
    assert blast["writes"] == {y, z}
    assert lin.blast_radius("mem", "x", in_prompt_only=True)["items"] == {("crm", "y")}
    t = lin.trace("mem", "z")
    assert t["writes"] == {z} and t["ancestors"] == {x, y} and t["roots"] == {x}
    assert unrelated not in blast["writes"]
    with pytest.raises(KeyError):
        lin.blast_radius("mem", "nope")
    # Re-shipping is a no-op.
    assert sa.sync() == 0 and sb.sync() == 0
    a.close()
    b.close()


def test_quarantine_reaches_every_node(tmp_path, prefix):
    a, b = Ledger(tmp_path / "a.db"), Ledger(tmp_path / "b.db")
    sa = Shipper(a, DSN, "host-a", prefix=prefix)
    sb = Shipper(b, DSN, "host-b", prefix=prefix)
    write(a, "mem", "x")
    a.quarantine([("mem", "x")], reason="false", actor="alice")
    sa.sync()
    sb.sync()
    assert b.quarantined() == {("mem", "x")}
    # The pulled row isn't shipped back; a release on b reaches a.
    b.release([("mem", "x")], actor="bob")
    sb.sync()
    sa.sync()
    assert a.quarantined() == set()
    lin = PgLineage(DSN, prefix)
    assert lin.quarantined() == set()
    import psycopg
    with psycopg.connect(DSN) as pg:
        actors = sorted(r[0] for r in pg.execute(f"SELECT actor FROM {prefix}quarantine"))
    assert actors == ["alice", "bob"]
    a.close()
    b.close()


def test_matches_local_queries_on_a_random_graph(tmp_path, prefix):
    rng = random.Random(7)
    led = Ledger(tmp_path / "l.db")
    items = []
    for i in range(300):
        ctx = [("mem", r, rng.choice(["in_prompt", "retrieved_in_scope"]))
               for r in rng.sample(items, min(len(items), rng.randrange(0, 4)))]
        parents = [("mem", rng.choice(items), "in_prompt")] if items and rng.random() < .2 else []
        rid = f"r{i}" if rng.random() < .85 or not items else rng.choice(items)  # some versions
        write(led, "mem", rid, parents=parents, context=ctx)
        if rid not in items:
            items.append(rid)
    s = Shipper(led, DSN, "solo", prefix=prefix, batch=97)  # several batches
    s.sync()
    lin = PgLineage(DSN, prefix)
    for rid in rng.sample(items, 25):
        for only in (False, True):
            local = blast_radius(led, "mem", rid, in_prompt_only=only)
            got = {i for its in local.bindings.values() for i in its} - {("mem", rid)}
            assert lin.blast_radius("mem", rid, in_prompt_only=only)["items"] == got, rid
        lt = trace(led, "mem", rid)
        assert lin.trace("mem", rid)["ancestors"] == {w.write_id for w in lt.ancestors}, rid
    led.close()


def test_append_only_and_validation(tmp_path, prefix):
    led = Ledger(tmp_path / "l.db")
    Shipper(led, DSN, "n", prefix=prefix).sync()
    write(led, "mem", "x")
    Shipper(led, DSN, "n", prefix=prefix).sync()
    import psycopg
    with psycopg.connect(DSN) as pg:
        with pytest.raises(psycopg.errors.RaiseException):
            pg.execute(f"DELETE FROM {prefix}writes")
    with pytest.raises(ValueError):
        Shipper(led, DSN, "dod-ship:x", prefix=prefix)
    with pytest.raises(ValueError):
        Shipper(Ledger(":memory:"), DSN, "n", prefix=prefix)
    led.close()


def test_background_thread_ships(tmp_path, prefix):
    import time
    led = Ledger(tmp_path / "l.db")
    with Shipper(led, DSN, "bg", prefix=prefix, interval=0.05):
        write(led, "mem", "x")
        write(led, "mem", "y", context=[("mem", "x", "in_prompt")])
        deadline = time.time() + 5
        lin = PgLineage(DSN, prefix)
        while time.time() < deadline:
            try:
                if lin.blast_radius("mem", "x")["items"] == {("mem", "y")}:
                    break
            except KeyError:
                pass
            time.sleep(0.05)
        else:
            pytest.fail("not shipped")
    led.close()


def test_cli_ship_once_pulls_quarantine_into_another_process_ledger(tmp_path, prefix, capsys):
    from dod.cli import main
    a, b = Ledger(tmp_path / "a.db"), Ledger(tmp_path / "b.db")
    write(a, "mem", "x")
    a.quarantine([("mem", "x")])
    a.close()
    b.close()
    for path, node in ((tmp_path / "a.db", "a"), (tmp_path / "b.db", "b")):
        assert main(["ship", "--ledger", str(path), "--dsn", DSN, "--node", node,
                     "--prefix", prefix, "--once", "--json"]) == 0
    with Ledger(tmp_path / "b.db") as b:
        assert b.quarantined() == {("mem", "x")}
    with pytest.raises(SystemExit, match="DSN"):
        main(["ship", "--ledger", str(tmp_path / "a.db"), "--node", "a"])
