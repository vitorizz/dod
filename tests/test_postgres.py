"""The LangGraph integration against a real Postgres (``PostgresStore``,
with pgvector search). Needs a server: set ``DOD_TEST_POSTGRES_URI`` (a
database the tests may write to, with the ``vector`` extension available)."""

import hashlib
import json
import os

import pytest

pytest.importorskip("langgraph.store.postgres")
URI = os.environ.get("DOD_TEST_POSTGRES_URI")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not URI, reason="DOD_TEST_POSTGRES_URI not set"),
]

from langgraph.store.postgres import PostgresStore  # noqa: E402

import dod  # noqa: E402
from dod.capture import Capture  # noqa: E402
from dod.integrations import crewai, langgraph  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.reconcile import reconcile  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402

S = "postgres"
rid = langgraph.record_id


def embed(texts):
    """Bag of words hashed into 16 dims: deterministic, no model."""
    out = []
    for text in texts:
        v = [0.0] * 16
        for word in text.lower().split():
            v[hashlib.sha256(word.encode()).digest()[0] % 16] += 1.0
        out.append(v)
    return out


@pytest.fixture
def store(request):
    ns = ("dodtest", request.node.name)
    with PostgresStore.from_conn_string(
        URI, index={"dims": 16, "embed": embed, "fields": ["text"]}
    ) as s:
        s.setup()
        _clear(s, ns)
        yield s, ns
        _clear(s, ns)  # with SQL: the ledger may already be closed


def _clear(s, ns):
    with s.conn.cursor() as cur:
        cur.execute("DELETE FROM store WHERE prefix = %s", (".".join(ns),))


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        yield led


def test_vector_search_is_captured_and_traced(store, ledger):
    s, ns = store
    langgraph.instrument(s, Capture(ledger), store_name=S)
    with agent_task_scope("support", "t1"):
        s.put(ns, "refunds", {"text": "ana says refunds are accepted for 90 days"})
        s.put(ns, "address", {"text": "ana lives in Brooklyn"})
    with agent_task_scope("support", "t2"):
        hits = s.search(ns, query="refund window days", limit=1)
        assert [h.key for h in hits] == ["refunds"]
        s.put(ns, "reply", {"text": "told ana a 45 day old order is refundable"})
    t = trace(ledger, S, rid(ns, "reply"))
    assert [e.parent_record_id for e in t.edges] == [rid(ns, "refunds")]
    assert blast_radius(ledger, S, rid(ns, "address")).descendants == []


def test_quarantine_hides_items_from_get_and_vector_search(store, ledger):
    s, ns = store
    langgraph.instrument(s, Capture(ledger), store_name=S)
    with agent_task_scope("support", "t1"):
        s.put(ns, "refunds", {"text": "refunds 90 days"})
    with agent_task_scope("support", "t2"):
        s.get(ns, "refunds")
        s.put(ns, "reply", {"text": "refund approved"})
    r = dod.excise(ledger, S, rid(ns, "refunds"))
    assert r.targets == {S: [rid(ns, "refunds"), rid(ns, "reply")]}
    assert s.get(ns, "reply") is None
    assert s.search(ns, query="refund", limit=5) == []


def test_delete_and_reconcile_catch_raw_sql(store, ledger):
    s, ns = store
    langgraph.instrument(s, Capture(ledger), store_name=S)
    with agent_task_scope("support", "t1"):
        s.put(ns, "refunds", {"text": "refunds 90 days"})
        s.put(ns, "address", {"text": "ana lives in Brooklyn"})
    r = dod.excise(ledger, S, rid(ns, "refunds"), mode="delete",
                   stores={S: langgraph.DeleteAdapter(s)})
    assert r.done == {S: [rid(ns, "refunds")]}
    with s.conn.cursor() as cur:  # a write behind capture's back
        cur.execute("INSERT INTO store (prefix, key, value) VALUES (%s, %s, %s)",
                    (".".join(ns), "sneaky", '{"text": "x"}'))
    rep = reconcile(ledger, {S: langgraph.ReconcileAdapter(s)}, grace=0.05)
    # Only this test's namespace: the database may hold other tests' rows.
    mine = [(f.kind, f.record_id) for f in rep.findings
            if tuple(json.loads(f.record_id)[:-1]) == ns]
    # The raw insert is found; the captured write still there ("address") isn't.
    assert mine == [("unrecorded", rid(ns, "sneaky"))]


def test_cross_store_with_crewai(store, ledger):
    from test_crewai import StubMemory

    s, ns = store
    cap = Capture(ledger)
    langgraph.instrument(s, cap, store_name=S)
    mem = crewai.instrument(StubMemory(pool=False), cap)
    with agent_task_scope("support", "t1"):
        bad = mem.remember("refunds are 90 days")
    with agent_task_scope("support", "t2"):
        mem.results["q"] = [bad]
        mem.recall("q")
        s.put(ns, "reply", {"text": "refund approved"})
    b = blast_radius(ledger, "crewai", bad.id)
    assert b.store_counts == {S: 1}
