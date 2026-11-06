import asyncio

import pytest

pytest.importorskip("langchain_core.vectorstores")
pytestmark = pytest.mark.langchain

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.vectorstores import InMemoryVectorStore  # noqa: E402

import dod  # noqa: E402
from dod.capture import Capture  # noqa: E402
from dod.integrations import langchain as lc  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.reconcile import reconcile  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        yield led


@pytest.fixture
def vs(ledger):
    return lc.instrument(InMemoryVectorStore(DeterministicFakeEmbedding(size=16)),
                         Capture(ledger), store="kb")


def derived(ledger, store, rid):
    got = {i for its in blast_radius(ledger, store, rid).bindings.values() for i in its}
    return got - {(store, rid)}


def writes(ledger):
    return ledger._read_conn.execute("SELECT count(*) FROM writes").fetchone()[0]


def test_write_then_search_links_parents(ledger, vs):
    with agent_task_scope("agent", "plant"):
        (bad,) = vs.add_texts(["the refund window is 90 days"], ids=["policy"])
    with agent_task_scope("agent", "t1"):
        hits = vs.similarity_search("refund window", k=1)
        assert [d.id for d in hits] == ["policy"]
        (note,) = vs.add_documents([Document(page_content="told Ben 90 days")])
    assert bad == "policy"
    assert writes(ledger) == 2  # add_texts -> add_documents recorded once
    t = trace(ledger, "kb", note)
    assert len(t.roots) == 1 and t.roots[0].store == "kb" and len(t.ancestors) == 1
    assert derived(ledger, "kb", "policy") == {("kb", note)}


def test_every_read_method_is_tracked(ledger, vs):
    with agent_task_scope("a", "w"):
        vs.add_texts(["alpha fact", "beta fact"], ids=["a", "b"])
    reads = [
        lambda: vs.similarity_search_with_score("alpha", k=1),
        lambda: vs.max_marginal_relevance_search("alpha", k=1, fetch_k=2),
        lambda: vs.search("alpha", "similarity", k=1),
        lambda: vs.get_by_ids(["b"]),
        lambda: vs.as_retriever(search_kwargs={"k": 1}).invoke("alpha"),
    ]
    for i, read in enumerate(reads):
        with agent_task_scope("a", f"r{i}"):
            got = read()
            assert got
            w = vs.add_texts([f"derived {i}"])[0]
        assert trace(ledger, "kb", w).ancestors, i


def test_overwrite_is_a_compaction(ledger, vs):
    with agent_task_scope("a", "1"):
        vs.add_texts(["v1"], ids=["x"])
    with agent_task_scope("a", "2"):
        vs.add_texts(["v2"], ids=["x"])
    parents = ledger._read_conn.execute(
        "SELECT parent_record_id, tag FROM direct_parents").fetchall()
    assert parents == [("x", "in_prompt")]


def test_quarantined_docs_are_dropped(ledger, vs):
    with agent_task_scope("a", "w"):
        vs.add_texts(["bad fact", "good fact"], ids=["bad", "good"])
    ledger.quarantine([("kb", "bad")])
    with agent_task_scope("a", "r"):
        assert {d.id for d in vs.similarity_search("fact", k=2)} == {"good"}
        assert vs.get_by_ids(["bad"]) == []


def test_delete_tombstones_existing_ids_only(ledger, vs):
    with agent_task_scope("a", "w"):
        vs.add_texts(["x"], ids=["x"])
    vs.delete(["x", "never"])
    rows = ledger._read_conn.execute("SELECT record_id, reason FROM tombstones").fetchall()
    assert rows == [("x", "forget")]


def test_excise_delete_and_reconcile(ledger, vs):
    with agent_task_scope("a", "plant"):
        vs.add_texts(["bad"], ids=["bad"])
        vs.add_texts(["fine"], ids=["fine"])
    with agent_task_scope("a", "t"):
        vs.get_by_ids(["bad"])
        (d,) = vs.add_texts(["derived"])
    r = dod.excise(ledger, "kb", "bad", mode="delete", stores={"kb": lc.DeleteAdapter(vs)})
    assert sorted(r.done["kb"]) == sorted(["bad", d])
    assert set(vs.store) == {"fine"}
    reasons = {x for (x,) in ledger._read_conn.execute("SELECT reason FROM tombstones")}
    assert reasons == {"excision"}
    assert reconcile(ledger, {"kb": lc.ReconcileAdapter(vs)}).ok
    # A document added behind the wrapper is found.
    InMemoryVectorStore.add_documents(vs, [Document(page_content="sneaky")], ids=["sneaky"])
    rep = reconcile(ledger, {"kb": lc.ReconcileAdapter(vs)})
    assert [f.record_id for f in rep.findings] == ["sneaky"]


def test_async_methods(ledger, vs):
    async def go():
        with agent_task_scope("a", "w"):
            await vs.aadd_texts(["async fact"], ids=["af"])
        with agent_task_scope("a", "r"):
            hits = await vs.asimilarity_search("async", k=1)
            (w,) = await vs.aadd_documents([Document(page_content="from async")])
        await vs.adelete(["af"])
        return hits, w

    hits, w = asyncio.run(go())
    assert [d.id for d in hits] == ["af"]
    assert derived(ledger, "kb", "af") == {("kb", w)}
    assert writes(ledger) == 2
    assert ledger._read_conn.execute("SELECT count(*) FROM tombstones").fetchone()[0] == 1


def test_prompt_confirmation(ledger, vs):
    with agent_task_scope("a", "w"):
        vs.add_texts(["the refund window is ninety days for all orders"], ids=["p"])
        vs.add_texts(["shipping is free over fifty dollars always"], ids=["s"])
    with agent_task_scope("a", "r"):
        vs.similarity_search("anything", k=2)
        from dod import prompt
        prompt.confirm("Context: the refund window is ninety days for all orders. Answer:")
        (w,) = vs.add_texts(["answer"])
    tags = dict(ledger._read_conn.execute(
        "SELECT m.record_id, m.tag FROM writes w JOIN context_chain c ON c.set_hash ="
        " w.context_set JOIN context_members m ON m.set_hash = c.part"
        " WHERE w.seq = (SELECT max(seq) FROM writes)").fetchall())
    assert tags == {"p": "in_prompt", "s": "retrieved_in_scope"}


def test_instrument_twice_and_type_check(ledger, vs):
    assert lc.instrument(vs, Capture(ledger)) is vs
    with pytest.raises(TypeError):
        lc.instrument(object(), Capture(ledger))
    with pytest.raises(TypeError):
        lc.DeleteAdapter(InMemoryVectorStore(DeterministicFakeEmbedding(size=4)))
