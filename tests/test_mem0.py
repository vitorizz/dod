import hashlib
import json
import os

import pytest

os.environ["MEM0_TELEMETRY"] = "False"  # before mem0 is imported: no calls home
pytest.importorskip("mem0")
pytestmark = pytest.mark.mem0

from mem0 import Memory  # noqa: E402

import dod  # noqa: E402
from dod.capture import Capture  # noqa: E402
from dod.integrations import mem0 as m0  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.reconcile import reconcile  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402


class ScriptedLLM:
    """Returns the next scripted list of extracted facts; keeps the prompts."""

    def __init__(self):
        self.script, self.prompts = [], []

    def generate_response(self, messages, response_format=None, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return json.dumps({"memory": [{"text": t} for t in self.script.pop(0)]})


class HashEmbedder:
    def embed(self, text, action=None):
        return [b / 255 for b in hashlib.sha256(str(text).encode()).digest()[:8]]

    def embed_batch(self, texts, action=None):
        return [self.embed(t) for t in texts]


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        yield led


@pytest.fixture
def mem(tmp_path, ledger, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # the default clients are replaced below
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0"))
    m = Memory.from_config({
        "vector_store": {"provider": "qdrant", "config": {
            "path": str(tmp_path / "qdrant"), "embedding_model_dims": 8, "collection_name": "t"}},
        "history_db_path": str(tmp_path / "history.db"),
    })
    m.llm, m.embedding_model = ScriptedLLM(), HashEmbedder()
    yield m0.instrument(m, Capture(ledger))
    m.close()


def ids(out):
    return [r["id"] for r in out["results"]]


def derived(ledger, rid):
    return {i for its in blast_radius(ledger, "mem0", rid).bindings.values() for i in its} \
        - {("mem0", rid)}


def test_internal_retrieval_is_the_parent(ledger, mem):
    mem.llm.script = [["User is allergic to peanuts"], ["Avoid Thai restaurants for the user"]]
    (bad,) = ids(mem.add("I'm allergic to peanuts", user_id="u1"))
    # No explicit search: Mem0 retrieves the existing memory for its prompt.
    (d1,) = ids(mem.add("what should I eat tonight?", user_id="u1"))
    assert "allergic to peanuts" in mem.llm.prompts[1]
    assert derived(ledger, bad) == {("mem0", d1)}
    tags = dict(ledger._read_conn.execute(
        "SELECT m.record_id, m.tag FROM writes w JOIN context_chain c ON c.set_hash ="
        " w.context_set JOIN context_members m ON m.set_hash = c.part"
        " WHERE w.seq = (SELECT max(seq) FROM writes)").fetchall())
    assert tags == {bad: "in_prompt"}
    t = trace(ledger, "mem0", d1)
    assert [r.store for r in t.roots] == ["mem0"]


def test_scope_reads_are_parents_too(ledger, mem):
    mem.llm.script = [["Ben ordered on March 3"], []]
    (fact,) = ids(mem.add("my order was March 3", user_id="u1"))
    with agent_task_scope("agent", "t1"):
        assert ids(mem.search("order", filters={"user_id": "u1"}, threshold=0)) == [fact]
        mem.add("raw agent note", user_id="u2", infer=False)
    (note,) = derived(ledger, fact)
    assert ledger._read_conn.execute(
        "SELECT source FROM writes ORDER BY seq DESC LIMIT 1").fetchone() == ("user",)
    assert note[0] == "mem0"


def test_infer_false_source_follows_role(ledger, mem):
    # Mem0 2.0.5 returns each infer=False message's role; system messages are skipped.
    out = mem.add([{"role": "system", "content": "be terse"},
                   {"role": "user", "content": "I'm vegetarian"},
                   {"role": "assistant", "content": "Noted: no meat"}],
                  user_id="u1", infer=False)
    by_text = {r["memory"]: r["id"] for r in out["results"]}
    assert set(by_text) == {"I'm vegetarian", "Noted: no meat"}
    source = dict(ledger._read_conn.execute(
        "SELECT b.record_id, w.source FROM bindings b JOIN writes w ON w.write_id = b.write_id"
        " WHERE b.store = 'mem0'").fetchall())
    assert source == {by_text["I'm vegetarian"]: "user",
                      by_text["Noted: no meat"]: "agent_inference"}
    # A plain string is a user message.
    (s,) = ids(mem.add("I live in Brooklyn", user_id="u1", infer=False))
    assert ledger._read_conn.execute(
        "SELECT w.source FROM bindings b JOIN writes w ON w.write_id = b.write_id"
        " WHERE b.record_id = ?", (s,)).fetchone() == ("user",)
    # With infer=True, every extracted memory is agent_inference.
    mem.llm.script = [["User is vegetarian"]]
    (x,) = ids(mem.add("I'm vegetarian", user_id="u2"))
    assert ledger._read_conn.execute(
        "SELECT w.source FROM bindings b JOIN writes w ON w.write_id = b.write_id"
        " WHERE b.record_id = ?", (x,)).fetchone() == ("agent_inference",)


def test_get_get_all_and_quarantine(ledger, mem):
    mem.llm.script = [["fact one is here"], ["fact two is here"]]
    (a,) = ids(mem.add("one", user_id="u1", infer=True))
    (b,) = ids(mem.add("two", user_id="u2"))
    ledger.quarantine([("mem0", a)])
    with agent_task_scope("agent", "r"):
        assert mem.get(a) is None and mem.get(b)["id"] == b
        assert ids(mem.get_all(filters={"user_id": "u1"})) == []
        assert ids(mem.search("fact", filters={"user_id": "u2"}, threshold=0)) == [b]


def test_update_is_a_compaction_and_delete_tombstones(ledger, mem):
    mem.llm.script = [["User likes tea"]]
    (a,) = ids(mem.add("tea", user_id="u1"))
    with agent_task_scope("agent", "fix"):
        mem.update(a, "User likes green tea")
    assert ledger._read_conn.execute(
        "SELECT parent_record_id, tag FROM direct_parents").fetchall() == [(a, "in_prompt")]
    mem.delete(a)
    mem.llm.script = [["x fact one"], ["y fact two"]]
    mem.add("x", user_id="u9")
    mem.add("y", user_id="u9")
    mem.delete_all(user_id="u9")
    rows = ledger._read_conn.execute("SELECT reason FROM tombstones").fetchall()
    assert rows == [("forget",)] * 3


def test_excise_delete_and_reconcile(ledger, mem):
    mem.llm.script = [["User is allergic to peanuts"], ["Avoid Thai restaurants"],
                      ["User lives in Brooklyn"]]
    (bad,) = ids(mem.add("peanuts", user_id="u1"))
    (d1,) = ids(mem.add("dinner?", user_id="u1"))
    (good,) = ids(mem.add("I live in Brooklyn", user_id="u2"))
    r = dod.excise(ledger, "mem0", bad, mode="delete", stores={"mem0": m0.DeleteAdapter(mem)})
    assert sorted(r.done["mem0"]) == sorted([bad, d1])
    assert ids(mem.get_all(filters={"user_id": "u1"})) == []
    assert ids(mem.get_all(filters={"user_id": "u2"})) == [good]
    assert reconcile(ledger, {"mem0": m0.ReconcileAdapter(mem)}).ok
    # A memory written behind the wrapper is reported.
    mem._dod_base["add"]("sneaky", user_id="u3", infer=False)
    rep = reconcile(ledger, {"mem0": m0.ReconcileAdapter(mem)})
    assert [f.kind for f in rep.findings] == ["unrecorded"]


def test_no_content_in_ledger(ledger, mem, tmp_path):
    mem.llm.script = [["secret allergy fact"]]
    mem.add("secret", user_id="u1")
    ledger.close()
    assert b"secret allergy" not in (tmp_path / "ledger.db").read_bytes()


def test_instrument_twice_and_type_check(ledger, mem):
    assert m0.instrument(mem, Capture(ledger)) is mem
    with pytest.raises(TypeError):
        m0.instrument(object(), Capture(ledger))


def test_updates_inside_add_are_recorded(ledger, mem):
    # mem0 2.0.5's add() only adds; a version (or custom pipeline) that
    # rewrites inside add goes through _update_memory, which is captured.
    mem.llm.script = [["User likes tea"], ["User likes green tea"]]
    (a,) = ids(mem.add("tea", user_id="u1"))
    original = mem._dod_base["add"]

    def add_that_updates(messages, **kwargs):
        out = original(messages, **kwargs)
        mem._update_memory(a, "User likes green tea", {"User likes green tea": [0.1] * 8})
        return out

    mem._dod_base["add"] = add_that_updates
    with agent_task_scope("agent", "t"):
        mem.add("actually green tea", user_id="u1")
    rows = ledger._read_conn.execute(
        "SELECT w.agent_id, d.parent_record_id, d.tag FROM writes w"
        " JOIN direct_parents d ON d.write_id = w.write_id").fetchall()
    assert rows == [("agent", a, "in_prompt")]
