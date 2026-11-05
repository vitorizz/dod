import asyncio
import json
from types import SimpleNamespace

import pytest

import dod
from dod import prompt
from dod.capture import Capture, probes
from dod.ledger import Ledger
from dod.query import blast_radius
from dod.scope import agent_task_scope, register

A = ("mem", "a")
B = ("mem", "b")
TA = "Refunds are accepted within 30 days of delivery."
TB = "Ben reported that the refund window is 90 days."


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        yield led


def tags(ledger, write_id):
    with ledger._lock:
        return dict(((s, r), t) for s, r, t in ledger._conn.execute(
            "SELECT parent_store, parent_record_id, tag FROM parents WHERE write_id = ?",
            (write_id,)))


def bound(cap, key, content, rid):
    w = cap.on_write(key, content, "mem", "agent_inference", *key)
    cap.bind(w, rid)
    return w


# --- probes and confirm ---------------------------------------------------------


def test_probes_take_the_start_of_each_long_enough_string():
    assert probes("  Refunds   are\naccepted  ") == ("Refunds are accepted",)
    assert probes("ok") == ()
    assert probes({"text": TA, "n": 3, "tags": ["short", TB]}) == (TA, TB)
    assert probes("x" * 500) == ("x" * 64,)
    assert probes(None) == () and probes(b"Refunds are accepted within") != ()


def test_confirm_upgrades_only_items_in_the_prompt(ledger):
    cap = Capture(ledger)
    key = ("agent", "t1")
    cap.on_retrieve(key, [A, B], texts=[TA, TB])
    before = bound(cap, key, "first", "w1")
    assert cap.confirm(key, f"POLICIES:\n- {TA}\nQUESTION: can I return it?") == [A]
    after = bound(cap, key, "second", "w2")
    assert tags(ledger, before) == {A: "retrieved_in_scope", B: "retrieved_in_scope"}
    assert tags(ledger, after) == {A: "in_prompt", B: "retrieved_in_scope"}
    # Already confirmed: nothing left to upgrade.
    assert cap.confirm(key, TA) == []


def test_confirm_matches_through_whitespace_and_a_truncated_tail(ledger):
    cap = Capture(ledger)
    key = ("agent", "t1")
    long = TA + " " + "Some much longer trailing text. " * 20
    cap.on_retrieve(key, [A], texts=[long])
    rendered = "context: " + TA.replace(" ", "\n  ") + " Some much longer trailing… (truncated)"
    assert cap.confirm(key, rendered) == [A]


def test_items_without_text_are_never_confirmed_and_stay_parents(ledger):
    cap = Capture(ledger)
    key = ("agent", "t1")
    cap.on_retrieve(key, [A])  # no texts
    assert cap.confirm(key, TA) == []
    w = bound(cap, key, "x", "w")
    assert tags(ledger, w) == {A: "retrieved_in_scope"}


def test_reset_drops_the_held_texts(ledger):
    cap = Capture(ledger)
    key = ("agent", "t1")
    cap.on_retrieve(key, [A], texts=[TA])
    cap.reset(key)
    assert cap._probes == {} and cap.confirm(key, TA) == []


def test_prompt_confirm_uses_the_current_scope_and_registered_captures(ledger):
    cap = Capture(ledger)
    register(cap)
    assert prompt.confirm(TA) == []  # outside any scope
    with agent_task_scope("agent", "t1") as key:
        cap.on_retrieve(key, [A, B], texts=[TA, TB])
        assert prompt.confirm(f"{TB}") == [B]
        assert cap._context[key] == {A: "retrieved_in_scope", B: "in_prompt"}


def test_texts_are_never_written_to_the_ledger(tmp_path):
    with Ledger(tmp_path / "l.db") as ledger:
        cap = Capture(ledger)
        key = ("agent", "t1")
        cap.on_retrieve(key, [A], texts=["SECRET-PAYLOAD that must stay out"])
        cap.confirm(key, "SECRET-PAYLOAD that must stay out")
        bound(cap, key, "x", "w")
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.is_file())
    assert b"SECRET-PAYLOAD" not in raw


# --- in_prompt_only walks ---------------------------------------------------------


def chain(ledger):
    """a and b retrieved; w1 written; a confirmed; w2 written; then d is
    written from a direct in_prompt parent w2."""
    cap = Capture(ledger)
    key = ("agent", "t1")
    bound(cap, key, "a", "a")
    bound(cap, key, "b", "b")
    cap.on_retrieve(key, [A, B], texts=[TA, TB])
    w1 = bound(cap, key, "w1", "w1")
    cap.confirm(key, TA)
    w2 = bound(cap, key, "w2", "w2")
    k2 = ("agent", "t2")
    d = cap.on_compaction(k2, [("mem", "w2")], "d", "mem", "agent_inference")
    cap.bind(d, "d")
    return cap


def reached(b):
    return sorted({r for its in b.bindings.values() for _, r in its})


@pytest.mark.parametrize("indexed", [False, True])
def test_blast_radius_in_prompt_only_follows_confirmed_edges(ledger, indexed):
    chain(ledger)
    if indexed:
        while ledger._fill_index():
            pass
    assert reached(blast_radius(ledger, *A)) == ["a", "d", "w1", "w2"]
    assert reached(blast_radius(ledger, *A, in_prompt_only=True)) == ["a", "d", "w2"]
    assert reached(blast_radius(ledger, *B)) == ["b", "d", "w1", "w2"]
    assert reached(blast_radius(ledger, *B, in_prompt_only=True)) == ["b"]
    edges = blast_radius(ledger, *A, in_prompt_only=True).edges
    assert edges and all(e.tag == "in_prompt" for e in edges)


def test_excise_in_prompt_only(ledger):
    chain(ledger)
    assert dod.excise(ledger, *A, dry_run=True).targets == {"mem": ["a", "w1", "w2", "d"]}
    assert dod.excise(ledger, *A, dry_run=True, in_prompt_only=True).targets == {
        "mem": ["a", "w2", "d"]}


# --- model clients -------------------------------------------------------------------


def anthropic_client(sent):
    anthropic = pytest.importorskip("anthropic")
    try:  # the SDK's HTTP library: httpx2 in recent builds, else httpx
        import httpx2 as httpx
    except ImportError:
        import httpx

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})

    return anthropic.Anthropic(api_key="test", http_client=httpx.Client(
        transport=httpx.MockTransport(handler)))


def test_anthropic_client_calls_confirm_their_prompt(ledger):
    from dod.integrations.llm import instrument_anthropic

    cap = Capture(ledger)
    register(cap)
    sent = []
    client = instrument_anthropic(anthropic_client(sent))
    assert instrument_anthropic(client) is client
    with agent_task_scope("agent", "t1") as key:
        cap.on_retrieve(key, [A, B], texts=[TA, TB])
        r = client.messages.create(
            model="m", max_tokens=10, system="You are support.",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Policy: {TA}"}, {"type": "text", "text": "Q?"}]}])
        assert r.content[0].text == "ok" and sent  # the request still went out unchanged
        assert sent[0]["messages"][0]["content"][0]["text"] == f"Policy: {TA}"
        assert cap._context[key] == {A: "in_prompt", B: "retrieved_in_scope"}


def test_tool_results_in_the_prompt_confirm_too(ledger):
    from dod.integrations.llm import instrument_anthropic

    cap = Capture(ledger)
    register(cap)
    client = instrument_anthropic(anthropic_client([]))
    with agent_task_scope("agent", "t1") as key:
        cap.on_retrieve(key, [B], texts=[{"text": TB}])
        client.messages.create(model="m", max_tokens=10, messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t", "name": "recall", "input": {"q": "refund"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": f"Found: {TB}"}]}])
        assert cap._context[key][B] == "in_prompt"


def test_openai_style_clients(ledger):
    from dod.integrations.llm import instrument_openai

    cap = Capture(ledger)
    register(cap)
    calls = []

    async def acreate(**kw):
        calls.append(kw)
        return "done"

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: calls.append(kw))),
        responses=SimpleNamespace(create=acreate))
    instrument_openai(client)
    with agent_task_scope("agent", "t1") as key:
        cap.on_retrieve(key, [A, B], texts=[TA, TB])
        client.chat.completions.create(model="m", messages=[
            {"role": "system", "content": "support"}, {"role": "user", "content": TA}])
        assert cap._context[key][A] == "in_prompt" and cap._context[key][B] != "in_prompt"
        assert asyncio.run(client.responses.create(input=[{"role": "user", "content": TB}],
                                                   instructions="x")) == "done"
        assert cap._context[key][B] == "in_prompt"
    assert len(calls) == 2
    with pytest.raises(TypeError):
        instrument_openai(object())


# --- integrations pass their texts -------------------------------------------------------


def test_crewai_and_langgraph_store_reads_are_confirmable(ledger):
    # MCP resource reads and graph state: test_mcp.py, test_langgraph_graph.py.
    from test_crewai import StubMemory

    from dod.integrations import crewai

    cap = Capture(ledger)
    mem = crewai.instrument(StubMemory(pool=False), cap)
    with agent_task_scope("agent", "seed"):
        rec = mem.remember(TA)
    with agent_task_scope("agent", "t1") as key:
        mem.results["q"] = [rec]
        mem.recall("q")
        assert prompt.confirm(f"you know: {TA}") == [("crewai", rec.id)]

    lg = pytest.importorskip("langgraph.store.memory")
    from dod.integrations import langgraph

    store = langgraph.instrument(lg.InMemoryStore(), cap)
    with agent_task_scope("agent", "seed2"):
        store.put(("kb",), "p", {"text": TB})
    with agent_task_scope("agent", "t2"):
        store.search(("kb",))
        assert prompt.confirm(TB) == [("langgraph", langgraph.record_id(("kb",), "p"))]
