import json
import operator
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph.graph")
pytestmark = pytest.mark.langgraph

import anyio  # noqa: E402

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

import dod  # noqa: E402
from dod.capture import Capture  # noqa: E402
from dod.integrations import langgraph, langgraph_graph  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.scope import agent_task_scope, current_key  # noqa: E402

STATE = "langgraph_state"
NS = ("users", "ana")


class S(TypedDict):
    messages: Annotated[list, operator.add]
    note: str


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        yield led


def build(cap, checkpointer=True, seen=None):
    store = langgraph.instrument(InMemoryStore(), cap)

    def remember(state, *, store):
        if seen is not None:
            seen.append(current_key())
        store.put(NS, "fact", {"text": state["messages"][-1]})
        return {"messages": ["noted"], "note": "saved"}

    g = StateGraph(S)
    g.add_node("remember", remember)
    g.add_edge(START, "remember")
    g.add_edge("remember", END)
    saver = InMemorySaver() if checkpointer else None
    app = g.compile(checkpointer=saver, store=store, name="bot")
    return langgraph_graph.instrument_graph(app, cap), store


def cfg(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def item(channel, thread="t1"):
    return langgraph_graph.record_id(thread, "", channel)


def sources(ledger):
    with ledger._lock:
        return ledger._conn.execute(
            "SELECT b.store, b.record_id, w.source FROM bindings b"
            " JOIN writes w ON w.write_id = b.write_id ORDER BY b.rowid").fetchall()


def test_store_write_in_a_node_traces_to_the_user_input(ledger):
    cap = Capture(ledger)
    seen = []
    app, _ = build(cap, seen=seen)
    out = app.invoke({"messages": ["I'm allergic to peanuts"]}, cfg())
    assert out["messages"] == ["I'm allergic to peanuts", "noted"]
    assert seen == [("bot", "t1")]
    rows = sources(ledger)
    assert (STATE, item("messages"), "user") in rows
    assert (STATE, item("note"), "agent_inference") in rows
    assert not any("__start__" in r or "branch:" in r for _, r, _ in rows)
    fact = langgraph.record_id(NS, "fact")
    t = trace(ledger, "langgraph", fact)
    assert (STATE, item("messages")) in {(e.parent_store, e.parent_record_id) for e in t.edges}
    assert [w.source for w in t.roots] == ["user"]
    assert cap._context == {}  # the run's scope was cleared


def test_later_runs_load_the_thread_state_as_context(ledger):
    cap = Capture(ledger)
    app, _ = build(cap)
    app.invoke({"messages": ["bad fact"]}, cfg())
    first_input = [w for w in sources(ledger) if w[2] == "user"]
    app.invoke({"messages": ["second"]}, cfg())
    # The second input overwrites the channel: a compaction of the first.
    b = blast_radius(ledger, STATE, item("messages"))
    assert b.store_counts.get("langgraph", 0) >= 1
    assert len(first_input) == 1


def test_threads_are_separate(ledger):
    cap = Capture(ledger)
    app, _ = build(cap)
    app.invoke({"messages": ["a"]}, cfg("t1"))
    app.invoke({"messages": ["b"]}, cfg("t2"))
    fact_writes = trace(ledger, "langgraph", langgraph.record_id(NS, "fact"))
    sessions = {w.session_id for w in fact_writes.items}
    assert sessions == {"t1", "t2"}


def test_without_a_checkpointer_runs_are_still_scoped(ledger):
    cap = Capture(ledger)
    seen = []
    app, _ = build(cap, checkpointer=False, seen=seen)
    app.invoke({"messages": ["hi"]})
    assert seen == [("bot", "default")]
    # State is still recorded (the run's input and node updates), keyed by
    # the default thread.
    assert {s for s, _, _ in sources(ledger)} == {"langgraph", STATE}
    assert (STATE, item("messages", "default"), "user") in sources(ledger)


def test_an_existing_scope_is_kept(ledger):
    cap = Capture(ledger)
    seen = []
    app, _ = build(cap, seen=seen)
    with agent_task_scope("planner", "task-9"):
        app.invoke({"messages": ["hi"]}, cfg())
    assert seen == [("planner", "task-9")]
    # State is still recorded, under the caller's scope.
    with ledger._lock:
        rows = ledger._conn.execute(
            "SELECT DISTINCT agent_id, session_id FROM writes WHERE store = ?", (STATE,)
        ).fetchall()
    assert rows == [("planner", "task-9")]
    fact = trace(ledger, "langgraph", langgraph.record_id(NS, "fact"))
    assert [w.source for w in fact.roots] == ["user"]


def test_inside_an_outer_langchain_run(ledger):
    from langchain_core.runnables import RunnableLambda

    cap = Capture(ledger)
    app, _ = build(cap)
    outer = RunnableLambda(lambda x: app.invoke(x, cfg()))
    outer.invoke({"messages": ["nested"]})
    assert (STATE, item("messages"), "user") in sources(ledger)


def test_abandoned_stream_frees_its_context(ledger):
    cap = Capture(ledger)
    app, _ = build(cap)
    gen = app.stream({"messages": ["s"]}, cfg("s2"))
    next(gen)
    gen.close()
    assert cap._context == {}


def test_stream_and_async(ledger):
    cap = Capture(ledger)
    seen = []
    app, _ = build(cap, seen=seen)
    list(app.stream({"messages": ["s"]}, cfg("s1")))

    async def main():
        await app.ainvoke({"messages": ["a"]}, cfg("a1"))
        async for _ in app.astream({"messages": ["b"]}, cfg("b1")):
            pass

    anyio.run(main)
    assert seen == [("bot", "s1"), ("bot", "a1"), ("bot", "b1")]
    assert len([r for r in sources(ledger) if r[2] == "user"]) == 3


def test_excise_from_a_bad_input_reaches_long_term_memory(ledger):
    cap = Capture(ledger)
    app, store = build(cap)
    app.invoke({"messages": ["refunds within 90 days"]}, cfg())
    r = dod.excise(ledger, STATE, item("messages"))
    assert langgraph.record_id(NS, "fact") in r.targets["langgraph"]
    assert store.get(NS, "fact") is None  # quarantined from instrumented reads


def test_instrument_is_idempotent_and_checks_its_argument(ledger):
    cap = Capture(ledger)
    app, _ = build(cap)
    first = app.invoke
    assert langgraph_graph.instrument_graph(app, cap).invoke is first
    with pytest.raises(TypeError, match="compiled graph"):
        langgraph_graph.instrument_graph(object(), cap)


def test_content_is_never_stored(tmp_path):
    with Ledger(tmp_path / "l.db") as ledger:
        app, _ = build(Capture(ledger))
        app.invoke({"messages": ["SECRET-PAYLOAD"]}, cfg())
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.is_file())
    assert b"SECRET-PAYLOAD" not in raw and json.dumps("SECRET").encode() not in raw


# --- prompt confirmation of state channels -------------------------------------------

TA = "Refunds are accepted within 30 days of delivery."
TB = "Ben reported that the refund window is 90 days."


class Q(TypedDict, total=False):
    question: str
    note: str
    answer: str


def confirming(cap, render):
    """A graph whose node confirms ``render(state)`` as its prompt, then
    writes a fact to the store."""
    from dod import prompt

    store = langgraph.instrument(InMemoryStore(), cap)

    def answer(state, *, store):
        prompt.confirm(render(state))
        store.put(NS, "fact", {"text": "answered"})
        return {"answer": "ok"}

    g = StateGraph(Q)
    g.add_node("answer", answer)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    app = g.compile(checkpointer=InMemorySaver(), store=store, name="bot")
    return langgraph_graph.instrument_graph(app, cap)


def last_fact_parent_tags(ledger):
    """Parent tags of the latest write bound to the store's fact."""
    with ledger._lock:
        (w,) = ledger._conn.execute(
            "SELECT write_id FROM bindings WHERE store = 'langgraph' AND record_id = ?"
            " ORDER BY rowid DESC LIMIT 1", (langgraph.record_id(NS, "fact"),)).fetchone()
        return {(s, r): t for s, r, t in ledger._conn.execute(
            "SELECT parent_store, parent_record_id, tag FROM parents WHERE write_id = ?",
            (w,))}


def test_saved_state_channels_are_confirmable(ledger):
    cap = Capture(ledger)
    app = confirming(cap, lambda s: "NOTES: " + s.get("note", ""))
    app.invoke({"question": "first question", "note": TA}, cfg())
    # Second run: the note reaches the node only from the thread's saved state.
    app.invoke({"question": "What is the refund window?"}, cfg())
    tags = last_fact_parent_tags(ledger)
    assert tags[(STATE, item("note"))] == "in_prompt"
    assert tags[(STATE, item("question"))] == "retrieved_in_scope"


def test_recorded_input_channels_are_confirmable(ledger):
    cap = Capture(ledger)
    app = confirming(cap, lambda s: "Q: " + s["question"])
    app.invoke({"question": TB, "note": "an unrelated note here"}, cfg())
    tags = last_fact_parent_tags(ledger)
    assert tags[(STATE, item("question"))] == "in_prompt"
    assert tags[(STATE, item("note"))] == "retrieved_in_scope"
