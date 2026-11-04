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


def writes_for(ledger, store, record_id, source=None):
    """Write IDs bound to an item, in order (optionally of one source)."""
    sql = ("SELECT w.write_id FROM bindings b JOIN writes w ON w.write_id = b.write_id"
           " WHERE b.store = ? AND b.record_id = ?")
    args = [store, record_id]
    if source is not None:
        sql += " AND w.source = ?"
        args.append(source)
    with ledger._lock:
        return [r[0] for r in ledger._conn.execute(sql + " ORDER BY b.rowid", args)]


def parents_of(ledger, store, record_id, write_id):
    """``{(parent store, parent record, tag)}`` of one write of an item."""
    t = trace(ledger, store, record_id)
    return {(e.parent_store, e.parent_record_id, e.tag)
            for e in t.edges if e.child_write_id == write_id}


def test_later_runs_load_the_thread_state_as_context(ledger):
    cap = Capture(ledger)
    app, _ = build(cap)
    app.invoke({"messages": ["bad fact"]}, cfg())
    app.invoke({"messages": ["second"]}, cfg())
    first, second = writes_for(ledger, STATE, item("messages"), "user")
    # The second input overwrites the channel: a compaction of its last value.
    p = parents_of(ledger, STATE, item("messages"), second)
    assert (STATE, item("messages"), "in_prompt") in p
    # The thread's saved channels, loaded by get_state, are its context.
    assert (STATE, item("note"), "retrieved_in_scope") in p
    # Run 2's store write takes the loaded state as parents: `note` was only
    # written by run 1's node, so it reaches run 2 through get_state alone.
    fact = langgraph.record_id(NS, "fact")
    _, fact2 = writes_for(ledger, "langgraph", fact)
    assert (STATE, item("note"), "retrieved_in_scope") in parents_of(
        ledger, "langgraph", fact, fact2)
    # And the bad first input reaches run 2's write.
    assert fact2 in {w.write_id for w in blast_radius(ledger, STATE, item("messages")).descendants}


def test_a_failed_state_read_is_logged_and_the_run_continues(ledger, caplog):
    cap = Capture(ledger)
    app, _ = build(cap)
    app.invoke({"messages": ["first"]}, cfg())

    def broken(config, *a, **k):
        raise RuntimeError("checkpointer down")

    object.__setattr__(app, "get_state", broken)
    out = app.invoke({"messages": ["second"]}, cfg())
    assert out["messages"][-1] == "noted"
    assert "couldn't read the thread's state" in caplog.text
    _, second = writes_for(ledger, STATE, item("messages"), "user")
    p = parents_of(ledger, STATE, item("messages"), second)
    assert (STATE, item("note"), "retrieved_in_scope") not in p
    assert cap._context == {}


def test_interleaved_streams_do_not_share_a_scope(ledger):
    cap = Capture(ledger)
    seen = []
    app, _ = build(cap, seen=seen)
    a = app.stream({"messages": ["a"]}, cfg("t1"))
    next(a)  # t1 is suspended mid-run, inside its scope
    assert current_key() is None  # ... which doesn't leak to the caller
    list(app.stream({"messages": ["b"]}, cfg("t2")))
    list(a)
    assert sorted(seen) == [("bot", "t1"), ("bot", "t2")]
    # t2 recorded its own state, and its store write has only t2 parents.
    assert (STATE, item("messages", "t2"), "user") in sources(ledger)
    fact = langgraph.record_id(NS, "fact")
    for w in writes_for(ledger, "langgraph", fact):
        threads = {json.loads(r)[0] for s, r, _ in parents_of(ledger, "langgraph", fact, w)
                   if s == STATE}
        assert len(threads) == 1
    assert cap._context == {}


@pytest.mark.parametrize("how", ["keyword", "list", "manager"])
def test_config_forms_and_user_callbacks(ledger, how):
    from langchain_core.callbacks import BaseCallbackHandler, CallbackManager

    class Count(BaseCallbackHandler):
        def __init__(self):
            self.starts = 0

        def on_chain_start(self, *a, **k):
            self.starts += 1

    cap = Capture(ledger)
    app, _ = build(cap)
    mine = Count()
    config = cfg()
    if how == "list":
        config["callbacks"] = [mine]
    elif how == "manager":
        config["callbacks"] = CallbackManager([mine])
    if how == "keyword":
        app.invoke({"messages": ["hi"]}, config=config)
    else:
        app.invoke({"messages": ["hi"]}, config)
        assert mine.starts > 0  # the caller's handler still runs
        assert config["callbacks"] == [mine] or how == "manager"  # not mutated
    rows = sources(ledger)
    assert (STATE, item("messages"), "user") in rows
    assert (STATE, item("note"), "agent_inference") in rows


def test_non_dict_updates_are_not_recorded(ledger):
    from langgraph.types import Command

    cap = Capture(ledger)

    def cmd(state):
        return Command(update={"note": "via command"})

    g = StateGraph(S)
    g.add_node("cmd", cmd)
    g.add_edge(START, "cmd")
    g.add_edge("cmd", END)
    app = langgraph_graph.instrument_graph(
        g.compile(checkpointer=InMemorySaver(), name="bot"), cap)
    out = app.invoke({"messages": ["hi"]}, cfg())
    assert out["note"] == "via command"
    # Documented gap: the input is recorded, the Command's update is not.
    assert sources(ledger) == [(STATE, item("messages"), "user")]


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
