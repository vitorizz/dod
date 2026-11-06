"""Hard rule: the ledger never stores memory content.

Every integration available here writes and reads a canary through the
real capture path, with prompt confirmation and an excision; then every
file the ledger owns (the ledger, its WAL, the lineage index and its WAL)
is scanned for the canary, as text and as UTF-16. Content hashes and IDs
are all that may be there.
"""

import asyncio
import glob
import os

os.environ["MEM0_TELEMETRY"] = "False"  # before anything imports mem0: no calls home

import pytest
from test_crewai import StubMemory

import dod
from dod import prompt
from dod.capture import Capture
from dod.integrations import crewai
from dod.ledger import Ledger
from dod.scope import agent_task_scope

CANARY = "zebra-canary-7f3a the patient's allergy is penicillin"


def scan(directory):
    found = []
    for f in glob.glob(f"{directory}/*"):
        data = open(f, "rb").read()
        for needle in (CANARY.encode(), CANARY.encode("utf-16-le"), b"zebra-canary-7f3a",
                       b"penicillin"):
            if needle in data:
                found.append((f, needle[:20]))
    return found


@pytest.fixture
def cap(tmp_path):
    led = Ledger(tmp_path / "ledger.db", audit=True)
    yield Capture(led), led, tmp_path
    led.close()


def finish(led, tmp_path, store, record_id):
    prompt.confirm(f"Context: {CANARY}")
    dod.excise(led, store, record_id)
    led.close()
    assert scan(tmp_path) == []


def test_crewai(cap):
    capture, led, tmp = cap
    crewai._blocked.clear()
    mem = crewai.instrument(StubMemory(pool=False), capture)
    with agent_task_scope("agent", "t"):
        rec = mem.remember(CANARY)
        mem.results["q"] = [rec]
        mem.recall("q")
        mem.remember(CANARY + " (derived)")
        finish(led, tmp, "crewai", rec.id)
    crewai._blocked.clear()


def test_langgraph_store(cap):
    pytest.importorskip("langgraph.store.memory")
    from langgraph.store.memory import InMemoryStore

    from dod.integrations import langgraph as lg
    capture, led, tmp = cap
    store = lg.instrument(InMemoryStore(), capture)
    with agent_task_scope("agent", "t"):
        store.put(("u",), "fact", {"text": CANARY})
        store.get(("u",), "fact")
        store.search(("u",), query=None)
        store.put(("u",), "derived", {"text": CANARY + " again"})
        finish(led, tmp, "langgraph", lg.record_id(("u",), "fact"))


def test_langchain_vectorstore(cap):
    pytest.importorskip("langchain_core.vectorstores")
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.vectorstores import InMemoryVectorStore

    from dod.integrations import langchain as lc
    capture, led, tmp = cap
    vs = lc.instrument(InMemoryVectorStore(DeterministicFakeEmbedding(size=8)), capture, store="kb")
    with agent_task_scope("agent", "t"):
        vs.add_texts([CANARY], ids=["doc"], metadatas=[{"note": CANARY}])
        vs.similarity_search("allergy", k=1)
        vs.add_texts([CANARY + " derived"])
        finish(led, tmp, "kb", "doc")


def test_memory_tool(cap):
    pytest.importorskip("anthropic.tools.memory")
    from anthropic.tools.memory import BetaLocalFilesystemMemoryTool

    from dod.integrations.anthropic_memory import instrument, record_id
    capture, led, tmp = cap
    root = tmp.parent / "memories-root"
    root.mkdir()
    tool = instrument(BetaLocalFilesystemMemoryTool(base_path=str(root)), capture, hash_files=True)
    with agent_task_scope("agent", "t"):
        tool.call({"command": "create", "path": "/memories/a.md", "file_text": CANARY})
        tool.call({"command": "view", "path": "/memories/a.md"})
        tool.call({"command": "create", "path": "/memories/b.md", "file_text": CANARY + "!"})
        finish(led, tmp, "anthropic_memory", record_id("/memories/a.md"))


def test_mcp_gateway_tools(cap):
    pytest.importorskip("mcp")
    import anyio
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import create_connected_server_and_client_session as connect

    from dod.integrations import mcp as dmcp
    capture, led, tmp = cap
    server = FastMCP("crm")
    notes = {}

    @server.tool()
    def lookup(key: str) -> str:
        return notes.get(key, CANARY)

    @server.tool()
    def save(key: str, text: str) -> str:
        notes[key] = text
        return key

    async def go():
        async with connect(server._mcp_server) as session:
            s = dmcp.instrument(session, capture, "crm", read_only=["lookup"],
                                record_id=lambda tool, args, result: args.get("key"))
            with agent_task_scope("agent", "t"):
                await s.call_tool("lookup", {"key": "x"})
                await s.call_tool("save", {"key": "n1", "text": CANARY})

    anyio.run(go)
    with agent_task_scope("agent", "t"):
        finish(led, tmp, "mcp:crm", "n1")


def test_mem0_async(cap):
    pytest.importorskip("mem0")
    from test_mem0 import HashEmbedder, ScriptedLLM
    from mem0 import AsyncMemory

    from dod.integrations import mem0 as m0
    capture, led, tmp = cap
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")
    m = AsyncMemory.from_config({
        "vector_store": {"provider": "qdrant", "config": {
            "path": str(tmp.parent / "q"), "embedding_model_dims": 8, "collection_name": "t"}},
        "history_db_path": str(tmp.parent / "h.db")})
    m.llm, m.embedding_model = ScriptedLLM(), HashEmbedder()
    m0.instrument(m, capture)
    m.llm.script = [[CANARY], [CANARY + " derived"]]

    async def go():
        (r,) = (await m.add("x", user_id="u"))["results"]
        await m.add("y", user_id="u")
        return r["id"]

    rid = asyncio.run(go())
    finish(led, tmp, "mem0", rid)


def test_the_scan_finds_content(tmp_path):
    (tmp_path / "leak").write_bytes(b"prefix " + CANARY.encode("utf-16-le"))
    assert scan(tmp_path)


def test_pydantic_ai_tools(cap):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.toolsets import FunctionToolset

    from dod.integrations.pydantic_ai import CaptureToolset, call_record_id
    capture, led, tmp = cap
    ts = FunctionToolset()

    @ts.tool_plain
    def lookup(name: str) -> str:
        return CANARY

    @ts.tool_plain
    def save(text: str) -> dict:
        return {"id": "T1"}

    calls = [("lookup", {"name": "ana"}), ("save", {"text": CANARY})]

    def fn(messages, info):
        if calls:
            name, args = calls.pop(0)
            return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f"c{len(calls)}")])
        return ModelResponse(parts=[TextPart(CANARY)])

    agent = Agent(FunctionModel(fn), name="support", toolsets=[CaptureToolset(
        ts, capture, read_only=["lookup"], record_id=lambda t, a, r: r["id"] if t == "save" else None)])
    agent.run_sync(CANARY)
    rid = led._conn.execute("SELECT record_id FROM bindings ORDER BY rowid LIMIT 1").fetchone()[0]
    assert rid.startswith("call:") and call_record_id
    finish(led, tmp, "pydantic_ai", rid)


def test_openai_agents_tools(cap):
    pytest.importorskip("agents")
    from test_openai_agents import CONFIG, Scripted

    from agents import Agent, Runner, function_tool
    from dod.integrations.openai_agents import instrument_tools
    capture, led, tmp = cap

    @function_tool
    def lookup(topic: str) -> str:
        """Look up."""
        return CANARY

    @function_tool
    def save(note: str) -> str:
        """Save."""
        return "ok"

    tools = instrument_tools([lookup, save], capture, read_only=["lookup"],
                             record_id=lambda n, a, r: "note-1" if n == "save" else None)
    agent = Agent(name="support", model=Scripted([
        ("call", "lookup", {"topic": "x"}, "c1"), ("call", "save", {"note": CANARY}, "c2"),
        ("say", CANARY)]), tools=tools)
    asyncio.run(Runner.run(agent, CANARY, run_config=CONFIG))
    finish(led, tmp, "openai_agents", "call:c1")


def test_langgraph_graph_state(cap):
    pytest.importorskip("langgraph.graph")
    import operator
    from typing import Annotated, TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from dod.integrations.langgraph_graph import instrument_graph
    capture, led, tmp = cap

    class S(TypedDict):
        messages: Annotated[list, operator.add]
        note: str

    def node(state):
        return {"messages": [CANARY + " reply"], "note": CANARY}

    g = StateGraph(S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = instrument_graph(g.compile(checkpointer=InMemorySaver(), name="bot"), capture)
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"messages": [CANARY]}, cfg)
    app.invoke({"messages": [CANARY + " 2"]}, cfg)
    rid = led._conn.execute("SELECT record_id FROM bindings ORDER BY rowid LIMIT 1").fetchone()[0]
    finish(led, tmp, "langgraph_state", rid)
