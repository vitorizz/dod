import json

import pytest

pytest.importorskip("mcp")
pytestmark = pytest.mark.mcp

import anyio  # noqa: E402
from mcp import types  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.shared.exceptions import McpError  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session as connect  # noqa: E402

import dod  # noqa: E402
from dod.capture import Capture  # noqa: E402
from dod.integrations import mcp as dmcp  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.quarantine import blocked_retrievals  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402

RO = types.ToolAnnotations(readOnlyHint=True)
STORE = "mcp:crm"


def crm_server():
    """A toy CRM: read-only lookups, a ticket-creating tool, a resource."""
    server = FastMCP("crm")
    tickets = {}

    @server.tool(annotations=RO)
    def get_customer(name: str) -> str:
        return f"{name}: refund window 90 days"

    @server.tool()
    def create_ticket(text: str) -> dict:
        tid = f"T{len(tickets) + 1}"
        tickets[tid] = text
        return {"id": tid}

    @server.tool()
    def close_ticket(id: str) -> str:
        tickets.pop(id, None)
        return "closed"

    @server.tool()
    def fail() -> str:
        raise ValueError("boom")

    @server.resource("policy://refunds")
    def refunds() -> str:
        return "refunds within 30 days"

    return server


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        yield led


def run(fn):
    return anyio.run(fn)


def ticket_id(tool, arguments, result):
    if tool == "create_ticket":
        return tid(result)
    return None


def tid(result):
    return json.loads(result.content[0].text)["id"]


def bindings(ledger):
    with ledger._lock:
        return ledger._conn.execute(
            "SELECT b.store, b.record_id, w.source FROM bindings b"
            " JOIN writes w ON w.write_id = b.write_id ORDER BY b.rowid"
        ).fetchall()


def writes(ledger):
    with ledger._lock:
        return ledger._conn.execute("SELECT store, source FROM writes ORDER BY seq").fetchall()


def test_read_tool_result_becomes_a_parent_of_the_next_write(ledger):
    cap = Capture(ledger)

    async def main():
        async with connect(crm_server()._mcp_server) as session:
            dmcp.instrument(session, cap, "crm", record_id=ticket_id)
            await session.list_tools()
            with agent_task_scope("support", "t1"):
                await session.call_tool("get_customer", {"name": "ana"})
                r = await session.call_tool("create_ticket", {"text": "approve refund"})
            return r

    r = run(main)
    assert tid(r) == "T1"
    (read, wrote) = bindings(ledger)
    assert read[0] == STORE and read[1].startswith("call:") and read[2] == "tool"
    assert wrote == (STORE, "T1", "agent_inference")
    t = trace(ledger, STORE, "T1")
    assert [(e.parent_record_id, e.tag) for e in t.edges] == [(read[1], "in_prompt")]
    b = blast_radius(ledger, STORE, read[1])
    assert b.store_counts == {STORE: 1}


def test_unknown_tools_are_writes_and_read_only_can_be_named(ledger):
    cap = Capture(ledger)

    async def main():
        async with connect(crm_server()._mcp_server) as session:
            # Never listed, so get_customer's annotation is unknown.
            dmcp.instrument(session, cap, "crm", read_only=["close_ticket"])
            with agent_task_scope("support", "t1"):
                await session.call_tool("get_customer", {"name": "ana"})
                await session.call_tool("close_ticket", {"id": "T9"})

    run(main)
    assert [s for _, s in writes(ledger)] == ["agent_inference", "tool"]


def test_writes_overrides_a_read_only_hint(ledger):
    cap = Capture(ledger)

    async def main():
        async with connect(crm_server()._mcp_server) as session:
            dmcp.instrument(session, cap, "crm", writes=["get_customer"])
            await session.list_tools()
            with agent_task_scope("support", "t1"):
                await session.call_tool("get_customer", {"name": "ana"})

    run(main)
    assert [s for _, s in writes(ledger)] == ["agent_inference"]
    with pytest.raises(ValueError, match="both"):
        dmcp.instrument(object.__new__(type("S", (), {
            "call_tool": lambda *a: None, "read_resource": lambda *a: None,
            "list_tools": lambda *a: None})), cap, "x", read_only=["a"], writes=["a"])


def test_gateway_reads_a_resource_it_has_not_listed(ledger):
    cap = Capture(ledger)

    async def main():
        async with connect(crm_server()._mcp_server) as upstream:
            gw = dmcp.Gateway(cap, {"crm": upstream})
            async with connect(gw.server) as client:
                return (await client.read_resource("policy://refunds")).contents[0].text

    assert run(main) == "refunds within 30 days"


def test_failed_calls_stay_unbound(ledger):
    cap = Capture(ledger)

    async def main():
        async with connect(crm_server()._mcp_server) as session:
            dmcp.instrument(session, cap, "crm")
            with agent_task_scope("support", "t1"):
                r = await session.call_tool("fail", {})
            return r

    assert run(main).isError
    assert len(writes(ledger)) == 1 and bindings(ledger) == []


def test_resource_reads_are_context_and_quarantine_blocks_them(ledger):
    cap = Capture(ledger)

    async def main():
        async with connect(crm_server()._mcp_server) as session:
            dmcp.instrument(session, cap, "crm", record_id=ticket_id)
            with agent_task_scope("support", "t1"):
                await session.read_resource("policy://refunds")
                await session.call_tool("create_ticket", {"text": "30 days"})
            ledger.quarantine([(STORE, "policy://refunds")])
            with agent_task_scope("support", "t2"):
                with pytest.raises(McpError) as e:
                    await session.read_resource("policy://refunds")
            return e.value

    err = run(main)
    assert err.error.code == dmcp.RESOURCE_NOT_FOUND
    t = trace(ledger, STORE, "T1")
    assert [(e.parent_record_id, e.tag) for e in t.edges] == [
        ("policy://refunds", "retrieved_in_scope")
    ]
    assert blocked_retrievals()[(STORE, "policy://refunds")] >= 1


def test_resource_reads_are_confirmable_from_the_prompt(ledger):
    from dod import prompt

    cap = Capture(ledger)

    async def main():
        async with connect(crm_server()._mcp_server) as session:
            dmcp.instrument(session, cap, "crm", record_id=ticket_id)
            with agent_task_scope("support", "t1"):
                await session.read_resource("policy://refunds")
                assert prompt.confirm("Q: can I return it?") == []
                upgraded = prompt.confirm("POLICY: refunds within 30 days\nQ: can I return it?")
                await session.call_tool("create_ticket", {"text": "30 days"})
            return upgraded

    assert run(main) == [(STORE, "policy://refunds")]
    t = trace(ledger, STORE, "T1")
    assert [(e.parent_record_id, e.tag) for e in t.edges] == [("policy://refunds", "in_prompt")]


def test_deleted_records_are_tombstoned(ledger):
    cap = Capture(ledger)

    def deleted(tool, arguments, result):
        return [arguments["id"]] if tool == "close_ticket" else []

    async def main():
        async with connect(crm_server()._mcp_server) as session:
            dmcp.instrument(session, cap, "crm", record_id=ticket_id, deleted=deleted)
            with agent_task_scope("support", "t1"):
                await session.call_tool("create_ticket", {"text": "x"})
                await session.call_tool("close_ticket", {"id": "T1"})

    run(main)
    t = trace(ledger, STORE, "T1")
    assert [d.reason for d in t.deleted[(STORE, "T1")]] == ["forget"]


def test_excise_quarantines_mcp_items(ledger):
    cap = Capture(ledger)

    async def main():
        async with connect(crm_server()._mcp_server) as session:
            dmcp.instrument(session, cap, "crm", record_id=ticket_id)
            await session.list_tools()
            with agent_task_scope("support", "t1"):
                await session.call_tool("get_customer", {"name": "ana"})
                await session.call_tool("create_ticket", {"text": "90 days ok"})

    run(main)
    (read, _) = bindings(ledger)
    r = dod.excise(ledger, STORE, read[1])
    assert r.targets == {STORE: [read[1], "T1"]}


def test_instrument_is_idempotent_and_checks_the_session(ledger):
    cap = Capture(ledger)
    with pytest.raises(TypeError, match="call_tool"):
        dmcp.instrument(object(), cap, "crm")

    class Fake:
        async def call_tool(self, name, arguments=None):
            return types.CallToolResult(content=[])

        async def read_resource(self, uri):
            return None

        async def list_tools(self):
            return types.ListToolsResult(tools=[])

    s = Fake()
    first = dmcp.instrument(s, cap, "crm").call_tool
    assert dmcp.instrument(s, cap, "crm").call_tool is first


# --- Gateway --------------------------------------------------------------------


def test_gateway_proxies_and_captures_per_connection(ledger):
    cap = Capture(ledger)

    async def main():
        async with connect(crm_server()._mcp_server) as upstream:
            gw = dmcp.Gateway(cap, {"crm": upstream},
                              options={"crm": {"record_id": ticket_id}})
            async with connect(gw.server) as client:
                names = {t.name for t in (await client.list_tools()).tools}
                uris = {str(r.uri) for r in (await client.list_resources()).resources}
                await client.call_tool("crm__get_customer", {"name": "ana"})
                text = (await client.read_resource("policy://refunds")).contents[0].text
                r = await client.call_tool("crm__create_ticket", {"text": "refund"})
                bad = await client.call_tool("nope__x", {})
            # A second connection is a separate unit of work.
            async with connect(gw.server) as other:
                await other.call_tool("crm__create_ticket", {"text": "fresh"})
            return names, uris, text, r, bad

    names, uris, text, r, bad = run(main)
    assert "crm__get_customer" in names and "crm__create_ticket" in names
    assert uris == {"policy://refunds"} and text == "refunds within 30 days"
    assert tid(r) == "T1" and bad.isError
    parents = {e.parent_record_id for e in trace(ledger, STORE, "T1").edges}
    assert "policy://refunds" in parents and any(p.startswith("call:") for p in parents)
    assert trace(ledger, STORE, "T2").edges == []
    (agent, session), = {tuple(r) for r in ledger._conn.execute(
        "SELECT agent_id, session_id FROM writes WHERE source = 'agent_inference'"
        " ORDER BY seq LIMIT 1").fetchall()}
    assert agent == "mcp-client" and len(session) == 32


def test_gateway_meta_names_the_unit_of_work(ledger):
    cap = Capture(ledger)

    async def main():
        async with connect(crm_server()._mcp_server) as upstream:
            gw = dmcp.Gateway(cap, {"crm": upstream})
            async with connect(gw.server) as client:
                await client.list_tools()
                meta = {dmcp.META_AGENT: "planner", dmcp.META_TASK: "task-7"}
                await client.call_tool("crm__create_ticket", {"text": "x"}, meta=meta)

    run(main)
    rows = ledger._conn.execute("SELECT agent_id, session_id FROM writes").fetchall()
    assert rows == [("planner", "task-7")]


def test_gateway_rejects_ambiguous_server_names(ledger):
    with pytest.raises(ValueError, match="can't contain"):
        dmcp.Gateway(Capture(ledger), {"a__b": object()})


def test_content_is_never_stored(tmp_path):
    path = tmp_path / "l.db"
    with Ledger(path) as ledger:
        cap = Capture(ledger)

        async def main():
            async with connect(crm_server()._mcp_server) as session:
                dmcp.instrument(session, cap, "crm")
                with agent_task_scope("support", "t1"):
                    await session.call_tool("create_ticket", {"text": "SECRET-PAYLOAD"})

        run(main)
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.is_file())
    assert b"SECRET-PAYLOAD" not in raw and json.dumps("SECRET").encode() not in raw
