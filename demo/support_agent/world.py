"""The demo world: a support agent whose memory spans four real stores.

- ``memory``: long-term memory in Postgres + pgvector (LangGraph
  ``PostgresStore``), or an in-memory store when no Postgres URI is given.
  Company policies and per-customer notes live here.
- ``notes``: policy notes as files, one per note, through Anthropic's
  memory tool (``/memories/policies/<topic>/<n>.md``).
- ``mcp:crm``: the CRM, a separate MCP server process over SQLite: orders,
  tickets and replies sent to customers.
- ``langgraph_state``: the agent's conversation state, per customer thread.
- ``console``: corrections a support lead makes (``correction:<n>``); see
  ``correction.py``.

Every store is instrumented with dod against one ledger; the graph run is the
unit of work, so one customer message links everything it read to
everything it wrote, across stores.
"""

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from anthropic.lib.tools import ToolError
from anthropic.tools.memory import BetaLocalFilesystemMemoryTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import GetOp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing_extensions import Annotated, TypedDict
import operator

import dod
from dod import prompt as dod_prompt
from dod.capture import Capture
from dod.integrations import anthropic_memory, langgraph, langgraph_graph
from dod.integrations import mcp as dmcp
from dod.ledger import Ledger
from dod.query import blast_radius, trace
from dod.scope import agent_task_scope, current_key, use_key

from .llm import Context, from_spec

REPO = Path(__file__).resolve().parents[2]

POLICIES = {
    "refunds": "Refunds are accepted within 30 days of delivery.",
    "shipping": "Standard shipping takes 5 business days; express takes 2.",
    "warranty": "Electronics carry a 12 month warranty against defects.",
    "exchanges": "Exchanges are allowed within 60 days of delivery.",
    "support": "Support replies within 24 hours, seven days a week.",
    "loyalty": "Members earn 2 points per dollar spent.",
}
TOPICS = {
    "refunds": ("refund", "return", "money back"),
    "shipping": ("ship", "deliver", "arrive"),
    "warranty": ("warranty", "broken", "defect"),
}

# The story: a mistaken update from ops (a source the agent should trust),
# an unrelated question, then a customer it reaches twice (the second
# answer builds on the first).
STORY: List[Tuple[str, str]] = [
    ("ops", "Policy update: the refund window has been extended to 90 days, effective today."
            " Please apply it to refund requests."),
    ("dev", "How long does standard shipping take?"),
    ("ana", "Order 2001 was delivered 45 days ago. Can I still return it for a refund?"),
    ("ana", "When will the refund for order 2001 arrive?"),
]
STAFF = {"ops"}  # senders on the internal channel; everyone else is a customer
CUSTOMERS = ["ana", "dev", "ops"]
# What the support lead reports once a wrong answer surfaces.
CORRECTION = "The refund window is 30 days, not 90."

STORE_LABELS = {
    "langgraph_state": "Conversation state",
    "memory": "Long-term memory",
    "notes": "Policy notes (files)",
    "mcp:crm": "CRM (MCP)",
    "console": "Corrections",
}


def embed(texts):
    """A stand-in embedder (hashed bag of words) so the demo runs offline.
    Swap in a real embedding model for production."""
    out = []
    for text in texts:
        v = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[hashlib.sha256(word.encode()).digest()[0] % 64] += 1.0
        out.append(v)
    return out


class State(TypedDict):
    messages: Annotated[list, operator.add]


class World:
    def __init__(self, workdir: str, model: str = "scripted",
                 pg_uri: Optional[str] = None) -> None:
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.llm = from_spec(model)
        self.pg_uri = pg_uri
        self.memory_store = "memory"
        self.log: List[dict] = []  # chat transcript for the UI
        # One entry per unit of work, in order: the handbook seed, then each
        # customer message. Writes with seq in (start, end] belong to it.
        self.turns: List[dict] = []
        self._notes = 0
        self.corrections: List[dict] = []

    # --- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        self.ledger = Ledger(self.dir / "ledger.db", audit=True)
        self.cap = Capture(self.ledger)
        if self.pg_uri:
            from langgraph.store.postgres import PostgresStore

            self._pg_cm = PostgresStore.from_conn_string(
                self.pg_uri, index={"dims": 64, "embed": embed, "fields": ["text"]})
            raw = self._pg_cm.__enter__()
            raw.setup()
            with raw.conn.cursor() as cur:  # a fresh demo: clear what earlier runs left
                cur.execute("DELETE FROM store WHERE prefix = 'policies'"
                            " OR prefix LIKE 'customers.%'")
            self.backend = "Postgres + pgvector"
        else:
            from langgraph.store.memory import InMemoryStore

            raw = InMemoryStore(index={"dims": 64, "embed": embed, "fields": ["text"]})
            self.backend = "in-memory (no Postgres URI given)"
        self.memory = langgraph.instrument(raw, self.cap, store_name=self.memory_store)
        self.notes = anthropic_memory.instrument(
            BetaLocalFilesystemMemoryTool(base_path=str(self.dir / "notes")), self.cap,
            store="notes", hash_files=True)

        self._crm_db = str(self.dir / "crm.db")
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "demo.support_agent.crm_server", self._crm_db],
            cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)})
        # The MCP client's cancel scopes must be entered and exited in one
        # task, so the connection lives in a task of its own until stop().
        ready: "asyncio.Future" = asyncio.get_running_loop().create_future()
        self._closing = asyncio.Event()

        async def connection():
            try:
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        ready.set_result(session)
                        await self._closing.wait()
            except BaseException as exc:
                if not ready.done():
                    ready.set_exception(exc)
                raise

        self._crm_task = asyncio.create_task(connection())
        session = await ready
        self.crm = dmcp.instrument(session, self.cap, "crm", record_id=_result_id)
        await self.crm.list_tools()  # learn which tools are read-only

        g = StateGraph(State)
        g.add_node("agent", self._agent)
        g.add_edge(START, "agent")
        g.add_edge("agent", END)
        self.graph = langgraph_graph.instrument_graph(
            g.compile(checkpointer=InMemorySaver(), store=self.memory, name="support"),
            self.cap)
        self._seed()

    async def stop(self) -> None:
        self._closing.set()
        try:
            await self._crm_task
        except Exception:
            pass
        self.ledger.close()
        if self.pg_uri:
            self._pg_cm.__exit__(None, None, None)

    def _seq(self) -> int:
        with self.ledger._lock:
            return self.ledger._conn.execute(
                "SELECT coalesce(max(seq), 0) FROM writes").fetchone()[0]

    def _seed(self) -> None:
        start = self._seq()
        with agent_task_scope("onboarding", "policy-handbook"):
            for topic, text in POLICIES.items():
                self.memory.put(("policies",), topic, {"text": text})
        self.turns.append({"id": 0, "kind": "seed", "customer": None,
                           "message": "Company policy handbook loaded",
                           "start": start, "end": self._seq()})

    # --- the agent -----------------------------------------------------------

    async def chat(self, customer: str, message: str, thread: Optional[str] = None,
                   replaces: Optional[str] = None) -> dict:
        """One customer message through the agent; returns the reply.
        ``replaces`` is the reply ID a corrected answer stands in for: the
        reply is then sent as a correction of it."""
        start = self._seq()
        out = await self.graph.ainvoke(
            {"messages": [{"role": "customer", "customer": customer, "text": message,
                           "correction": bool(replaces)}]},
            {"configurable": {"thread_id": thread or customer}},
        )
        reply = out["messages"][-1]
        entry = {"customer": customer, "message": message, "reply": reply["text"],
                 "reply_id": reply["reply_id"], "thread": thread or customer,
                 "turn": len(self.turns), "replaces": replaces}
        self.log.append(entry)
        self.turns.append({"id": len(self.turns), "kind": "regenerated" if replaces else "chat",
                           "customer": customer, "message": message,
                           "reply_id": reply["reply_id"], "replaces": replaces,
                           "start": start, "end": self._seq()})
        return entry

    async def run_story(self) -> List[dict]:
        return [await self.chat(c, m) for c, m in STORY]

    async def _agent(self, state: State, *, store) -> dict:
        msg = state["messages"][-1]
        customer, text = msg["customer"], msg["text"]
        order = None  # looked up first: it depends only on the message
        m = re.search(r"\b(2\d{3})\b", text)
        if m:
            r = await self.crm.call_tool("lookup_order", {"order_id": m.group(1)})
            order = json.loads(r.content[0].text)
            if "error" in order:
                order = None
        # Semantic search, ranked by relevance. The handbook is small, so
        # every policy fits in the prompt; a real one would keep the top k.
        policies = [i.value["text"] for i in self.memory.search(("policies",), query=text,
                                                                 limit=len(POLICIES))]
        notes_about = [i.value["text"] for i in self.memory.search(("customers", customer),
                                                                    limit=10)]
        topics = [t for t, words in TOPICS.items() if any(w in text.lower() for w in words)]
        policy_notes: List[str] = []
        for topic in topics:
            policy_notes += self._read_note(topic)
        ctx = Context(customer, text, policies, policy_notes, notes_about, order,
                      staff=customer in STAFF)
        # What actually goes to the model: everything in it becomes in_prompt.
        dod_prompt.confirm(ctx.prompt())
        decision = await asyncio.to_thread(self.llm.decide, ctx)

        for note in decision.remember:
            self.memory.put(("customers", customer), uuid.uuid4().hex[:8], {"text": note})
        if decision.policy_update:
            topic = re.sub(r"[^a-z]", "", str(decision.policy_update.get("topic", "general"))
                           .lower()) or "general"
            self._append_note(topic, str(decision.policy_update.get("text", "")))
        if decision.ticket:
            t = decision.ticket
            await self.crm.call_tool("create_ticket", {
                "customer": customer, "order_id": str(t.get("order_id", "")),
                "action": str(t.get("action", "other")), "summary": str(t.get("summary", ""))})
        reply = decision.reply
        if msg.get("correction"):
            reply = "Correction to my earlier reply: " + reply
        r = await self.crm.call_tool("send_reply", {
            "customer": customer, "question": text, "text": reply})
        reply_id = _result_id("send_reply", {}, r)
        return {"messages": [{"role": "agent", "customer": customer, "text": reply,
                              "reply_id": reply_id}]}

    def _read_note(self, topic: str) -> List[str]:
        """The topic's policy notes, oldest first (one file per note; files
        dod has quarantined aren't listed)."""
        try:
            listing = self.notes.call({"command": "view", "path": f"/memories/policies/{topic}"})
        except ToolError:
            return []
        paths = sorted(p for p in (line.partition("\t")[2] for line in str(listing).splitlines()[1:])
                       if p.endswith(".md"))
        lines = []
        for path in paths:
            try:
                out = self.notes.call({"command": "view", "path": path})
            except ToolError:
                continue
            for line in str(out).splitlines()[1:]:
                body = line.partition("\t")[2].strip()
                if body:
                    lines.append(body)
        return lines

    def _append_note(self, topic: str, text: str) -> str:
        self._notes += 1
        path = f"/memories/policies/{topic}/{self._notes:03d}.md"
        self.notes.call({"command": "create", "path": path, "file_text": text + "\n"})
        return path

    async def regenerate(self, store: str, rid: str, correction: Tuple[str, str]) -> dict:
        """``dod.regenerate`` from a bad item, for a correction
        (``(record ID, text)``): what it reached is quarantined (kept on
        record, hidden from the agent), and every reply it reached is
        answered again as a correction.

        A reply built on the bad item gets the question re-run through the
        agent on a fresh thread (the old conversation is contaminated too).
        A reply to the bad message itself gets the correction's text instead:
        re-running the message would repeat the mistake. Notes and state
        aren't rewritten; a re-run writes again what it still needs.
        """
        loop = asyncio.get_running_loop()
        cid, text = correction
        runs = [0]

        def rewrite(task):
            if task.store != "mcp:crm" or not task.record_id.startswith("R"):
                return None
            db = sqlite3.connect(self._crm_db)
            try:
                customer, question = db.execute(
                    "SELECT customer, question FROM replies WHERE id = ?",
                    (task.record_id,)).fetchone()
            finally:
                db.close()
            key = current_key()
            notice = any(p.status == "bad" for p in task.parents)
            runs[0] += 1
            thread = f"{customer}-corrected-{runs[0]}"

            async def again():
                with use_key(*key):
                    if notice:
                        return await self._notice(customer, question, task.record_id, cid, text)
                    return await self.chat(customer, question, thread=thread,
                                           replaces=task.record_id)

            entry = asyncio.run_coroutine_threadsafe(again(), loop).result()
            return ("mcp:crm", entry["reply_id"])

        result = await asyncio.to_thread(dod.regenerate, self.ledger, store, rid, rewrite)
        return {
            "replaced": [[list(a), list(b)] for a, b in result.replaced.items()],
            "dropped": [list(i) for i in result.dropped],
            "failed": [list(f) for f in result.failed],
            "targets": result.targets,
        }

    async def _notice(self, customer: str, question: str, replaces: str, cid: str,
                      text: str) -> dict:
        """Send the correction itself as the reply to a bad message."""
        start = self._seq()
        self.cap.on_retrieve(current_key(), [("console", cid)], "in_prompt")
        reply = f"Correction to my earlier reply: {text}"
        r = await self.crm.call_tool("send_reply", {
            "customer": customer, "question": question, "text": reply})
        entry = {"customer": customer, "message": question, "reply": reply,
                 "reply_id": _result_id("send_reply", {}, r), "thread": customer,
                 "turn": len(self.turns), "replaces": replaces}
        self.log.append(entry)
        self.turns.append({"id": len(self.turns), "kind": "regenerated", "customer": customer,
                           "message": question, "reply_id": entry["reply_id"],
                           "replaces": replaces, "start": start, "end": self._seq()})
        return entry

    def record_correction(self, text: str, topic: str) -> Tuple[str, str]:
        """Start a correction: record what the support lead said (store
        ``console``, source ``user``), then, with it as the only context,
        write it as a policy note. Returns its record ID and the note's path.
        Tickets are cancelled by ``cancel_tickets`` in the same turn."""
        cid = f"correction:{len(self.corrections) + 1}"
        self._correction_start = self._seq()
        with agent_task_scope("support-lead", cid):
            key = current_key()
            wid = self.cap.on_write(key, text, "console", "user")
            self.cap.bind(wid, cid)
            self.cap.on_retrieve(key, [("console", cid)], "in_prompt")
            path = self._append_note(topic, f"Correction by the support lead: {text}")
        self.corrections.append({"id": cid, "text": text, "note": path})
        return cid, path

    async def cancel_tickets(self, cid: str, tickets: List[str]) -> None:
        """Cancel the tickets a correction reached, in the CRM (so fulfilment
        doesn't act on them), with the correction as their context; then
        close the correction's turn."""
        with agent_task_scope("support-lead", cid):
            self.cap.on_retrieve(current_key(), [("console", cid)], "in_prompt")
            for t in tickets:
                await self.crm.call_tool("update_ticket", {"ticket_id": t, "status": "cancelled"})
        text = next(c["text"] for c in self.corrections if c["id"] == cid)
        self.turns.append({"id": len(self.turns), "kind": "correction", "customer": None,
                           "message": text, "correction": cid,
                           "start": self._correction_start, "end": self._seq()})

    def reply_pair(self, old: str, new: str) -> dict:
        """A corrected reply next to the one it replaces."""
        db = sqlite3.connect(self._crm_db)
        try:
            rows = {r: db.execute("SELECT customer, question, text FROM replies WHERE id = ?",
                                  (r,)).fetchone() for r in (old, new)}
        finally:
            db.close()
        customer, question, before = rows[old]
        return {"customer": customer, "question": question, "before": before,
                "after": rows[new][2], "old": old, "new": new}

    def ticket_status(self, tid: str) -> Optional[str]:
        db = sqlite3.connect(self._crm_db)
        try:
            row = db.execute("SELECT status FROM tickets WHERE id = ?", (tid,)).fetchone()
        finally:
            db.close()
        return row[0] if row else None

    def _bound(self, write_id: str) -> List[Tuple[str, str]]:
        with self.ledger._lock:
            return self.ledger._conn.execute(
                "SELECT store, record_id FROM bindings WHERE write_id = ?", (write_id,)
            ).fetchall()


def _result_id(tool: str, arguments: Any, result: Any) -> Optional[str]:
    """The CRM's ID for what a tool created (tickets, replies)."""
    try:
        return json.loads(result.content[0].text).get("id")
    except Exception:
        return None
