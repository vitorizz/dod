"""The demo world: a support agent whose memory spans four real stores.

- ``memory``: long-term memory in Postgres + pgvector (LangGraph
  ``PostgresStore``), or an in-memory store when no Postgres URI is given.
  Company policies and per-customer notes live here.
- ``notes``: policy notes as files, one per note, through Anthropic's
  memory tool (``/memories/policies/<topic>/<n>.md``).
- ``mcp:crm``: the CRM, a separate MCP server process over SQLite: orders,
  tickets and replies sent to customers.
- ``langgraph_state``: the agent's conversation state, per customer thread.
- ``admin``: corrections a support lead makes, recorded by ``dod.admin``
  (``correction:<plan id>``); see ``correction.py``.

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
from dod.admin import Actions, Admin, keyword_matcher, llm_matcher
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
    "admin": "Corrections",
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
        self._raw_memory = raw
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
        self.admin = self._admin()
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

    # --- dod.admin: readers, actions and the correction's turn ---------------

    def _admin(self) -> Admin:
        """``dod.admin`` over the four stores. Readers read the stores
        directly (not through capture: reading for a person isn't the agent
        reading); actions do the app's side of a fix."""
        llm_complete = getattr(self.llm, "complete", None)
        matcher = llm_matcher(llm_complete) if llm_complete else keyword_matcher
        return Admin(
            self.ledger,
            readers={"notes": _Rows(self._read_notes), self.memory_store: _Rows(self._read_memory),
                     "langgraph_state": _Rows(self._read_threads), "mcp:crm": _Rows(self._read_crm)},
            capture=self.cap, matcher=matcher, classify=self._classify,
            searchable=["notes", self.memory_store],
            actions=Actions(correct=self._correct, cancel=self._cancel,
                            rerun=self._rerun, notify=self._notify),
            label=self.label)

    def _read_notes(self):
        base = self.dir / "notes"
        for f in sorted(base.rglob("*.md")):
            yield "/" + str(f.relative_to(base)), f.read_text()

    def _read_memory(self):
        for ns in [("policies",)] + list(self._raw_memory.list_namespaces(prefix=("customers",))):
            for item in self._raw_memory.search(ns, limit=1000):
                text = item.value.get("text", "")
                if ns[0] == "customers":
                    text = f"Note on {ns[1]}: {text}"
                yield json.dumps([*ns, item.key]), text

    def _read_threads(self):
        threads: Dict[str, List[str]] = {}
        for e in self.log:
            threads.setdefault(e["thread"], []).extend([e["message"], e["reply"]])
        for t, lines in threads.items():
            yield langgraph_graph.record_id(t, "", "messages"), "\n".join(lines)

    def _read_crm(self):
        db = sqlite3.connect(self._crm_db)
        try:
            for rid, customer, text in db.execute("SELECT id, customer, text FROM replies"):
                yield rid, f"To {customer}: {text}"
            for rid, summary, status in db.execute("SELECT id, summary, status FROM tickets"):
                yield rid, f"{summary} ({status})"
        finally:
            db.close()

    def _classify(self, item) -> str:
        store, rid = item
        if store == "mcp:crm":  # tickets, replies, and read tools' results (history)
            return {"T": "cancel", "R": "notify"}.get(rid[:1], "record")
        return "record" if store == "langgraph_state" else "fix"

    def label(self, item) -> str:
        from .lineage import item_label  # lazy: lineage imports world
        return item_label(self, *item)

    def _on_loop(self, coro_fn):
        """Run an async action on the event loop, under the caller's key, from
        the worker thread ``Admin.apply`` runs in."""
        key = current_key()

        async def run():
            with use_key(*key):
                return await coro_fn()
        return asyncio.run_coroutine_threadsafe(run(), self._loop).result()

    def _correct(self, text: str, plan) -> None:
        topics = [rid.split("/")[-2] for s, rid in plan.wrong if s == "notes"]
        topic = topics[0] if topics else (next(
            (t for t, w in TOPICS.items() if any(x in text.lower() for x in w)), "general"))
        path = self._append_note(topic, f"Correction by the support lead: {text}")
        self.corrections.append({"id": f"correction:{plan.id}", "text": text, "note": path})

    def _cancel(self, item) -> None:
        self._on_loop(lambda: self.crm.call_tool(
            "update_ticket", {"ticket_id": item[1], "status": "cancelled"}))

    def _close_correction_turn(self) -> None:
        """The correction's own writes (its record, note, cancellations) are
        one turn, up to the first corrected answer."""
        c = getattr(self, "_open_correction", None)
        if c is not None:
            self._open_correction = None
            self.turns.append({"id": len(self.turns), "kind": "correction", "customer": None,
                               "message": c["text"], "correction": c["id"],
                               "start": c["start"], "end": self._seq()})

    def _question(self, rid: str) -> Tuple[str, str]:
        db = sqlite3.connect(self._crm_db)
        try:
            return db.execute("SELECT customer, question FROM replies WHERE id = ?",
                              (rid,)).fetchone()
        finally:
            db.close()

    def _rerun(self, item):
        """Ask the reply's question again, on a fresh thread (the old
        conversation is contaminated too), and send it as a correction."""
        self._close_correction_turn()
        customer, question = self._question(item[1])
        self._reruns = getattr(self, "_reruns", 0) + 1
        entry = self._on_loop(lambda: self.chat(
            customer, question, thread=f"{customer}-corrected-{self._reruns}", replaces=item[1]))
        return ("mcp:crm", entry["reply_id"])

    def _notify(self, item, text):
        """A reply to the bad message itself: re-running it would repeat the
        mistake, so send the correction's text."""
        self._close_correction_turn()
        customer, question = self._question(item[1])
        entry = self._on_loop(lambda: self._notice(customer, question, item[1], text))
        return ("mcp:crm", entry["reply_id"])

    async def correct(self, statement: str, wrong: List[Tuple[str, str]]) -> dict:
        """Plan and apply the fix for ``wrong`` through ``dod.admin``; the
        correction's turn is recorded around it."""
        self._loop = asyncio.get_running_loop()
        plan = self.admin.plan(statement, wrong)
        self._open_correction = {"id": f"correction:{plan.id}", "text": statement,
                                 "start": self._seq()}
        try:
            applied = await asyncio.to_thread(self.admin.apply, plan, "support-lead")
        finally:
            self._close_correction_turn()
        return {"plan": plan, "applied": applied}

    async def _notice(self, customer: str, question: str, replaces: str, text: str) -> dict:
        start = self._seq()
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


class _Rows:
    """A ``dod.admin.Reader`` from a generator function."""

    def __init__(self, fn):
        self.fn = fn

    def records(self):
        return list(self.fn())


def _result_id(tool: str, arguments: Any, result: Any) -> Optional[str]:
    """The CRM's ID for what a tool created (tickets, replies)."""
    try:
        return json.loads(result.content[0].text).get("id")
    except Exception:
        return None
