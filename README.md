# dod

**Context management decides what goes into the context. We record what came out because of it.**

Agents with long-lived memory pick up false beliefs: a wrong fact from a user, a stale tool result, a poisoned document. Each time one is retrieved into context, the agent can write new memories derived from it, and those spread to further writes, sometimes in other stores. When a bad output finally surfaces, nothing records which memory caused it or what else it contaminated, so today the only reliable fix is wiping memory and losing everything the agent learned. `dod` records the one edge that can't be reconstructed after the fact, so you can find and remove only what's contaminated.

## How it works

At every write, `dod` records which retrieved items were in the model's context at that moment. Each write gets a row in an append-only SQLite ledger: write ID, content hash, timestamp, store, agent and session IDs, source (`user`, `tool`, `document`, `agent_inference`), and its parents, the `(store, record_id)` items that were in context, each tagged `in_prompt` or `retrieved_in_scope`.

Everything else is a query over that graph:

- `trace(ledger, store, record_id)` walks backward from a bad output to the writes it came from.
- `blast_radius(ledger, store, record_id)` walks forward to everything derived from an item, across stores.
- `ledger.quarantine(items)` marks items as contaminated (reversible with `ledger.release`); instrumented stores stop returning them, so they can't become parents of new writes.
- `excise(ledger, store, record_id)` quarantines (default) or, with `mode="delete"`, deletes an item and its whole blast radius across stores, leaving everything else untouched. `dry_run=True` shows the plan first.
- `regenerate(ledger, store, record_id, rewrite)` excises the same way, then calls your `rewrite(task)` for each derived item, oldest first, so it can be rebuilt from its clean inputs (see below).

```mermaid
flowchart LR
    subgraph CrewAI memory
        A["A: false fact<br/>(source: user)"]
        C["C: derived memory"]
    end
    subgraph LangGraph store
        B["B: derived item"]
    end
    A -- "recalled in step 1,<br/>in context at write" --> B
    B -- "read in step 2,<br/>in context at write" --> C
```

`trace(C)` reaches `A` through `B`; `blast_radius(A)` returns `B` and `C`. One `agent_task_scope` links retrievals and writes across every instrumented store, so the edge from a CrewAI recall to a LangGraph put is recorded like any other.

## Install

Not on PyPI yet. From a clone:

```bash
pip install -e ".[crewai]"      # CrewAI memory (crewai==1.15.22)
pip install -e ".[langgraph]"   # LangGraph Store (langgraph==1.2.14)
pip install -e ".[anthropic]"   # Anthropic memory tool (anthropic==1.12.1)
pip install -e ".[crewai,langgraph]"   # both, for the quickstart
pip install -e ".[otel]"        # OpenTelemetry export (opentelemetry-api/sdk==1.45.1)
pip install -e ".[mcp]"         # MCP client sessions and gateway (mcp==1.28.1)
pip install -e ".[pydantic_ai]" # Pydantic AI tool calls (pydantic-ai-slim==2.66.0)
pip install -e ".[postgres]"    # Postgres + pgvector through LangGraph's PostgresStore
```

The core library has no dependencies; integrations are optional extras.

## Quickstart

One `Capture` instruments a CrewAI `Memory` and a LangGraph `InMemoryStore`. A false fact is saved, two scoped steps derive new items from it across both stores, then the last one is traced back and everything derived from the false fact is quarantined. `toy_embed` is a stand-in embedder so the example runs offline; pass any embedding callable or crewai embedder config.

```python
from crewai.memory.unified_memory import Memory
from langgraph.store.memory import InMemoryStore

from dod.capture import Capture
from dod.integrations import crewai, langgraph
from dod.ledger import Ledger
from dod.query import blast_radius, trace
from dod.scope import agent_task_scope

ledger = Ledger(os.path.join(workdir, "ledger.db"))
cap = Capture(ledger)
memory = crewai.instrument(
    Memory(storage=os.path.join(workdir, "crewai"), embedder=toy_embed), cap
)
store = langgraph.instrument(InMemoryStore(), cap)
ns = ("users", "ana")

def remember(text, source=None):
    return memory.remember(text, scope="/", categories=[], importance=0.5,
                           source=source)

# A false fact gets saved.
with agent_task_scope("assistant", "onboarding"):
    root = remember("Ana is allergic to peanuts", source="user")

# Step 1: recall from CrewAI, write to LangGraph.
with agent_task_scope("assistant", "meal-plan"):
    memory.recall("what is Ana allergic to", depth="shallow")
    store.put(ns, "meal-plan", {"text": "no peanut dishes this week"})

# Step 2: read from LangGraph, write to CrewAI.
with agent_task_scope("assistant", "groceries"):
    store.get(ns, "meal-plan")
    bad = remember("Removed peanut butter from the grocery list")

# The bad output traces back to the false fact, across both stores.
t = trace(ledger, "crewai", bad.id)
print("trace roots:", [(w.store, w.write_id) for w in t.roots])
print("path:", [(e.parent_store, e.parent_record_id) for e in t.edges])

# Everything derived from the false fact.
b = blast_radius(ledger, "crewai", root.id)
print("blast radius:", b.store_counts)
affected = sorted({item for items in b.bindings.values() for item in items})

# Quarantine (reversible): instrumented CrewAI recall stops returning them.
ledger.quarantine(affected, reason="false allergy", actor="oncall")
with agent_task_scope("assistant", "after"):
    after = memory.recall("what is Ana allergic to", depth="shallow")
print("recall after quarantine:", [m.record.content for m in after])
```

Run it with `python examples/quickstart.py`. `tests/test_quickstart.py` runs the same file and checks this block still matches it.

Inside a crew, call `crewai.install_task_scopes()` once instead of entering scopes by hand: every task then runs in `agent_task_scope(agent.id, task.id)`.

## OpenTelemetry

`OTelObserver` adds each captured write to the traces you already collect: a `dod.memory_write` event on the current span (or a short span of that name if none is active), plus `dod.memory_bind` and `dod.memory_delete`.

```python
from opentelemetry import trace

from dod.capture import Capture
from dod.export.otel import OTelObserver

cap = Capture(ledger, observers=[OTelObserver()])
tracer = trace.get_tracer("my-agent")

with tracer.start_as_current_span("agent-step"):
    ...  # instrumented recalls and writes add dod.memory_write events here
```

Attributes are IDs, hashes and tags only (`dod.write.id`, `dod.write.content_hash`, `dod.parents.store`/`record_id`/`tag`, ...; see `dod/export/otel.py`), never memory content, and never `gen_ai.*` names. The observer only calls the OpenTelemetry API: where the data goes is decided by your own SDK and exporter configuration, and without one nothing is sent. Observers run after the ledger commit; one that raises is logged and never breaks capture.

## MCP

Agents increasingly write through MCP tools (tickets, CRM records, documents) rather than a memory store. `dod.integrations.mcp` captures those calls, either by instrumenting a `ClientSession` your harness owns or by running a gateway in front of your MCP servers, so any MCP client is captured without changes:

```python
from dod.integrations import mcp

# In your harness:
mcp.instrument(session, cap, "crm", record_id=lambda tool, args, result: ...)

# Or as a proxy: clients connect to the gateway; tools appear as crm__<tool>.
gateway = mcp.Gateway(cap, {"crm": crm_session, "docs": docs_session})
await gateway.run_stdio()
```

Each server is a store (`mcp:<server>`). Tools annotated `readOnlyHint` are reads: their result is recorded as an item (source `tool`) and enters the context `in_prompt`, so what the model writes from it traces back to the call. Every other tool is a write, recorded before the call and bound after it succeeds (to the ID `record_id` returns, or the call's own). Resource reads are context items keyed by URI; quarantined resources read as "not found". Through the gateway, each client connection is one unit of work, unless a request names its own with `_meta` keys `dod/agent_id` and `dod/task_id`.

## Pydantic AI

Pydantic AI agents remember through tools and through the message history passed back into each run. Wrap the agent's toolset:

```python
from pydantic_ai import Agent
from dod.integrations.pydantic_ai import CaptureToolset

tools = CaptureToolset(my_toolset, cap, read_only=["lookup_customer"],
                       passthrough=["remember"],  # writes to an instrumented store
                       record_id=lambda tool, args, result: result.get("id"))
agent = Agent(model, toolsets=[tools])
```

Each run is a unit of work. Read tools' results are recorded as items, and every read result still in the message history, including from earlier runs, is a parent (`in_prompt`) of the run's writes. Other tools are writes, recorded before the call and bound after it. `passthrough` tools are only scoped, so stores they use through another dod integration link to the run's context.

## Postgres

LangGraph's `PostgresStore` (with pgvector semantic search) keeps `BaseStore`'s routing, so the LangGraph integration captures it unchanged:

```python
from langgraph.store.postgres import PostgresStore

with PostgresStore.from_conn_string(uri, index={"dims": 1536, "embed": embed}) as pg:
    pg.setup()
    memory = langgraph.instrument(pg, cap, store_name="postgres")
```

Vector search results become parents like any other read, quarantined rows drop out of `get` and `search`, `excise(mode="delete")` deletes rows through the store, and `reconcile` reports rows written with raw SQL.

## LangGraph agents

`langgraph.instrument` captures a LangGraph `Store`. `langgraph_graph.instrument_graph` also captures the graph itself:

```python
from dod.integrations import langgraph, langgraph_graph

store = langgraph.instrument(InMemoryStore(), cap)
app = langgraph_graph.instrument_graph(builder.compile(checkpointer=saver, store=store), cap)
app.invoke({"messages": [user_msg]}, {"configurable": {"thread_id": "ana"}})
```

Every run is scoped to `(graph name, thread_id)`, so no `agent_task_scope` is needed around it. Each state channel of a thread is an item (`langgraph_state`): the run's input is recorded with source `user`, each node's update as a write, and the thread's saved state is context for the run. A memory a node writes to the store then traces back to the user message it came from.

## Reconcile

Capture only sees writes that go through an instrumented call. `reconcile` checks that nothing else changed your stores. It reads each store, compares it with the ledger, and reports every item they disagree on:

```python
from dod.integrations import anthropic_memory, crewai, langgraph
from dod.reconcile import reconcile

report = reconcile(ledger, {
    "crewai": crewai.ReconcileAdapter(memory),
    "langgraph": langgraph.ReconcileAdapter(store),
    "anthropic_memory": anthropic_memory.ReconcileAdapter(tool),
})
for f in report.findings:
    print(f.kind, f.store, f.record_id, "read into", len(f.read_into), "recorded writes")
```

The four kinds of finding:

- `unrecorded`: in the store but never recorded. It was written outside capture.
- `resurrected`: in the store, but recorded as deleted.
- `missing`: recorded as live, but gone from the store. It was deleted outside capture, for example by a TTL sweep.
- `modified`: content differs from the last recorded write. For the memory tool this needs `anthropic_memory.instrument(tool, capture, hash_files=True)` (off by default). That records each written file's hash after the command, at the cost of one extra `view` per file written. Without it, the memory tool is checked for existence only.

For `unrecorded` and `resurrected` items, `read_into` lists the recorded writes that already used the item as a parent. Those are the points where `trace` currently stops.

Reconcile is read-only. It never records the reads it makes and never changes a store. Each finding is re-checked after `grace` seconds (default 1), so writes still in flight aren't reported. One pass over 100k items takes about a second. Run it on a schedule to keep the capture guarantee checked.

## Regenerate

Excising loses what the agent learned downstream of the bad item, including the parts that were right. `regenerate` keeps them: it quarantines the bad item and everything derived from it, then hands each derived item to your function, parents first. The library never stores content or calls a model, so the rewrite is yours. Write the replacement as a new record through an instrumented store and return its `(store, record_id)`, or `None` to drop it:

```python
from dod import regenerate

def rewrite(task):
    # task.parents: each original parent with its status (clean, bad,
    # replaced, dropped); task.clean_parents(): the inputs you may use.
    inputs = [memory._storage.get_record(r) for _, r in task.clean_parents()]
    memory.recall("...")                      # retrievals here become the new parents
    new = memory.remember(my_llm_rewrite(inputs), scope="/", categories=[], importance=0.5)
    return ("crewai", new.id)

result = regenerate(ledger, "crewai", root.id, rewrite)
print(result.replaced, result.dropped, result.failed)
```

Each rewrite runs in its own `agent_task_scope`, so capture records it with whatever it retrieved as parents. Quarantined items can't be retrieved, so a rewrite can't pick the bad item back up. After each one, the replacement's recorded parents are checked: a replacement that wasn't captured, overwrites an old item in place, or was derived from a target is quarantined and reported in `failed`. Old items stay quarantined (or `then="delete"` deletes them through delete adapters); `dry_run=True` returns the ordered plan.

## Audit

Open the ledger with `Ledger(path, audit=True)` (every time, once enabled) to hash-chain it; the first audited open seals the rows already in the file. `dod.audit.report(ledger, store, record_id)` is the evidence for an excision: the item and everything derived from it, each with its writes, quarantine actions and deletions (time, reason, actor), the audit entry that recorded each, and the chain's verification at report time. Like the ledger, it holds IDs, hashes and timestamps only. `dod.audit.to_markdown(report)` renders it.

```python
from dod import audit

audit.verify(ledger).ok                     # True unless the file was tampered with
anchor = audit.head(ledger)                 # store outside the ledger
rep = audit.report(ledger, "crewai", bad_id)
print(audit.to_markdown(rep))
```

## Guarantees

- **Content-free ledger.** The ledger stores IDs, content hashes, edges, and metadata only. Memory content stays in your stores. (`reason` and `actor` on quarantine are stored as you pass them; keep content out of them.)
- **Append-only.** Writes, parents, bindings, deletions (tombstones), and quarantine actions are only ever inserted; schema triggers reject updates, deletes, and `INSERT OR REPLACE`. Quarantine and release are new rows, so every action stays auditable.
- **Tamper-evident (opt-in).** With `Ledger(path, audit=True)`, every transaction also appends a hash-chained audit entry over the rows it added (about 0.04 ms per append). `dod.audit.verify(ledger)` recomputes the chain and reports edited or deleted rows, broken links, and rows inserted outside the library. Anchor `dod.audit.head(ledger)` somewhere outside the file (your logs, an audit export) and `verify(ledger, expected_head=...)` also proves nothing was cut off the end.
- **No network calls home.** The library makes no telemetry or network calls. (Framework packages may have their own telemetry; the example turns off crewai's.)
- **p99 < 1 ms capture overhead per write**, asserted by benchmarks in the test suite (`pytest -q tests/test_*_perf.py -s`).

What it does **not** do:

- **No automated detection of bad memories.** You identify the bad output; `dod` traces it to its origin and shows what it reached.
- Quarantine is enforced by the instrumented stores (CrewAI recall, LangGraph `get`/`search`, memory tool `view`); reads that bypass them see quarantined items.
- Writes that bypass an instrumented store (direct database inserts, uninstrumented tools) are not captured; `reconcile` finds them afterwards.

## Status

Supported:

- **CrewAI 1.15.22** `Memory`: `recall`, `remember`, `remember_many`, `update`, `forget`, `reset`, `reset_all`, and consolidation on save. Rewrites are recorded as compactions; every deleted record gets an append-only tombstone, so `trace` and `blast_radius` report what was deleted and why.
- **LangGraph Store** (langgraph 1.2.14, langgraph-checkpoint 4.2.0): `get`, `search`, `put`, `delete`, and async variants, on stores that keep `BaseStore`'s default routing (such as `InMemoryStore`, and `PostgresStore` with pgvector from langgraph-checkpoint-postgres 3.1.2).
- **LangGraph graphs** (langgraph 1.2.14): `invoke`/`ainvoke`/`stream`/`astream` scoped per thread; input, node updates and loaded thread state as `langgraph_state` items.
- **Anthropic memory tool** (anthropic 1.12.1, `memory_20250818`): `view`, `create`, `str_replace`, `insert`, `delete`, `rename`, and `clear_all_memory`, on `BetaAbstractMemoryTool`/`BetaAsyncAbstractMemoryTool` subclasses that keep the default `call` (such as `BetaLocalFilesystemMemoryTool`). A viewed file is an `in_prompt` parent; the record ID is the memory path.
- **MCP** (mcp 1.28.1): `ClientSession` `call_tool`, `read_resource` and `list_tools`, and a proxying `Gateway` server.
- **Pydantic AI** (pydantic-ai-slim 2.66.0): any toolset, through `CaptureToolset`, with message history as context across runs.
- `trace`, `blast_radius`, quarantine/release, `excise` (quarantine or delete) and `regenerate` (callback rewrite), across every store through one ledger.
- `reconcile`: finds writes, edits and deletions made outside capture, in every store with a reconcile adapter (CrewAI, LangGraph Store, memory tool).
- OpenTelemetry export of every write, bind and deletion as `dod.*` span events (opentelemetry-api 1.45.1).

Next: more integrations (other frameworks, more MCP-reachable stores).

## Demo

**The support-agent demo** (`demo/`, see [demo/README.md](demo/README.md)) is the full version: a LangGraph agent whose memory spans Postgres + pgvector, an MCP CRM server, memory-tool files and its own conversation state, with a web console to trace a wrong reply to a customer's false claim, see its blast radius across all four stores, and excise or regenerate only that. It runs with a scripted model, Ollama/any OpenAI-compatible server, or Claude:

```bash
pip install -e ".[langgraph,anthropic,mcp,postgres]"
python -m demo.support_agent.app   # then open http://localhost:8000
```

The original, terminal-only demo:

A support agent keeps memory in CrewAI and a LangGraph store, with 12 correct items. One customer message gets saved as a "policy update" (refunds within 90 days; the real policy is 30), and scripted steps derive four items from it across both stores. The last one is a wrong reply to a customer. The demo traces that reply back to its roots, shows the blast radius per store, quarantines only that subgraph, and re-runs the reply. Then it compares the result with wiping everything.

```bash
pip install -e ".[crewai,langgraph]"
python examples/false_memory_demo.py
```

It runs in seconds, offline, with no LLM calls (the agents are fixed rules), and prints the same output every time. It ends with:

```
== 6. Wipe everything vs. excise ==================================
Wipe both stores:        loses all 17 items, including 12 correct ones.
Roll back to the root:   loses 3 correct items learned after it.
Excise its subgraph:     removes 5, keeps all 12 correct items.

Summary: excised 5 items, preserved 12, stores touched: crewai, langgraph.
```

`tests/test_demo.py` runs the demo and checks these numbers.
