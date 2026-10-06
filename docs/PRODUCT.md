# Product background

Background for humans and agents. Rules and commands live in `CLAUDE.md`.

## What we're building
A capture library for AI agents with long-lived memory. At every memory write, it records which retrieved items were in the model's context at that moment. That single edge turns agent memory into a traceable lineage graph.

One-liner: **Your agent learned something false. We tell you which belief, and everything it touched.**

## Why it matters
- Agents with persistent memory accumulate false beliefs (wrong facts, stale info, poisoned inputs).
- When a bad output surfaces, teams can't tell which memory caused it or what else it contaminated.
- Today's only fix is wiping memory and losing months of learning.
- The context-to-write edge is the one thing that **cannot be reconstructed after the fact**. If it isn't captured at write time, it's gone.

## Core primitive
At each memory write, record:
- `write_id`, content hash, timestamp, store, agent/session id
- `parent_ids`: IDs of retrieved items present in context when the write happened
- source (user / tool / document / agent inference)

Everything else is a query over this graph:
- `trace <bad_output>`: walk backward to the originating memory
- `blast-radius <memory_id>`: walk forward to everything derived from it
- `excise <memory_ids>`: remove the contaminated subgraph, keep the rest

## Positioning (important)
- **Sits at the orchestration/harness layer**, not the storage layer. The database only sees the finished write; it never sees the context window.
- **Cross-store**: real deployments span Mongo, Postgres, vector DBs, framework state. No single DB vendor can join across them.
- **Model-agnostic**: works with Claude, OpenAI, or self-hosted open-weight models.
- **Sell correctness, not degradation.** Target false beliefs, not context fuzziness (bigger windows will absorb fuzziness).
- **Sell trace + surgical excision. Never promise automated detection of bad memories.** Detection is unreliable (best methods ~14% step-level accuracy); tracing from a recorded edge is cheap and reliable.

## Build plan
1. **CrewAI first.** Wrap the `Memory` instance's `recall` and `remember` (synchronous, exact record IDs), scoping context per agent/task with `contextvars` so concurrent agents never cross-link parents. Don't bind `MemoryQueryCompleted` → `MemorySaveStarted` events: the save events don't carry the record ID, and handlers run in a thread pool, so they can arrive out of order. Pinned to crewai 1.15.22; bump deliberately and re-run the integration test.
2. Append-only store (SQLite to start). Capture overhead must be sub-millisecond.
3. Implement `trace` and `blast-radius`.
4. Add a second store (LangGraph `Store` or pgvector) to prove the cross-store join.
5. Then LangGraph, Pydantic AI, Anthropic `/memories` execute backend, MCP gateway.
6. Export edges as OpenTelemetry custom attributes in our own namespace (never `gen_ai.*`).

## Demo to build toward
Plant a false memory, let the agent derive 3–5 downstream memories across two stores, surface a confirmed bad output, trace to root, show blast radius, excise only the affected subgraph, re-run correctly. Compare side-by-side with "wipe everything."

## Biggest technical risk
**Capture completeness.** Any write path that bypasses the hook (direct DB insert, out-of-wrapper tool, compaction rewriting memory) silently breaks the guarantee. Make capture the only write path, or continuously reconcile store contents against the ledger.
