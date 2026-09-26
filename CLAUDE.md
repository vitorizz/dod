# CLAUDE.md

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
1. **CrewAI first.** It already emits `MemoryQueryCompleted` and `MemorySaveStarted` events; bind them into one edge with minimal glue.
2. Append-only store (SQLite to start). Capture overhead must be sub-millisecond.
3. Implement `trace` and `blast-radius`.
4. Add a second store (LangGraph `Store` or pgvector) to prove the cross-store join.
5. Then LangGraph, Pydantic AI, Anthropic `/memories` execute backend, MCP gateway.
6. Export edges as OpenTelemetry custom attributes in our own namespace (never `gen_ai.*`).

## Demo to build toward
Plant a false memory, let the agent derive 3–5 downstream memories across two stores, surface a confirmed bad output, trace to root, show blast radius, excise only the affected subgraph, re-run correctly. Compare side-by-side with "wipe everything."

## Biggest technical risk
**Capture completeness.** Any write path that bypasses the hook (direct DB insert, out-of-wrapper tool, compaction rewriting memory) silently breaks the guarantee. Make capture the only write path, or continuously reconcile store contents against the ledger.

## Business model
- Open-source capture library (self-host free).
- Paid hosted console: storage, retention, querying, rollback, audit.
- Priced like insurance: small monthly fee by volume/retention; investigations included.
- Keep the learned authority layer (which sources are trustworthy for which claim types) proprietary. That's the long-term moat.

## Competitive notes
- **memory-blackbox** (GitHub, lavkumarv): closest artifact, early stage. Records store-boundary provenance, not explicitly context-at-write. Watch it.
- Storage-layer provenance: Zep/Graphiti, Mem0, Letta. Different layer.
- Observability (Langfuse, LangSmith, Phoenix, AgentOps): no first-class memory-write span; don't model retrieval→write causality.
- Research: MemLineage, Dependency-Guided Rollback Repair (85.3% recovery), MemQ, MAP-Graph. The DAG idea is known; our edge is cross-store capture at the write boundary.

## Founder context
Background in LLM agents with retrieval memory and a precomputed state graph constraining valid actions (PAINT LAB research). Building part-time alongside a Cornell Tech MEng in NYC.

## Working conventions
- Keep responses concise.
- Prefer small, working increments over large scaffolds.
- Instrument everything: we'll run an experiment measuring how far false memories propagate.
