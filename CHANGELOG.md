# Changelog

## 0.1.0 (unreleased)

First release of `dod-lineage` (Python, `import dod`) and `@dod-lineage/capture` (TypeScript).

### Capture
- An append-only, content-free SQLite ledger. Every memory write records the items that were in context when it happened, tagged `in_prompt` or `retrieved_in_scope`.
- p99 under 1 ms per captured write.
- `agent_task_scope` links retrievals and writes across stores, and keeps concurrent agents apart.
- Integrations, each an optional extra:
  - CrewAI
  - LangGraph `BaseStore` (including `PostgresStore`) and graph state
  - LangChain vector stores
  - Mem0 (sync and async)
  - the Anthropic memory tool
  - MCP sessions and an MCP gateway
  - Pydantic AI toolsets
  - OpenAI Agents SDK function tools
- Prompt confirmation (`dod.prompt.confirm`, `dod.integrations.llm`) upgrades the retrieved items that actually reached the prompt to `in_prompt`.
- OpenTelemetry export as `dod.*` span events.
- TypeScript capture for Node 22.13+. It writes the same ledger format and includes a tool wrapper for Vercel AI SDK–style tools.

### Queries and repair
- `trace`: walk back from an item. Version-aware.
- `blast_radius`: walk forward from an item, with `in_prompt_only` and `precise` (per-version) options.
- `excise`: quarantine, or delete through store adapters.
- `regenerate`: excise, then rebuild derived items through a callback.
- `reconcile`: compare store contents with the ledger, to check capture completeness.
- `dod.admin`: trace and fix from a statement instead of record IDs (find what contradicts it, plan from its origin, apply, undo), and the same as `@commands` (`@trace`, `@impact`, `@excise`, `@undo`, `@ledger`) for a chat or Slack bot. Your app supplies readers and the side effects. A plan keeps one origin per chain (an origin another one reached is derived), and each output is corrected once.
- Cross-host lineage: `dod ship` copies each node's ledger to Postgres. `PgLineage` and `dod pg` trace, show blast radius and excise across nodes, and quarantines fan out to every node.

### Audit and operations
- An opt-in hash-chained ledger (`audit=True`), with `dod verify` and per-excision reports (`dod report`).
- `dod doctor`: schema, WAL size, index lag, unbound writes and audit status.
- The `dod` CLI: `trace`, `blast-radius`, `excise`, `release`, `quarantined`, `stats`, `doctor`, `verify`, `report`, `ship`, `pg`.
- The WAL stays bounded under continuous writes.

### Guarantees
- The ledger stores IDs, content hashes, edges and metadata, never content. A test pushes a canary through every integration and checks no ledger file contains it.
- No telemetry and no network calls from the library.
