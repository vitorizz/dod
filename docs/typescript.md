# TypeScript: design options

Status: option A built (`ts/`, `@dod-lineage/capture`); see decisions.md (2026-11-06).

## Why

Many agent teams ship in TypeScript: the Vercel AI SDK, LangGraph.js, Mastra, the OpenAI Agents SDK for JS, MCP servers in Node. Today they can't capture without a Python sidecar.

## What has to exist in TypeScript

Capture is small. It needs:

- the scope (`AsyncLocalStorage` replaces `contextvars`);
- the per-scope context set and probes;
- the append call.

Everything else (`trace`, `blast_radius`, `excise`, `reconcile`, `audit`, the CLI) can stay in Python and read the same ledger.

## Options

**A. Native capture, same SQLite ledger format (recommended).**
A `@dod/capture` package writes the same schema with `better-sqlite3` (synchronous, in-process, fast enough for the 1 ms bar). Python tools read the file unchanged.
- Pros: no service to run, the same guarantees, and a mixed Python/TS deployment shares one ledger.
- Cons: two implementations of the append path (context-set hashing, Fenwick deltas, schema migrations) that must stay byte-compatible.
- Mitigation: a shared golden-file test suite. The same sequence of operations must produce identical rows in both languages.

**B. Capture over IPC to a Python daemon.**
TS sends retrieve, write and bind events over a Unix socket to `dod serve`.
- Pros: one implementation.
- Cons: a process to deploy and supervise, and IPC latency (~50–100 µs, so it fits the bar). When the daemon is down, capture fails open, which loses lineage.

**C. Through the MCP gateway only.**
TS agents that reach their stores through MCP already work today via `dod.integrations.mcp.Gateway`.
- Pros: zero new code.
- Cons: only covers MCP-reached stores, not framework memory or direct DB clients.

## Recommendation

C works now and is worth documenting as the answer for MCP-based agents. For everything else, A. Start with the append path plus scope, and one integration (the Vercel AI SDK's tool calls, mirroring `instrument_tools`). The golden-file suite comes first, so the two implementations can't drift.

## Questions to decide

- Which TS framework has the design partners?
- Is a Python dependency for queries (CLI, console) acceptable for TS teams, or must `trace` exist in TS too?
