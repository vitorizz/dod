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

## Stack
- Python >= 3.9 (CI runs 3.12). Package `dod` in `src/` layout, built with hatchling (`pyproject.toml`).
- Ledger: stdlib `sqlite3`, append-only, WAL mode. No ORM.
- Tests: pytest. Benchmarks are plain pytest tests using `tests/perf.py` (p50/p99 via `time.perf_counter_ns`).
- Framework integrations are optional extras, never core dependencies: `[crewai]` pins `crewai==1.15.22`.
- CI and agents: GitHub Actions (`ci.yml`, `claude.yml`, `claude-code-review.yml`, `project-board.yml`).

## Test commands
```bash
pip install -e ".[dev]"              # install package + test deps
pytest -q                            # full suite, including p99 < 1 ms capture benchmarks
pytest -q tests/test_*_perf.py -s    # benchmarks only, printing p50/p99
pip install -e ".[dev,crewai]"       # add the real CrewAI dependency
pytest -q -m crewai                  # CrewAI integration tests (skipped without crewai)
```
In agent runs, use these exact forms. Don't create a venv or call pip through `python -m`: they are not in the allowed tools and get denied. (`python -m pytest` is allowed.)
Benchmark files and the `crewai` marker arrive with #11 and #13.

## Workflow
1. Every change starts as an issue using the agent-task template (Goal / Acceptance criteria / Files involved), sized for one PR. New issues go on the board as Backlog with Priority and Size.
2. A human approves by adding the `agent-ready` label. The board moves the issue to Ready, then to In progress when Claude starts.
3. Claude works on a `claude/issue-<N>-*` branch and pushes. A workflow step opens the PR with `Closes #N` (using `PAT_TOKEN`, so CI and review trigger).
4. CI (pytest + benchmarks) and Claude Code Review run on the PR. Fix findings with `@claude` comments on the PR.
5. A human merges. Merging closes the issue.

If blocked or unsure (ambiguous acceptance criteria, a design decision the issue doesn't settle, a conflict with this file, a missing dependency or permission):
1. Comment the specific question on the issue: `gh issue comment <N> --body "..."`. Say what you need decided, the options you see, and what you'd need to proceed.
2. Add the label: `gh issue edit <N> --add-label needs-human`.
3. Stop. No more commits and no PR; the PR-opening step skips issues labeled `needs-human`.

The human answers with an `@claude` comment on the issue. That run removes `needs-human` before Claude starts, so continue from the answer; the PR-opening step then works again.

**Never guess on design decisions.** Data model, APIs, capture semantics, dependencies and anything recorded in `docs/decisions.md` get asked about, not assumed.

Rules for every PR:
- New or changed behavior comes with tests. Anything on the capture path includes or updates a benchmark asserting p99 < 1 ms.
- One issue, one PR. Stay within "Files involved"; if more is needed, say so in the PR instead of expanding scope.
- Don't edit `.github/workflows/` from agent runs (the GitHub App can't push workflow changes).
- Keep strategy (pricing, positioning, competitors) out of code, comments, and commit messages.
- Record decisions in `docs/decisions.md`, one line of why each.

Hard rules (never break, in any PR):
- The ledger never stores memory content. IDs, content hashes, edges, and metadata only; content stays in the customer's stores.
