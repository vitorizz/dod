# CLAUDE.md

## What we're building
A capture library for AI agents with long-lived memory. At every memory write, it records which retrieved items were in the model's context at that moment. That single edge turns agent memory into a traceable lineage graph.

One-liner: **Context management decides what goes into the context. We record what came out because of it.**

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
- `regenerate <memory_id>`: excise, then rebuild each derived item from its clean inputs through a caller callback

## Positioning (important)
- **Lineage for long-lived agent context.** Context managers decide what goes into context; we record what came out because of it, and clean only what's contaminated.
- **Complementary, not competing**, with Mem0, Supermemory, Zep, and markdown-based memory: they decide what goes in; none record lineage.
- **Any agent write**: memory, DB, CRM, tickets. Not just memory stores.
- **Target**: teams building long-lived agents with memory for their users (personal agents, AI companions, support agents, "agent employees"). Opening pitch: real forgetting, deleting a memory plus everything derived from it.
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

The demo lives in `demo/support_agent/` (not part of the package): a LangGraph support agent over Postgres/pgvector, an MCP CRM server process, memory-tool files and graph state, with a Starlette web console (`python -m demo.support_agent.app`); `lineage.py` turns the ledger into the console's turn-by-turn view. `demo-v1` is the first version, kept as a branch. `tests/test_demo_support_agent.py` runs its story with the scripted model.

## Biggest technical risk
**Capture completeness.** Any write path that bypasses the hook (direct DB insert, out-of-wrapper tool, compaction rewriting memory) silently breaks the guarantee. Make capture the only write path, or continuously reconcile store contents against the ledger.

## Business model
- Free, open-source capture library. Security teams can audit exactly what capture records.
- Paid self-hosted system, running in the customer's cloud: console, cross-store connectors, excision/rollback, reconciliation, audit exports, access controls. We never see customer data.
- Paid pilots first, with 3–5 design partners. Then an annual license: base fee plus a metric that scales with the customer (writes tracked or active end users), not seats. Compliance pack (proof of real deletion, audit exports) as an add-on.
- Optional hosted version later, for smaller customers.
- The authority layer (which sources are trustworthy for which claim types) learns per deployment; opt-in anonymized sharing is a later option. The moat is cross-store connectors and reconciliation.

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
- Ledger: stdlib `sqlite3`, append-only, WAL mode (autocheckpoint off: a background thread checkpoints PASSIVE every interval, and the writer TRUNCATEs the WAL, without waiting, after a commit once it passes 64 MiB). No ORM. The lineage index (integers, for `blast_radius`) lives in a companion file, `<ledger>-lineage`, filled by the ledger's background thread; it is derived, so it is rebuilt (never migrated) when missing, stale or at another `INDEX_VERSION`.
- Tests: pytest. Benchmarks are plain pytest tests using `tests/perf.py` (p50/p99 via `time.perf_counter_ns`). Bars are CI's; `DOD_PERF_SCALE` (default 1.0) divides measured times so a slower dev machine can scale them, and benchmarks that time themselves multiply their bar by `perf.SCALE`.
- Framework integrations are optional extras, never core dependencies: `[crewai]` pins `crewai==1.15.22`; `[langgraph]` pins `langgraph==1.2.14` and `langgraph-checkpoint==4.2.0` (where `BaseStore` lives); `[langchain]` pins `langchain-core==1.7.5` (also installed by `[langgraph]`); `[mem0]` pins `mem0ai==2.0.5` (the last release whose posthog range fits crewai's chromadb); `[anthropic]` pins `anthropic==1.12.1` (memory tool `memory_20250818`, `anthropic.tools.memory`); `[otel]` pins `opentelemetry-api==1.45.1` and `opentelemetry-sdk==1.45.1` (the SDK for tests); `[mcp]` pins `mcp==1.28.1` (the version crewai 1.15.22 requires, so both install together); `[pydantic_ai]` pins `pydantic-ai-slim==2.66.0`; `[openai_agents]` pins `openai-agents==0.20.0` (the last release on openai 2.x, which crewai requires); `[postgres]` pins `langgraph-checkpoint-postgres==3.1.2` (the last release that accepts langgraph-checkpoint 4.2.0) and `psycopg[binary]==3.3.8`, for LangGraph's `PostgresStore`, which the LangGraph integration captures unchanged.
- Exporters are `Capture` observers in `dod.export` (`Capture(ledger, observers=[...])`, called after the ledger commit, failing open). `dod.export.otel.OTelObserver` emits `dod.*` span events only, never `gen_ai.*`.
- With `Ledger(path, audit=True)` (opt-in, off by default so capture pays nothing) the ledger is hash-chained (schema 6, `audit_chain`): every appending transaction adds an entry with the id ranges it added per table and sha256(prev digest, those rows read back as JSON in one SQL statement). Context members aren't hashed directly; `dod.audit.verify` recomputes each set's content address instead. The first audited open seals the rows already in the file in one entry. Audited benchmarks are in `tests/test_audit_perf.py`. `dod.audit.report` is the per-excision evidence export.
- `dod.reconcile.reconcile(ledger, {store: adapter})` checks capture completeness: each integration has a `ReconcileAdapter` (read-only, around capture, `scan()`/`get()`), alongside its `DeleteAdapter` for `dod.excise` (and `dod.regenerate`'s `then="delete"`). Bindings carry an optional `content_hash` (schema 5) for integrations whose write hash isn't the stored content; the memory tool fills it only with `hash_files=True` (opt-in, off by default).
- `dod.regenerate.regenerate(ledger, store, record_id, rewrite)` quarantines `excise`'s targets, then calls `rewrite(task)` per derived item (parents first, each in its own scope) and checks each returned replacement is a new, captured record with no contaminated direct parent.
- `dod.integrations.mcp` wraps an MCP `ClientSession` (`instrument`) or proxies upstream sessions as an MCP server (`Gateway`); each server is store `mcp:<server>`. The gateway keys context per client connection (`dod.scope.use_key`, which sets a key without resetting it) unless a request's `_meta` names `dod/agent_id`/`dod/task_id`.
- `dod.integrations.pydantic_ai.CaptureToolset` wraps any Pydantic AI toolset: unit of work = run (`(agent name, run_id)`), read tools' results become `call:<run_id>:<tool_call_id>` items, and read results in the message history are `in_prompt` context; `passthrough` tools are only scoped.
- `dod.integrations.langgraph_graph.instrument_graph` scopes a compiled graph's runs to `(graph name, thread_id)` and records state channels (store `langgraph_state`): input as `user`, top-level node updates as `agent_inference`, observed with a LangChain callback (synchronous, unlike checkpointer writes); the thread's saved state (`get_state`) is the run's starting context.
- Prompt confirmation: integrations pass each retrieved item's content to `Capture.on_retrieve(texts=...)`, held in memory only as short normalized probes (`dod.capture.probes`, first 64 chars per string); `dod.prompt.confirm(text)` (or the client wrappers in `dod.integrations.llm`) upgrades items whose probe appears in the prompt to `in_prompt`. `blast_radius`/`excise(in_prompt_only=True)` follow only `in_prompt` edges; `precise=True` (post-filter over the walk's edges, in `seq` order) drops writes that only read an earlier version of a reached item.
- `dod.integrations.langchain.instrument(vectorstore, capture, store="langchain")` wraps any LangChain `VectorStore` instance's read/write/delete methods as instance attributes; a contextvar guard records only the outermost call (`add_texts` → `add_documents`, `similarity_search` → `..._with_score`). Record ID = `Document.id`.
- `dod.cli` is the `dod` command (`[project.scripts]`, also `python -m dod`): `trace`, `blast-radius`, `excise` (quarantine only), `release` (refuses items that aren't quarantined), `quarantined`, `stats` (read-only), `verify`, `report`, `ship`, `pg` (trace/blast-radius/excise/release/quarantined across nodes), each with `--json`; it opens an audited file with `audit=True`; the ledger is `--ledger` or `$DOD_LEDGER`. The PyPI distribution is `dod-lineage` (`dod` is taken). Publishing: `release.yml` (PyPI trusted publishing on a `v*` tag), added in its own workflow-only PR; the license is Apache-2.0 (`LICENSE`, PEP 639 `license` field, hatchling >= 1.27); it needs a PyPI trusted publisher before the first tag, and a human's go-ahead.
- `dod.integrations.openai_agents.instrument_tools(tools, capture, read_only=, passthrough=, record_id=, deleted=)` wraps `FunctionTool.on_invoke_tool` in place, with `CaptureToolset`'s rules. Outside a scope the run key is `(agent name, "run-<n>")`, keyed by the run's shared `Usage` object (cleared by `weakref.finalize`); read outputs in `ctx.turn_input` (earlier runs) are `in_prompt` context. Tests drive real `Runner.run` with a scripted `Model` (no network).
- `dod.integrations.mem0.instrument(memory, capture, store="mem0")` wraps a `mem0.Memory` or `AsyncMemory` (one `_wrap` builds sync or async wrappers): the matches Mem0's own `add` retrieves (its `vector_store.search`, put in the extraction prompt) are `in_prompt` parents of each added memory; an `add` outside a scope gets a temporary one. Deletes are recorded in the wrapped `_delete_memory`.
- `dod.ship`: each node keeps its local SQLite ledger; `Shipper` (run as `dod ship`, its own process, so capture never competes for the GIL) copies rows by watermark (`<prefix>ship_state`) to Postgres tables with a `node_id`, idempotently, and pulls other nodes' quarantine rows into the local ledger (actor `dod-ship:<node>`, never shipped back). `PgLineage.trace`/`blast_radius` are recursive CTEs over the `<prefix>parents` view; `PgLineage.excise`/`quarantine`/`release` (and `dod pg ...`) append rows under node `dod-console`, which every shipper pulls; across nodes, trace orders versions by `ts`. Tests (`test_ship*.py`, marked `postgres`) use a fresh table prefix each and check results equal the local queries on a random graph.
- `ts/` is the TypeScript capture package (`@dod-lineage/capture`, Node >= 22.13, built-in `node:sqlite`, no runtime deps): `Ledger`, `Capture`, `agentTaskScope` (AsyncLocalStorage), `instrumentTools`. It writes the Python ledger format: the schema is generated into `ts/src/schema.ts` by `scripts/export_ts_schema.py` (`tests/test_ts_compat.py` fails if stale); content addresses, hashes and write IDs match Python's; context sets are stored as full sets; audited ledgers are refused; WAL checkpoints run on a worker thread. `tests/test_ts_compat.py` runs `ts/test/golden.mjs` and reads its ledger with Python (skips without node and a built `ts/`).
- The context scope (`agent_task_scope`) lives in `dod.scope` and is shared by every integration, so one scope links retrievals and writes across stores.
- CI and agents: GitHub Actions (`ci.yml`, `claude.yml`, `claude-code-review.yml`, `project-board.yml`, `update-branches.yml`).

## Test commands
```bash
pip install -e ".[dev,crewai,langgraph,langchain,anthropic,otel,mcp,pydantic_ai,postgres,mem0,openai_agents]"   # what CI runs: package, test deps, real integrations
pip install -e ".[dev]"              # core only (CI also runs this, as test-core)
pytest -q                            # full suite, including p99 < 1 ms capture benchmarks
pytest -q tests/test_*_perf.py -s    # benchmarks only, printing p50/p99
DOD_PERF_SCALE=2 pytest -q           # slow dev machine: every benchmark bar x2 (CI never sets it)
pip install -e ".[dev,crewai]"       # add the real CrewAI dependency
pytest -q -m crewai                  # CrewAI integration tests (skipped without crewai)
pip install -e ".[dev,langgraph]"    # add LangGraph (InMemoryStore needs no service)
pytest -q -m langgraph               # LangGraph + cross-store tests (skipped without langgraph)
pip install -e ".[dev,anthropic]"    # add the Anthropic SDK (memory tool; no API calls in tests)
pytest -q -m anthropic               # Anthropic memory tool tests (skipped without anthropic)
pytest -q -m langchain               # LangChain VectorStore tests (langchain-core, installed with langgraph)
pip install -e ".[dev,mem0]"         # add Mem0 (local qdrant, scripted LLM; MEM0_TELEMETRY=False; no network)
pytest -q -m mem0                    # Mem0 tests (skipped without mem0ai)
pip install -e ".[dev,otel]"         # add the OpenTelemetry API + SDK (in-memory exporter; no network)
pytest -q -m otel                    # OTel export tests (skipped without otel)
pip install -e ".[dev,mcp]"          # add the MCP SDK (in-memory transport; no network)
pytest -q -m mcp                     # MCP session + gateway tests (skipped without mcp)
pip install -e ".[dev,pydantic_ai]"  # add Pydantic AI (FunctionModel; no network)
pytest -q -m pydantic_ai             # Pydantic AI tests (skipped without pydantic-ai-slim)
pip install -e ".[dev,openai_agents]"  # add the OpenAI Agents SDK (scripted Model; no network)
pytest -q -m openai_agents           # OpenAI Agents SDK tests (skipped without openai-agents)
pip install -e ".[dev,postgres]"     # add PostgresStore (needs a server with pgvector)
DOD_TEST_POSTGRES_URI=postgresql://... pytest -q -m postgres   # skipped without the URI
cd ts && npm ci && npm test          # TypeScript package: build, unit tests, p99 benchmark
python scripts/export_ts_schema.py   # after a schema change: regenerate ts/src/schema.ts
```
In agent runs, use these exact forms. Don't create a venv or call pip through `python -m`: they are not in the allowed tools and get denied. (`python -m pytest` is allowed.)
CI's `test` job installs every integration (`crewai`, `langgraph`, `langchain`, `anthropic`, `otel`, `mcp`, `pydantic_ai`, `postgres`, `mem0`, `openai_agents`, with `MEM0_TELEMETRY=False`) and runs a pgvector Postgres service (`DOD_TEST_POSTGRES_URI`), so every test runs; `test-core` runs without them to keep the core dependency-free. CrewAI unit tests use a stub mirroring the pinned `Memory` signatures; tests that need the real package are marked `crewai` and skip without it. LangGraph tests run against the real `InMemoryStore` (and compiled `StateGraph`s with `InMemorySaver`), are marked `langgraph`, and skip without it. Memory tool tests run `BetaLocalFilesystemMemoryTool` in a temp dir via `tool.call({...})` (no network), are marked `anthropic`, and skip without it. OTel tests use the SDK's `InMemorySpanExporter` with their own `TracerProvider` (never the global one), clear `OTEL_SDK_DISABLED` (which `test_crewai` sets process-wide), are marked `otel`, and skip without it. MCP tests run FastMCP servers over the SDK's in-memory transport (`create_connected_server_and_client_session`) with `anyio.run`, are marked `mcp`, and skip without it. Pydantic AI tests drive real `Agent` runs with a scripted `FunctionModel` (no network), are marked `pydantic_ai`, and skip without it. Postgres tests run `PostgresStore` against a real server (`DOD_TEST_POSTGRES_URI`, with pgvector), each in its own namespace, are marked `postgres`, and skip without the URI.

## Workflow
1. Every change starts as an issue using the agent-task template (Goal / Acceptance criteria / Files involved), sized for one PR. New issues go on the board as Backlog with Priority and Size.
2. A human approves by adding the `agent-ready` label. The board moves the issue to Ready, then to In progress when Claude starts.
3. Claude works on a `claude/issue-<N>-*` branch and pushes. A workflow step opens the PR with `Closes #N` (using `PAT_TOKEN`, so CI and review trigger).
4. CI (pytest + benchmarks) runs on the PR. Claude Code Review runs only after CI passes, and not on docs-only PRs (all files Markdown or under `docs/`). The review ends with `BLOCKING: <count>`; above 0, it posts `@claude fix the blocking items in the latest review` automatically, up to 3 rounds per PR, then adds `needs-human`. If every blocking finding is in `.github/workflows/` (per its `BLOCKING_FILE:` lines), it adds `needs-human` instead of requesting a fix. The logic lives in `scripts/review-auto-fix.sh` and `scripts/update-branches.sh`, tested in `tests/test_ci_scripts.py` with a fake `gh`. Other fixes go through `@claude` comments on the PR. After every push to main, open PRs that are behind get main merged in (CI and review rerun; if GitHub's merge conflicts but a local merge with `.gitattributes` drivers is clean, the workflow merges locally and pushes); a merge conflict gets a comment listing the files plus `needs-human`, which is removed automatically on a later push to main once the conflict is gone.
5. A human merges. Merging closes the issue.

If blocked or unsure (ambiguous acceptance criteria, a design decision the issue doesn't settle, a conflict with this file, a missing dependency or permission):
1. Comment the specific question on the issue: `gh issue comment <N> --body "..."`. Say what you need decided, the options you see, and what you'd need to proceed.
2. Add the label: `gh issue edit <N> --add-label needs-human`.
3. Stop. No more commits and no PR; the PR-opening step skips issues labeled `needs-human`.

The human answers with an `@claude` comment on the issue. That run removes `needs-human` before Claude starts, so continue from the answer; the PR-opening step then works again.

Handoffs (Instinct → planner → builders; see `docs/ORCHESTRATION.md`):
- Instinct hands off through a feature issue labeled `needs-breakdown` stating goal, why, priority, and definition of done.
- The planner reads it once and creates the technical sub-issues.
- Questions go on the issue, with at most one clarifying round. Direction or design questions go to the human via `needs-human`.
- No open-ended agent discussions in Slack.

**Never guess on design decisions.** Data model, APIs, capture semantics, dependencies and anything recorded in `docs/decisions.md` get asked about, not assumed.

Rules for every PR:
- New or changed behavior comes with tests. Anything on the capture path includes or updates a benchmark asserting p99 < 1 ms.
- One issue, one PR. Stay within "Files involved"; if more is needed, say so in the PR instead of expanding scope.
- Don't edit `.github/workflows/` from agent runs (the GitHub App can't push workflow changes).
- Keep strategy (pricing, positioning, competitors) out of code, comments, and commit messages.
- Record decisions in `docs/decisions.md`, one line of why each.

Keeping CLAUDE.md and decisions.md current:
- Update CLAUDE.md in the same PR when you change technical content it describes: stack, commands, workflow, conventions.
- Changes to the business model, positioning, hard rules, or existing `docs/decisions.md` entries must be flagged at the top of the PR description. Never make them just to satisfy a review comment; ask via `needs-human` instead.
- Any PR touching CLAUDE.md or `docs/decisions.md` lists those changes first in its description.

Hard rules (never break, in any PR):
- The ledger never stores memory content. IDs, content hashes, edges, and metadata only; content stays in the customer's stores.
- The library makes no telemetry or network calls home.
