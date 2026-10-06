# Decisions log

Append new decisions at the bottom. One line of why each.

- **Orchestration layer, not storage layer.** The database never sees the context window; only the harness can capture what was in context at write time.
- **Cross-store from early on.** No single DB vendor can join lineage across Mongo + Postgres + vector DB + framework state. This is the moat.
- **Correctness, not degradation.** Bigger context windows will fix fuzziness; they won't fix false beliefs.
- **Trace + excise, never detection.** Detection is unreliable; tracing from a recorded edge is cheap and reliable.
- **CrewAI first.** Already emits both memory query and memory save events; least glue to a working demo.
- **SQLite append-only store to start.** Simple, local, fast; swap later.
- **Library, not OTel exporter.** OTel loses the retrieval→write binding. Capture ourselves, export as OTel attributes in our own namespace.
- **Open-core.** OSS capture library; paid hosted console. Learned authority layer stays proprietary.
- **Insurance-style pricing.** Small monthly fee by volume/retention, investigations included. Not pay-per-incident.
- **CrewAI: wrap `Memory.recall`/`remember`, not its events.** Save events carry no record ID and handlers run in a thread pool (can reorder); wrapping is synchronous with exact IDs. Scope per agent/task via `contextvars` so concurrent agents never cross-link parents; pin crewai 1.15.22. (2026-09-26, #13)
- **Ask instead of guess: `needs-human` gate before opening a PR.** Agents that guess on unsettled design decisions produce work that has to be redone; a comment + label + stop is cheaper than a wrong PR. The `@claude` reply clears the label and resumes work. (2026-09-27, #17)
- **Content-free ledger.** The ledger stores IDs, content hashes, edges, and metadata only; memory content stays in the customer's stores. This is a requirement on the schema #11 will build, not yet verified against implemented code. (2026-09-27)
- **Instinct as PM layer above the Claude planner.** One owner for roadmap, priorities, and dispatch keeps agents on the highest-value work; the human keeps both gates (`agent-ready`, merge). (2026-10-06, #19)
- **No automated fix-round cap.** The reviewer doesn't emit `@claude` yet, so there's no automated loop to bound; add a cap only if one appears. (2026-10-06, #19)
- **Review after CI, skip docs-only PRs.** Reviewing a PR whose tests fail, or one that only changes Markdown, spends tokens without catching code bugs. (2026-10-06)
- **CLAUDE.md holds rules and commands only.** Every agent run loads it; background lives in `docs/PRODUCT.md` and strategy in private notes, so runs load less. (2026-10-06)
