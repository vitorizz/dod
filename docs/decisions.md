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
- **Self-hosted system, not hosted SaaS.** The paid system runs in the customer's cloud; we never see their data. Supersedes "Open-core... paid hosted console." (2026-09-27, founder decision)
- **Few customers, high ticket.** Annual license sized by deployment, starting with 3–5 design partners. Supersedes "Insurance-style pricing." (2026-09-27, founder decision)
- **Authority layer learns per deployment.** We never see cross-customer data; opt-in anonymized sharing is a later option. The moat shifts to cross-store connectors and reconciliation. (2026-09-27, founder decision)
- **Open-source library builds trust.** Security teams can audit exactly what capture records. (2026-09-27, founder decision)
- **Positioning: lineage for long-lived agent context.** Context managers decide what goes into context; we record what came out because of it, and clean only what's contaminated. Complementary to Mem0, Supermemory, Zep, and markdown-based memory, not competing. Scope covers any agent write (memory, DB, CRM, tickets). (2026-10-07, founder decision)
- **Target: teams building long-lived agents with memory for their users.** Personal agents like Instinct, AI companions, support agents, "agent employees". Instinct-sized startups are design-partner targets; "real forgetting" (deleting a memory plus everything derived from it) is the opening pitch. (2026-10-07, founder decision)
- **Pricing shape: paid pilots, then base fee plus a scaling metric.** Paid pilots first; then an annual license with a base fee plus a metric that scales with the customer (writes tracked or active end users), not seats; a compliance pack (proof of real deletion, audit exports) as an add-on. Refines "Few customers, high ticket." (2026-10-07, founder decision)
- **Edges are tagged `in_prompt` or `retrieved_in_scope`.** Distinguishes items the model actually saw from items retrieved in scope but not necessarily in the prompt. (2026-10-07, founder decision)
- **Over-include rather than miss a parent.** A missed parent breaks trace and excision silently; an extra edge only costs precision, which the tag recovers. (2026-10-07, founder decision)
- **Context scope is the session's context window.** Parents are whatever was in that window at write time, not just the latest retrieval. (2026-10-07, founder decision)
- **Compaction counts as a write.** Compaction rewrites memory; if it isn't captured, lineage breaks at the summary. (2026-10-07, founder decision)
- **Excision offers delete, quarantine (default), and regenerate.** Quarantine is reversible, so it's the safe default when over-inclusion flags clean items. (2026-10-07, founder decision)
- **Duplicate parent with different tags keeps `in_prompt`.** The stronger claim wins; dropping it would understate what the model saw. (2026-10-07, founder decision)
- **Compaction parents are tagged `in_prompt` and include the key's current context.** Compaction reads everything it replaces, and over-inclusion covers whatever else was in the window. (2026-10-07, founder decision)
- **Retrievals default to `retrieved_in_scope`; `in_prompt` only when confirmed.** Claiming an item was in the prompt without evidence would make the tag meaningless for filtering. (2026-10-07, founder decision)
- **Compaction has its own capture call, `on_compaction(key, replaced, ...)`.** A plain write can't carry the set it replaced, and those are its parents. (2026-10-07, founder decision)
- **Ledger checkpoints run off the append path.** SQLite's auto-checkpoint fsyncs inside the commit that crosses the threshold, pushing append p99 to ~4 ms; a background thread on a second connection keeps it ~0.2 ms. (2026-10-07, #11)
