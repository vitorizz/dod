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
