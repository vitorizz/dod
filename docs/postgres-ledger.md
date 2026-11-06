# Postgres ledger: design options

Status: option A built (`dod.ship`, `dod ship`); see decisions.md (2026-11-06).

## Why

Teams running several agent processes on several hosts can't share one SQLite file. They want one ledger per deployment, in the database they already operate, with the same guarantees: append-only, content-free, sub-millisecond capture.

## What makes it hard

The SQLite ledger is more than storage. These parts depend on SQLite specifically:

- **Capture latency.** An append is an in-process, page-cache write (~0.1–0.3 ms p99). A Postgres round trip on the same host is ~0.2–0.5 ms, and across a network 1+ ms. A synchronous Postgres append would break the p99 < 1 ms bar for most deployments.
- **Walks.** `trace` and `blast_radius` run as many small queries over one snapshot. Over a network, each query costs a round trip. They need to become recursive CTEs or server-side functions.
- **The lineage index** is a companion SQLite file filled by a background thread. In Postgres it would be a table, or a materialized view, maintained by the same writer.
- **Append-only enforcement** uses triggers. Postgres has the equivalent (`BEFORE UPDATE OR DELETE ... RAISE`) plus `REVOKE UPDATE, DELETE`, which is stronger.
- **Fenwick context sets** are plain tables and port as they are.

## Options

**A. Local SQLite capture, shipped to Postgres (recommended).**
Each process keeps capturing into a local SQLite ledger: same latency, same code. A shipper thread copies committed rows (by `seq`, which is already monotonic per table) into Postgres tables with the same schema plus a `node_id` column. Queries (`trace`, `blast_radius`, `report`) gain a Postgres implementation that reads the merged tables. Excision writes quarantine rows to Postgres; each node pulls them, the way the quarantine cache already polls.
- Pros: capture stays sub-millisecond, and a Postgres outage never blocks agents. Capture code doesn't change.
- Cons: cross-node lineage is visible after a short lag (one shipping interval). Two query implementations to maintain.

**B. Postgres as the only ledger, async appends.**
Capture enqueues rows, and a background writer batches them into Postgres.
- Pros: one source of truth.
- Cons: a crash loses the queue unless it is spilled to disk, which is option A with extra steps. Ordering across processes needs a sequence.

**C. Postgres as the only ledger, synchronous appends.**
- Pros: simplest model.
- Cons: misses the latency bar except on the same host. Only worth offering as an opt-in for low-volume agents.

## Recommendation

A. It keeps the capture path untouched, which is where the guarantee lives, and it moves the cross-host problem to the query side. Build order:

1. Shipper with `node_id`, idempotent upserts keyed on `(node_id, table, seq)`.
2. Postgres `trace` and `blast_radius` as recursive CTEs, tested against the SQLite implementation on the same ledgers (property test: same results).
3. Quarantine fan-out, Postgres to nodes.
4. Audit chain: each node's chain ships as is; `verify` runs per node.

## Questions to decide

- Is a shipping lag of 1–5 s acceptable for cross-node `blast_radius`?
- Is a per-node audit chain enough, or must there be one global chain?
- Which Postgres versions and hosting (RDS, Cloud SQL, self-hosted) to support first?
