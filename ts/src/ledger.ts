// The ledger file, in the format dod (Python) reads and writes.
import { DatabaseSync, type StatementSync } from "node:sqlite";
import { Worker } from "node:worker_threads";
import { statSync } from "node:fs";
import { contentHash, contextSet, nowNs, TAGS, writeId, type Parent, type Tag } from "./hash.js";
import { SCHEMA, SCHEMA_VERSION } from "./schema.js";

export type Source = "user" | "tool" | "document" | "agent_inference";
export type Reason = "consolidation" | "excision" | "forget" | "other";
export type Item = readonly [store: string, recordId: string];
const SOURCES: readonly Source[] = ["user", "tool", "document", "agent_inference"];
const REASONS: readonly Reason[] = ["consolidation", "excision", "forget", "other"];

export interface WriteInput {
  content: string | Uint8Array;
  store: string;
  source: Source;
  agentId?: string | null;
  sessionId?: string | null;
  /** The write's own parents (e.g. the record it rewrites). */
  parents?: Iterable<Parent>;
  /** The scope's context: what was retrieved before this write. */
  context?: Iterable<Parent>;
  /** Bind the write to this record in the same transaction. */
  recordId?: string | null;
}

export class LedgerError extends Error {}

/** The writer restarts the WAL past this size (as dod.ledger does). */
export let WAL_RESTART_BYTES = 64 * 1024 * 1024;
/** For tests. */
export function setWalRestartBytes(n: number): void {
  WAL_RESTART_BYTES = n;
}

function walBytes(path: string): number {
  try {
    return statSync(path).size;
  } catch {
    return 0;
  }
}

// Checkpoints the WAL from its own thread and connection (PASSIVE: never
// waits for writers), so a commit never runs one. Like dod.ledger's
// checkpoint thread.
const CHECKPOINTER = `
const { workerData, parentPort } = require("node:worker_threads");
const { DatabaseSync } = require("node:sqlite");
const db = new DatabaseSync(workerData.path);
db.exec("PRAGMA busy_timeout=5000");
const timer = setInterval(() => {
  try { db.exec("PRAGMA wal_checkpoint(PASSIVE)"); } catch (e) {}
}, workerData.ms);
parentPort.on("message", () => { clearInterval(timer); try { db.exec("PRAGMA wal_checkpoint(PASSIVE)"); } catch (e) {} db.close(); process.exit(0); });
`;

/**
 * Append-only, content-free lineage ledger (SQLite, WAL), the same file
 * format as `dod.ledger.Ledger`: Python's `trace`, `blast_radius`, `excise`
 * and the `dod` CLI read what this writes. Context sets are stored as full
 * sets (Python also stores deltas; both read either). Audited ledgers
 * (`audit=True` in Python) can't be written from here.
 */
export class Ledger {
  readonly db: DatabaseSync;
  private stmts = new Map<string, StatementSync>();

  private checkpointer?: Worker;

  /** `checkpointMs`: how often a worker thread checkpoints the WAL (0: never). */
  constructor(readonly path: string, opts: { checkpointMs?: number } = {}) {
    this.db = new DatabaseSync(path);
    this.db.exec("PRAGMA busy_timeout=5000");
    if (path !== ":memory:") this.db.exec("PRAGMA journal_mode=WAL");
    this.db.exec("PRAGMA synchronous=NORMAL");
    this.db.exec("PRAGMA wal_autocheckpoint=0");
    const version = Number((this.db.prepare("PRAGMA user_version").get() as any).user_version);
    if (version === 0) {
      const any = this.db.prepare("SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").get();
      if (any) throw new LedgerError(`ledger file ${path} is unversioned; this library uses version ${SCHEMA_VERSION}`);
      this.db.exec("BEGIN IMMEDIATE");
      try {
        this.db.exec(SCHEMA);
        this.db.exec(`PRAGMA user_version=${SCHEMA_VERSION}`);
        this.db.exec("COMMIT");
      } catch (e) {
        this.db.exec("ROLLBACK");
        throw e;
      }
    } else if (version !== SCHEMA_VERSION) {
      throw new LedgerError(`ledger file ${path} has schema version ${version}; this library uses version`
        + ` ${SCHEMA_VERSION}. Open it once with the Python library to migrate it.`);
    }
    if (this.db.prepare("SELECT 1 FROM audit_chain LIMIT 1").get()) {
      this.db.close();
      throw new LedgerError(`ledger file ${path} is audited (hash-chained); write to it from Python`);
    }
    const ms = opts.checkpointMs ?? 1000;
    if (path !== ":memory:" && ms > 0) {
      this.checkpointer = new Worker(CHECKPOINTER, { eval: true, workerData: { path, ms } });
      this.checkpointer.unref();
    }
  }

  private q(sql: string): StatementSync {
    let s = this.stmts.get(sql);
    if (!s) {
      s = this.db.prepare(sql);
      s.setReadBigInts(true);
      this.stmts.set(sql, s);
    }
    return s;
  }

  private tx<T>(fn: () => T): T {
    this.db.exec("BEGIN IMMEDIATE");
    let out: T;
    try {
      out = fn();
      this.db.exec("COMMIT");
    } catch (e) {
      this.db.exec("ROLLBACK");
      throw e;
    }
    if (this.path !== ":memory:" && walBytes(this.path + "-wal") > WAL_RESTART_BYTES) this.restartWal();
    return out;
  }

  /**
   * Start the WAL over once it is large. With autocheckpoint off, SQLite
   * restarts it only when a writer begins while the log is fully copied and
   * unread, which continuous writes rarely allow, so it would grow without
   * bound (as in dod.ledger, which does the same). No busy wait: if a reader
   * is in the way, the next commit tries again.
   */
  private restartWal(): void {
    this.db.exec("PRAGMA busy_timeout=0");
    try {
      this.db.exec("PRAGMA wal_checkpoint(TRUNCATE)");
    } catch {
      /* busy: retried at the next commit */
    } finally {
      this.db.exec("PRAGMA busy_timeout=5000");
    }
  }

  /** Record a write (and its parent edges) in one transaction; return its ID. */
  appendWrite(w: WriteInput): string {
    return this.appendBatch([w])[0];
  }

  /** Record several writes in one transaction; return their IDs in order. */
  appendBatch(writes: WriteInput[]): string[] {
    const prepared = writes.map((w) => {
      if (!SOURCES.includes(w.source)) throw new Error(`unknown source ${JSON.stringify(w.source)}`);
      const direct = new Map<string, Parent>();
      for (const p of w.parents ?? []) {
        if (!Array.isArray(p) || p.length !== 3 || !TAGS.includes(p[2])) throw new Error("bad parent");
        const k = p[0] + "\u0000" + p[1], prev = direct.get(k);
        if (!prev || p[2] === "in_prompt") direct.set(k, [String(p[0]), String(p[1]), p[2]]);
      }
      return { w, set: contextSet(w.context ?? []), direct: [...direct.values()], hash: contentHash(w.content) };
    });
    return this.tx(() => {
      const ts = nowNs();
      return prepared.map(({ w, set, direct, hash }) => {
        if (set.hash && !this.q("SELECT 1 FROM context_sets WHERE set_hash = ?").get(set.hash)) {
          this.q("INSERT INTO context_sets (set_hash, size, ts, base, step, upgrades) VALUES (?, ?, ?, NULL, 0, 0)")
            .run(set.hash, set.keys.length, ts);
          this.q("INSERT INTO context_chain (set_hash, part) VALUES (?, ?)").run(set.hash, set.hash);
          const ins = this.q("INSERT INTO context_members (set_hash, store, record_id, tag) VALUES (?, ?, ?, ?)");
          for (const [s, r] of set.keys) ins.run(set.hash, s, r, set.members.get(s + "\u0000" + r)!);
        }
        const id = writeId(ts);
        this.q("INSERT INTO writes (write_id, content_hash, ts, store, agent_id, session_id, source, context_set)"
          + " VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
          .run(id, hash, ts, w.store, w.agentId ?? null, w.sessionId ?? null, w.source, set.hash);
        const dp = this.q("INSERT INTO direct_parents (write_id, parent_store, parent_record_id, tag) VALUES (?, ?, ?, ?)");
        for (const [s, r, t] of direct) dp.run(id, s, r, t);
        if (w.recordId != null) this.bindIn(id, w.store, String(w.recordId), ts, null);
        return id;
      });
    });
  }

  private bindIn(writeIdValue: string, store: string, recordId: string, ts: bigint, hash: string | null) {
    this.q("INSERT INTO bindings (write_id, store, record_id, ts, content_hash) VALUES (?, ?, ?, ?, ?)")
      .run(writeIdValue, store, recordId, ts, hash);
  }

  /** Bind an earlier write to the record the store gave it. */
  bind(writeIdValue: string, store: string, recordId: string, contentHashValue: string | null = null): void {
    this.bindIn(writeIdValue, store, String(recordId), nowNs(), contentHashValue);
  }

  /** Record that items were deleted from their store. */
  appendTombstones(items: Iterable<Item>, reason: Reason, opts: { writeId?: string; agentId?: string; sessionId?: string } = {}): void {
    if (!REASONS.includes(reason)) throw new Error(`unknown tombstone reason ${JSON.stringify(reason)}`);
    const list = [...items];
    this.tx(() => {
      const ts = nowNs();
      const ins = this.q("INSERT INTO tombstones (store, record_id, ts, reason, write_id, agent_id, session_id)"
        + " VALUES (?, ?, ?, ?, ?, ?, ?)");
      for (const [s, r] of list) ins.run(s, r, ts, reason, opts.writeId ?? null, opts.agentId ?? null, opts.sessionId ?? null);
    });
  }

  quarantine(items: Iterable<Item>, reason: string | null = null, actor: string | null = null): void {
    this.appendQuarantine(items, "quarantine", reason, actor);
  }

  release(items: Iterable<Item>, reason: string | null = null, actor: string | null = null): void {
    this.appendQuarantine(items, "release", reason, actor);
  }

  private appendQuarantine(items: Iterable<Item>, action: string, reason: string | null, actor: string | null) {
    const list = [...items];
    this.tx(() => {
      const ts = nowNs();
      const ins = this.q("INSERT INTO quarantine (ts, store, record_id, action, reason, actor) VALUES (?, ?, ?, ?, ?, ?)");
      for (const [s, r] of list) ins.run(ts, s, r, action, reason, actor);
    });
  }

  /** Quarantine actions with seq > after, in seq order. */
  quarantineLog(after = 0n): Array<{ seq: bigint; store: string; recordId: string; action: string }> {
    return (this.q("SELECT seq, store, record_id, action FROM quarantine WHERE seq > ? ORDER BY seq").all(after) as any[])
      .map((r) => ({ seq: r.seq, store: r.store, recordId: r.record_id, action: r.action }));
  }

  /** Items whose latest quarantine action is `quarantine`. */
  quarantined(): Set<string> {
    const out = new Set<string>();
    for (const r of this.quarantineLog()) {
      const k = r.store + "\u0000" + r.recordId;
      if (r.action === "quarantine") out.add(k); else out.delete(k);
    }
    return out;
  }

  close(): void {
    if (this.checkpointer) {
      this.checkpointer.postMessage("stop");
      this.checkpointer = undefined;
    }
    try { this.db.exec("PRAGMA wal_checkpoint(PASSIVE)"); } catch { /* busy: the next open checkpoints */ }
    this.db.close();
  }
}
