// Scope and capture: what an agent retrieved, linked to what it writes.
import { AsyncLocalStorage } from "node:async_hooks";
import type { Parent, Tag } from "./hash.js";
import { Ledger, type Item, type Reason, type Source } from "./ledger.js";

export type Key = readonly [agentId: string, taskId: string];
const scopeStore = new AsyncLocalStorage<Key>();

/**
 * Run `fn` as one unit of work. Everything retrieved inside it (across
 * stores and awaits) is context for every write inside it; concurrent
 * scopes never share context. Like `dod.scope.agent_task_scope`.
 */
export function agentTaskScope<T>(agentId: string, taskId: string, fn: () => T): T {
  return scopeStore.run([agentId, taskId], fn);
}

/** The current scope's key, or undefined outside one. */
export function currentKey(): Key | undefined {
  return scopeStore.getStore();
}

const keyString = (k: Key) => JSON.stringify(k);

/** Records retrievals and writes into a ledger, keyed by scope. */
export class Capture {
  private contexts = new Map<string, Parent[]>();
  private qSeq = 0n;
  private qSet = new Set<string>();

  constructor(readonly ledger: Ledger) {}

  /** Add retrieved items to the scope's context (no-op outside a scope). */
  onRetrieve(items: Iterable<Item>, tag: Tag = "retrieved_in_scope", key: Key | undefined = currentKey()): void {
    if (!key) return;
    const k = keyString(key);
    let ctx = this.contexts.get(k);
    if (!ctx) this.contexts.set(k, (ctx = []));
    for (const [s, r] of items) ctx.push([s, r, tag]);
  }

  /**
   * Record a write with the scope's context as its parents; bind it to
   * `recordId` if given. Returns the write ID.
   */
  onWrite(w: {
    store: string; content: string | Uint8Array; source: Source; recordId?: string | null;
    replaces?: Iterable<Item>;
  }, key: Key | undefined = currentKey()): string {
    const ctx = key ? this.contexts.get(keyString(key)) ?? [] : [];
    return this.ledger.appendWrite({
      content: w.content, store: w.store, source: w.source, recordId: w.recordId,
      agentId: key?.[0] ?? null, sessionId: key?.[1] ?? null,
      parents: [...(w.replaces ?? [])].map(([s, r]) => [s, r, "in_prompt"] as Parent),
      context: [...ctx],
    });
  }

  /** Bind a write recorded before the store call to the ID the store returned. */
  onBind(writeId: string, store: string, recordId: string): void {
    this.ledger.bind(writeId, store, recordId);
  }

  /** Record deletions (tombstones). */
  onDelete(store: string, recordIds: Iterable<string>, reason: Reason = "forget", key: Key | undefined = currentKey()): void {
    this.ledger.appendTombstones([...recordIds].map((r) => [store, r] as Item), reason,
      { agentId: key?.[0], sessionId: key?.[1] });
  }

  /** Forget a scope's context (call when a long-lived unit of work ends). */
  reset(key: Key): void {
    this.contexts.delete(keyString(key));
  }

  /** Whether an item is quarantined (cached; catches up from the ledger). */
  isQuarantined(store: string, recordId: string): boolean {
    for (const r of this.ledger.quarantineLog(this.qSeq)) {
      const k = r.store + "\u0000" + r.recordId;
      if (r.action === "quarantine") this.qSet.add(k); else this.qSet.delete(k);
      this.qSeq = r.seq;
    }
    return this.qSet.has(store + "\u0000" + recordId);
  }
}
