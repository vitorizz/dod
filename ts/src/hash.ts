// Hashes and IDs, byte-compatible with dod.ledger / dod.capture (Python).
import { createHash, randomFillSync } from "node:crypto";

export type Tag = "in_prompt" | "retrieved_in_scope";
export type Parent = readonly [store: string, recordId: string, tag: Tag];
export const TAGS: readonly Tag[] = ["in_prompt", "retrieved_in_scope"];

/** sha256 of the UTF-8 (or raw) bytes, hex: the ledger's content hash. */
export function contentHash(content: string | Uint8Array): string {
  return createHash("sha256").update(content).digest("hex");
}

/** Compare strings by code point, as Python sorts str (JS compares UTF-16 units). */
export function cmpCodePoints(a: string, b: string): number {
  if (a === b) return 0;
  const ia = a[Symbol.iterator](), ib = b[Symbol.iterator]();
  for (;;) {
    const x = ia.next(), y = ib.next();
    if (x.done) return y.done ? 0 : -1;
    if (y.done) return 1;
    const cx = x.value.codePointAt(0)!, cy = y.value.codePointAt(0)!;
    if (cx !== cy) return cx < cy ? -1 : 1;
  }
}

/** JSON like Python's json.dumps(separators=(",", ":")) with ensure_ascii. */
export function pyJson(value: unknown): string {
  return JSON.stringify(value).replace(/[\u007f-￿]/g, (c) =>
    c === "\u007f" ? c : "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"));
}

export interface ContextSet {
  hash: string | null;
  /** store \0 recordId -> tag */
  members: Map<string, Tag>;
  keys: Array<[string, string]>;
}

const keyOf = (s: string, r: string) => s + "\u0000" + r;

/** Deduped context members and their content address (dod.ledger._context_set). */
export function contextSet(context: Iterable<Parent>): ContextSet {
  const members = new Map<string, Tag>();
  const pairs = new Map<string, [string, string]>();
  for (const p of context) {
    if (!Array.isArray(p) || p.length !== 3) throw new Error(`parent must be [store, recordId, tag]`);
    const [s0, r0, t] = p;
    if (!TAGS.includes(t)) throw new Error(`unknown parent tag ${JSON.stringify(t)}`);
    const s = String(s0), r = String(r0), k = keyOf(s, r);
    if (!members.has(k)) { members.set(k, t); pairs.set(k, [s, r]); }
    else if (t === "in_prompt") members.set(k, t);
  }
  if (!members.size) return { hash: null, members, keys: [] };
  const keys = [...pairs.values()].sort((a, b) => cmpCodePoints(a[0], b[0]) || cmpCodePoints(a[1], b[1]));
  const tags = keys.map(([s, r]) => members.get(keyOf(s, r))!);
  let blob = keys.flat().join("\u0000");
  if ((blob.match(/\u0000/g) || []).length === 2 * keys.length - 1) {
    blob = "\u0000" + blob + "\u0000" + tags.join("\u0000");
  } else {
    blob = pyJson([keys, tags]);
  }
  return { hash: createHash("sha256").update(blob, "utf8").digest("hex"), members, keys };
}

let lastId = "";
const pool = Buffer.alloc(16 * 256);
let poolAt = pool.length;

/** A UUIDv7-layout write ID (32 hex chars), strictly increasing in-process. */
export function writeId(tsNs: bigint): string {
  if (poolAt === pool.length) { randomFillSync(pool); poolAt = 0; }
  const b = pool.subarray(poolAt, poolAt + 16);
  poolAt += 16;
  let ms = Number((tsNs < 0n ? 0n : tsNs) / 1_000_000n);
  for (let i = 5; i >= 0; i--) { b[i] = ms % 256; ms = Math.floor(ms / 256); }
  b[6] = 0x70 | (b[6] & 0x0f);
  b[8] = 0x80 | (b[8] & 0x3f);
  let id = b.toString("hex");
  if (id <= lastId) id = (BigInt("0x" + lastId) + 1n).toString(16).padStart(32, "0");
  lastId = id;
  return id;
}

const startNs = BigInt(Date.now()) * 1_000_000n;
const startHr = process.hrtime.bigint();

/** Wall-clock nanoseconds (monotonic within the process). */
export function nowNs(): bigint {
  return startNs + (process.hrtime.bigint() - startHr);
}
