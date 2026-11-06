// Capture for tool-calling agents (Vercel AI SDK `tool()` objects, or any
// `{ execute(args, { toolCallId }) }` map), like dod's Python tool wrappers.
import { Capture, agentTaskScope, currentKey } from "./capture.js";

type Execute = (args: any, options?: any) => any;
type ToolLike = { execute?: Execute; [k: string]: unknown };

export interface ToolOptions {
  store?: string;
  /** Tools that only read: their results become context. */
  readOnly?: Iterable<string>;
  /** Tools that are neither: only scoped. */
  passthrough?: Iterable<string>;
  /** The record a write tool created or changed; defaults to the call item. */
  recordId?: (tool: string, args: any, result: any) => string | null | undefined;
  /** Records a write tool deleted (tombstones). */
  deleted?: (tool: string, args: any, result: any) => Iterable<string> | null | undefined;
  /**
   * Scope key when a call runs outside agentTaskScope (default: one per
   * call, whose context is reset when the call ends). A key from here can
   * span calls, so its context is kept: call `capture.reset(key)` when done.
   */
  key?: (tool: string, options: any) => readonly [string, string];
}

function canonical(value: unknown): string {
  return JSON.stringify(value, (_k, v) =>
    v && typeof v === "object" && !Array.isArray(v)
      ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, v[k]])) : v) ?? "null";
}

let calls = 0;

/** Run a capture step; a ledger error is logged, never thrown into the agent. */
function safely<T>(what: string, fn: () => T): T | undefined {
  try {
    return fn();
  } catch (e) {
    console.warn(`dod: ${what} failed; lineage for this call is incomplete:`, e);
    return undefined;
  }
}

/**
 * Wrap each tool's `execute` (a new object; tools are not mutated). A read
 * tool's result is recorded as item `call:<toolCallId>` (source `tool`) and
 * joins the scope's context as `in_prompt` (it goes back to the model). A
 * write tool is recorded before it runs (source `agent_inference`, content
 * its name and arguments) and bound after it succeeds.
 */
export function instrumentTools<T extends Record<string, ToolLike>>(tools: T, capture: Capture, opts: ToolOptions = {}): T {
  const store = opts.store ?? "tools";
  const readOnly = new Set(opts.readOnly ?? []), passthrough = new Set(opts.passthrough ?? []);
  const out: Record<string, ToolLike> = {};
  for (const [name, tool] of Object.entries(tools)) {
    const execute = tool.execute;
    if (typeof execute !== "function") { out[name] = tool; continue; }
    out[name] = {
      ...tool,
      execute: (args: any, options: any = {}) => {
        const key = currentKey();
        const run = async () => {
          const callId = String(options?.toolCallId ?? `local-${++calls}`);
          const item = `call:${callId}`;
          if (passthrough.has(name)) return execute(args, options);
          if (readOnly.has(name)) {
            const result = await execute(args, options);
            safely(`recording read tool ${name}`, () => {
              capture.onWrite({ store, content: canonical({ tool: name, result }), source: "tool", recordId: item });
              capture.onRetrieve([[store, item]], "in_prompt");
            });
            return result;
          }
          // Recorded before the call; if the tool throws, the write stays
          // unbound (it changed nothing the ledger knows of).
          const wid = safely(`recording write tool ${name}`, () =>
            capture.onWrite({ store, content: canonical({ tool: name, arguments: args }), source: "agent_inference" }));
          const result = await execute(args, options);
          safely(`binding write tool ${name}`, () => {
            const rid = String(opts.recordId?.(name, args, result) ?? item);
            if (wid) capture.onBind(wid, store, rid);
            const gone = [...(opts.deleted?.(name, args, result) ?? [])];
            if (gone.length) capture.onDelete(store, gone.map(String), "forget");
            capture.onRetrieve([[store, rid]], "in_prompt");
          });
          return result;
        };
        if (key) return run();
        if (opts.key) {
          const [a, t] = opts.key(name, options);
          return agentTaskScope(a, t, run);
        }
        // A per-call scope ends with the call: forget its context.
        const own = [name, `call-${++calls}`] as const;
        return agentTaskScope(own[0], own[1], async () => {
          try { return await run(); } finally { capture.reset(own); }
        });
      },
    };
  }
  return out as T;
}
