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
  /** Scope key when a call runs outside agentTaskScope (default: one per call). */
  key?: (tool: string, options: any) => readonly [string, string];
}

function canonical(value: unknown): string {
  return JSON.stringify(value, (_k, v) =>
    v && typeof v === "object" && !Array.isArray(v)
      ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, v[k]])) : v) ?? "null";
}

let calls = 0;

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
            capture.onWrite({ store, content: canonical({ tool: name, result }), source: "tool", recordId: item });
            capture.onRetrieve([[store, item]], "in_prompt");
            return result;
          }
          const wid = capture.onWrite({ store, content: canonical({ tool: name, arguments: args }), source: "agent_inference" });
          const result = await execute(args, options);
          const rid = opts.recordId?.(name, args, result) ?? item;
          capture.onBind(wid, store, String(rid));
          const gone = [...(opts.deleted?.(name, args, result) ?? [])];
          if (gone.length) capture.onDelete(store, gone.map(String), "forget");
          capture.onRetrieve([[store, String(rid)]], "in_prompt");
          return result;
        };
        if (key) return run();
        const [a, t] = opts.key?.(name, options) ?? [name, `call-${++calls}`];
        return agentTaskScope(a, t, run);
      },
    };
  }
  return out as T;
}
