# @dod-lineage/capture (TypeScript)

Lineage capture for Node agents. It records which retrieved items were in context at every write, into the **same ledger file** the Python `dod` library reads. So `dod trace`, `dod blast-radius`, `dod excise`, `dod ship` and the Python API work on what TypeScript agents write, and a mixed Python/TS deployment shares one ledger.

Node >= 22.13 (built-in `node:sqlite`). No runtime dependencies.

```ts
import { Ledger, Capture, agentTaskScope, instrumentTools } from "@dod-lineage/capture";

const cap = new Capture(new Ledger("agent.db"));

await agentTaskScope("support", `ticket-${id}`, async () => {
  const docs = await kb.search(q);
  cap.onRetrieve(docs.map((d) => ["kb", d.id]), "in_prompt");
  const note = await notes.save(text);
  cap.onWrite({ store: "notes", content: text, source: "agent_inference", recordId: note.id });
});
```

## Tool-calling agents (Vercel AI SDK and similar)

`instrumentTools` wraps a map of tools with `execute(args, { toolCallId })`, as the Vercel AI SDK's `tool()` objects have:
- A read tool's result becomes item `call:<toolCallId>`, in the prompt from then on.
- A write tool is recorded before it runs and bound after it succeeds, to `recordId(tool, args, result)` or the call item.

```ts
const tools = instrumentTools({ lookup, saveNote }, cap, { store: "crm", readOnly: ["lookup"] });
await generateText({ model, tools, prompt });
```

## Then, from Python or the CLI

```bash
dod trace notes note-17 --ledger agent.db
dod excise kb policy-3 --ledger agent.db      # TS agents check cap.isQuarantined(store, id)
```

## Compatibility

- **Schema:** the schema is generated from the Python library (`scripts/export_ts_schema.py`).
- **Hashes and IDs:** content addresses, content hashes and write IDs are computed the same way as in Python. `tests/test_ts_compat.py` checks this, including Unicode and the JSON fallback.
- **Context sets:** stored as full sets. Python reads both full sets and the delta chains it writes itself.
- **Audited ledgers:** a ledger hash-chained with `audit=True` must be written from Python. Opening one here throws.
- **WAL checkpoints:** run on a worker thread, never inside a write. Capture p99 is ~0.3 ms with 10 context items.
