import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Capture, Ledger, agentTaskScope, instrumentTools, contextSet, setWalRestartBytes } from "../dist/index.js";

const tmp = () => join(mkdtempSync(join(tmpdir(), "dod-ts-")), "ledger.db");

test("retrieval in a scope becomes the write's context", () => {
  const led = new Ledger(tmp());
  const cap = new Capture(led);
  agentTaskScope("agent", "t1", () => {
    cap.onRetrieve([["kb", "policy"]], "in_prompt");
    cap.onWrite({ store: "notes", content: "90 days", source: "agent_inference", recordId: "n1" });
  });
  const rows = led.db.prepare(
    "SELECT m.store, m.record_id, m.tag FROM writes w JOIN context_chain c ON c.set_hash = w.context_set"
    + " JOIN context_members m ON m.set_hash = c.part").all();
  assert.deepEqual(rows.map((r) => [r.store, r.record_id, r.tag]), [["kb", "policy", "in_prompt"]]);
  led.close();
});

test("concurrent scopes don't share context", async () => {
  const led = new Ledger(tmp());
  const cap = new Capture(led);
  const tick = () => new Promise((r) => setTimeout(r, 5));
  await Promise.all(["a", "b"].map((who) => agentTaskScope(who, "t", async () => {
    cap.onRetrieve([["kb", `doc-${who}`]]);
    await tick();
    cap.onWrite({ store: "notes", content: who, source: "agent_inference", recordId: `note-${who}` });
  })));
  const rows = led.db.prepare(
    "SELECT b.record_id note, m.record_id parent FROM writes w JOIN bindings b ON b.write_id = w.write_id"
    + " JOIN context_members m ON m.set_hash = w.context_set ORDER BY note").all();
  assert.deepEqual(rows.map((r) => [r.note, r.parent]), [["note-a", "doc-a"], ["note-b", "doc-b"]]);
  led.close();
});

test("same context, same set; reordered context, same address", () => {
  const a = contextSet([["s", "x", "in_prompt"], ["s", "y", "retrieved_in_scope"]]);
  const b = contextSet([["s", "y", "retrieved_in_scope"], ["s", "x", "in_prompt"], ["s", "x", "retrieved_in_scope"]]);
  assert.equal(a.hash, b.hash);
  assert.equal(contextSet([]).hash, null);
});

test("tools: a read's result is a parent of a later write", async () => {
  const led = new Ledger(tmp());
  const cap = new Capture(led);
  const db = { refunds: "90 days" };
  const tools = instrumentTools({
    lookup: { description: "look up", execute: async ({ topic }) => db[topic] },
    save: { description: "save", execute: async ({ note }) => ({ id: "note-1", note }) },
  }, cap, { store: "crm", readOnly: ["lookup"], recordId: (_n, _a, r) => r?.id });
  await agentTaskScope("agent", "run-1", async () => {
    assert.equal(await tools.lookup.execute({ topic: "refunds" }, { toolCallId: "c1" }), "90 days");
    await tools.save.execute({ note: "told Ben 90 days" }, { toolCallId: "c2" });
  });
  const rows = led.db.prepare("SELECT b.record_id, m.record_id parent, m.tag FROM bindings b"
    + " JOIN writes w ON w.write_id = b.write_id LEFT JOIN context_members m ON m.set_hash = w.context_set"
    + " ORDER BY b.rowid").all();
  assert.deepEqual(rows.map((r) => [r.record_id, r.parent, r.tag]),
    [["call:c1", null, null], ["note-1", "call:c1", "in_prompt"]]);
  led.close();
});

test("tools: a call outside a scope leaves no context behind", async () => {
  const led = new Ledger(tmp());
  const cap = new Capture(led);
  const tools = instrumentTools({
    lookup: { execute: async () => "x" },
    save: { execute: async () => ({ id: "n" }) },
    fail: { execute: async () => { throw new Error("boom"); } },
  }, cap, { readOnly: ["lookup"] });
  await tools.lookup.execute({}, { toolCallId: "c1" });
  await tools.save.execute({}, { toolCallId: "c2" });
  await assert.rejects(tools.fail.execute({}, { toolCallId: "c3" }), /boom/);
  assert.equal(cap["contexts"].size, 0);
  // A caller-supplied key may span calls, so its context is kept.
  const keyed = instrumentTools({ lookup: { execute: async () => "x" } }, cap,
    { readOnly: ["lookup"], key: () => ["agent", "run-1"] });
  await keyed.lookup.execute({}, { toolCallId: "c4" });
  assert.equal(cap["contexts"].size, 1);
  cap.reset(["agent", "run-1"]);
  assert.equal(cap["contexts"].size, 0);
  led.close();
});

test("quarantine is visible through the capture cache", () => {
  const led = new Ledger(tmp());
  const cap = new Capture(led);
  assert.equal(cap.isQuarantined("kb", "x"), false);
  led.quarantine([["kb", "x"]], "false fact", "alice");
  assert.equal(cap.isQuarantined("kb", "x"), true);
  led.release([["kb", "x"]]);
  assert.equal(cap.isQuarantined("kb", "x"), false);
  led.close();
});

test("rejects bad input and other schema versions", () => {
  const path = tmp();
  const led = new Ledger(path);
  assert.throws(() => led.appendWrite({ content: "x", store: "s", source: "nope" }));
  assert.throws(() => led.appendWrite({ content: "x", store: "s", source: "user", context: [["s", "r", "bad"]] }));
  led.db.exec("PRAGMA user_version=4");
  led.close();
  assert.throws(() => new Ledger(path), /schema version 4/);
});

test("the WAL is checkpointed off the write path", async () => {
  const { statSync } = await import("node:fs");
  const path = tmp();
  const led = new Ledger(path, { checkpointMs: 20 });
  const cap = new Capture(led);
  const before = statSync(path).size;
  for (let i = 0; i < 2000; i++) cap.onWrite({ store: "s", content: `c${i}`, source: "user", recordId: `r${i}` });
  // Writes only go to the WAL (autocheckpoint is off); the main file grows
  // only when the worker checkpoints. Wait up to 10 s for a slow runner.
  const deadline = Date.now() + 10_000;
  while (statSync(path).size === before && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  assert.ok(statSync(path).size > before, "the worker never checkpointed");
  led.close();
});

test("tools: deletes, passthrough, a throwing write, and a failing ledger", async () => {
  const led = new Ledger(tmp());
  const cap = new Capture(led);
  const seen = [];
  const tools = instrumentTools({
    forget: { execute: async ({ id }) => ({ ok: true, id }) },
    peek: { execute: async () => "x" },
    boom: { execute: async () => { throw new Error("down"); } },
  }, cap, { store: "crm", passthrough: ["peek"], deleted: (n, a) => (n === "forget" ? [a.id] : []) });
  await agentTaskScope("agent", "t", async () => {
    await tools.forget.execute({ id: "doc-9" }, { toolCallId: "c1" });
    assert.equal(await tools.peek.execute({}, { toolCallId: "c2" }), "x");
    await assert.rejects(tools.boom.execute({}, { toolCallId: "c3" }), /down/);
  });
  const t = led.db.prepare("SELECT record_id, reason FROM tombstones").all();
  assert.deepEqual(t.map((r) => [r.record_id, r.reason]), [["doc-9", "forget"]]);
  const writes = Number(led.db.prepare("SELECT count(*) n FROM writes").get().n);
  const bound = Number(led.db.prepare("SELECT count(*) n FROM bindings").get().n);
  assert.equal(writes, 2);  // forget and boom; peek is passthrough
  assert.equal(bound, 1);   // boom threw: its write stays unbound
  // Capture fails open: a closed ledger doesn't break the tool call.
  led.close();
  const warn = console.warn;
  console.warn = (...a) => seen.push(a.join(" "));
  try {
    assert.deepEqual(await tools.forget.execute({ id: "z" }, { toolCallId: "c4" }), { ok: true, id: "z" });
  } finally {
    console.warn = warn;
  }
  assert.ok(seen.some((m) => m.includes("lineage for this call is incomplete")));
});

test("appendBatch merges duplicate parents, in_prompt wins", () => {
  const led = new Ledger(tmp());
  led.appendWrite({ content: "x", store: "s", source: "user",
    parents: [["s", "a", "retrieved_in_scope"], ["s", "a", "in_prompt"], ["s", "b", "retrieved_in_scope"]] });
  const rows = led.db.prepare("SELECT parent_record_id r, tag FROM direct_parents ORDER BY r").all();
  assert.deepEqual(rows.map((x) => [x.r, x.tag]), [["a", "in_prompt"], ["b", "retrieved_in_scope"]]);
  led.close();
});


test("the WAL stays bounded under continuous writes", async () => {
  const { statSync } = await import("node:fs");
  setWalRestartBytes(1 << 20);
  try {
    const path = tmp();
    const led = new Ledger(path, { checkpointMs: 0 });  // the writer alone keeps it bounded
    let peak = 0;
    for (let i = 0; i < 200; i++) {
      led.appendTombstones(Array.from({ length: 200 }, (_, j) => ["s", `r${i}-${j}`]), "forget");
      peak = Math.max(peak, statSync(path + "-wal").size);
    }
    assert.ok(peak < 4 << 20, `WAL grew to ${peak} bytes`);
    assert.equal(Number(led.db.prepare("SELECT count(*) n FROM tombstones").get().n), 40000);
    led.close();
  } finally {
    setWalRestartBytes(64 * 1024 * 1024);
  }
});
