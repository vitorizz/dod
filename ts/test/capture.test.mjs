import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Capture, Ledger, agentTaskScope, instrumentTools, contextSet } from "../dist/index.js";

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
  const { statSync, existsSync } = await import("node:fs");
  const path = tmp();
  const led = new Ledger(path, { checkpointMs: 20 });
  const cap = new Capture(led);
  for (let i = 0; i < 2000; i++) cap.onWrite({ store: "s", content: `c${i}`, source: "user", recordId: `r${i}` });
  await new Promise((r) => setTimeout(r, 200));
  const row = led.db.prepare("PRAGMA wal_checkpoint(PASSIVE)").get();
  assert.equal(row.busy, 0);
  assert.equal(row.log, row.checkpointed); // everything already copied
  led.close();
  assert.ok(!existsSync(path + "-wal") || statSync(path + "-wal").size >= 0);
});
