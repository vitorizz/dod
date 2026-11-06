// Writes a fixed scenario for tests/test_ts_compat.py, and prints the
// content addresses of some contexts for comparison with Python.
import { Capture, Ledger, agentTaskScope, contextSet, contentHash } from "../dist/index.js";

const [, , path] = process.argv;
const led = new Ledger(path);
const cap = new Capture(led);
const ids = {};
agentTaskScope("agent", "plant", () => {
  ids.bad = cap.onWrite({ store: "kb", content: "refunds within 90 days", source: "user", recordId: "policy" });
});
agentTaskScope("agent", "t1", () => {
  cap.onRetrieve([["kb", "policy"]], "in_prompt");
  ids.d1 = cap.onWrite({ store: "crm", content: "told Ana 90 days", source: "agent_inference", recordId: "R1" });
});
agentTaskScope("agent", "t2", () => {
  cap.onRetrieve([["crm", "R1"]]);
  ids.d2 = cap.onWrite({ store: "notes", content: "Ana gets a refund", source: "agent_inference", recordId: "ana.md" });
  cap.onWrite({ store: "kb", content: "refunds within 90 days (v2)", source: "agent_inference", recordId: "policy",
    replaces: [["kb", "policy"]] });
});
agentTaskScope("agent", "t3", () => {
  ids.ok = cap.onWrite({ store: "crm", content: "shipping is free", source: "agent_inference", recordId: "R2" });
});
cap.onDelete("notes", ["ana.md"], "forget");
const vectors = [
  [["s", "x", "in_prompt"], ["s", "y", "retrieved_in_scope"]],
  [["mem0", "é", "in_prompt"], ["mem0", "😀", "retrieved_in_scope"], ["mem0", "￿", "in_prompt"]],
  [["a\u0000b", "c", "in_prompt"], ["d", "ü\u0000", "retrieved_in_scope"]],
];
console.log(JSON.stringify({
  ids,
  sets: vectors.map((v) => contextSet(v).hash),
  hashes: ["", "héllo", "😀"].map(contentHash),
}));
led.close();
