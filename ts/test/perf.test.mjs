import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Capture, Ledger, agentTaskScope } from "../dist/index.js";

const SCALE = Number(process.env.DOD_PERF_SCALE || 1);

function p99(fn, n = 5000, warmup = 1000) {
  for (let i = 0; i < warmup; i++) fn(i);
  const t = [];
  for (let i = 0; i < n; i++) {
    const s = process.hrtime.bigint();
    fn(i);
    t.push(Number(process.hrtime.bigint() - s) / 1e6);
  }
  t.sort((a, b) => a - b);
  return { p50: t[n >> 1], p99: t[Math.floor(n * 0.99)] };
}

test("capture write with 10 context items: p99 < 1 ms (on-disk)", () => {
  const led = new Ledger(join(mkdtempSync(join(tmpdir(), "dod-ts-perf-")), "l.db"));
  const cap = new Capture(led);
  const rounds = [];
  agentTaskScope("agent", "task", () => {
    cap.onRetrieve(Array.from({ length: 10 }, (_, i) => ["kb", `doc-${i}`]));
    for (let r = 0; r < 3; r++) {
      rounds.push(p99((i) => cap.onWrite({ store: "notes", content: `note ${r} ${i}`,
        source: "agent_inference", recordId: `n-${r}-${i}` })));
    }
  });
  led.close();
  for (const [i, x] of rounds.entries()) console.log(`ts onWrite round ${i + 1}/3: p50=${x.p50.toFixed(4)} ms p99=${x.p99.toFixed(4)} ms`);
  const median = rounds.map((x) => x.p99).sort((a, b) => a - b)[1] / SCALE;
  assert.ok(median < 1, `p99 ${median} ms`);
});
