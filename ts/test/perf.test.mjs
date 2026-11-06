import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Capture, Ledger, agentTaskScope, setWalRestartBytes } from "../dist/index.js";

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

// The writer restarts the WAL (a TRUNCATE checkpoint) after a commit once it
// passes the threshold. Here the threshold is 8 MiB instead of 64 MiB, so
// restarts happen in the timed window 8x as often as by default, and the
// worker is off, so every restart runs on the writer.
test("capture write across WAL restarts: p99 < 1 ms (on-disk)", () => {
  setWalRestartBytes(8 << 20);
  try {
    const path = join(mkdtempSync(join(tmpdir(), "dod-ts-perf-")), "l.db");
    const led = new Ledger(path, { checkpointMs: 0 });
    const cap = new Capture(led);
    const wal = () => {
      try { return statSync(path + "-wal").size; } catch { return 0; }
    };
    const t = [];
    const atRestart = [];
    let restarts = 0;
    let prev = 0;
    agentTaskScope("agent", "task", () => {
      cap.onRetrieve(Array.from({ length: 10 }, (_, i) => ["kb", `doc-${i}`]));
      for (let i = 0; i < 21000; i++) {
        const s = process.hrtime.bigint();
        cap.onWrite({ store: "notes", content: `note ${i}`, source: "agent_inference", recordId: `n-${i}` });
        const ms = Number(process.hrtime.bigint() - s) / 1e6;
        const now = wal();  // untimed: a smaller WAL file means this commit restarted it
        if (i >= 1000) {
          t.push(ms);
          if (now < prev) { restarts++; atRestart.push(ms); }
        }
        prev = now;
      }
    });
    led.close();
    t.sort((a, b) => a - b);
    const p50 = t[t.length >> 1];
    const p = t[Math.floor(t.length * 0.99)];
    const worst = atRestart.length ? Math.max(...atRestart) : 0;
    console.log(`ts onWrite across ${restarts} WAL restarts: p50=${p50.toFixed(4)} ms p99=${p.toFixed(4)} ms`
      + ` (slowest restarting commit ${worst.toFixed(4)} ms)`);
    assert.ok(restarts > 0, "no WAL restart in the timed window");
    assert.ok(p / SCALE < 1, `p99 ${p / SCALE} ms`);
  } finally {
    setWalRestartBytes(64 * 1024 * 1024);
  }
});
