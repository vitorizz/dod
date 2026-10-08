# Propagation experiment

How far does one planted false memory spread through a long-lived agent's
memory? The harness plants it, runs the agent, and measures the spread with
the ledger itself.

```bash
pip install -e ".[dev,crewai,langgraph]"
python -m experiments.propagation.run --mode scripted --steps 60 --seed 0 --out results.json
python -m experiments.propagation.run --sweep --out sweep.json   # 5 seeds x use-prob 0.1, 0.5, 0.9
```

Run from the repository root. Scripted mode makes no network calls; a
60-step run takes a few seconds.

## Scenario

A support agent for an online store keeps memory in CrewAI `Memory` and a
LangGraph `InMemoryStore`, both instrumented with one `Capture`
(`scenario.py`).

1. **Seed:** one correct policy fact per topic (`--facts`, default 6) in
   CrewAI, source `document`.
2. **Plant:** one false fact on the first topic (refunds: 90 days instead of
   30), source `user`: a customer's message saved as a policy update.
3. **Steps** (`--steps`): each picks a task, topic and customer with seeded
   randomness and runs in `agent_task_scope(role, "step-NNNN")`.
   - `answer` replies to the customer: LangGraph `("tickets", step)`. These
     replies are the **outputs**.
   - `note` saves a customer note in CrewAI. Past `--max-notes` notes per
     customer the oldest is forgotten (a tombstone).
   - `summarize` rewrites the topic's KB entry in LangGraph (a compaction
     once it exists).

   Every step recalls the topic's facts, all of the customer's notes (any
   topic) and the topic's KB entry. All of them are parents
   (`retrieved_in_scope`). At most `--prompt-k` items about the topic go in
   the prompt (facts first, then the newest notes, then the KB entry) and
   are confirmed `in_prompt`.

**Scripted agent (`--mode scripted`, default).** If the prompt holds a value
other than the policy fact's, the agent uses it with probability
`--use-prob`, otherwise the policy's value. With probability `--error-rate`
it makes an unrelated mistake instead, so not every wrong output comes from
the planted fact. Deterministic for a fixed `--seed`.

**LLM agent (`--mode llm`, manual).** The same scenario, but a real model
picks the value from the same prompt, through CrewAI's `LLM` (any provider
LiteLLM supports; keys come from the environment). One call per step. It
prints the estimated number of calls and stops unless `--yes` is given:

```bash
ANTHROPIC_API_KEY=... python -m experiments.propagation.run --mode llm \
    --model anthropic/<model> --steps 20 --yes
```

`--use-prob` and `--error-rate` don't apply and are reported as `null`.
Never run from tests or CI.

To run it without a key on your machine, start the **Propagation
experiment** workflow (`.github/workflows/experiment.yml`) from the Actions
tab. It takes a model, the steps per run and the number of seeds, and it
reads the key from the `ANTHROPIC_API_KEY` repository secret. Each run's JSON
and printed table are committed to the `experiment-results` branch under
`propagation/`, named by time and run ID. The commit message records the
model and the code SHA. Nothing starts it automatically.

In both modes the harness renders the written text from the chosen value,
so the ground truth is exact: an item is **wrong** if its value differs from
its topic's true value. `ground_truth()` is the only code that reads store
content for checking, and it returns item IDs only.

## Metrics

Lineage is computed only from the ledger (`metrics.py`): `trace`,
`blast_radius`, a dry-run `excise`, tombstones and edge tags. The
ground-truth set labels items; it never adds or removes edges.

| Metric | Definition |
|---|---|
| `derived_writes` | Writes in the planted item's blast radius (depth >= 1), over all edges. |
| `derived_writes_in_prompt` | The same, following `in_prompt` edges only. |
| `derived_over_time` | Cumulative `derived_writes` after each step (step from the write's `session_id`). |
| `max_depth` | Deepest descendant. |
| `stores_reached` | Descendant writes per store. |
| `contaminating_edges` | Edges `blast_radius` walked forward, by tag, and the `in_prompt` share. |
| `deleted_derived_items` | Derived items that already have a tombstone. |
| `excise` | Live items (bound, no tombstone) a quarantine `excise` of the planted item would remove, split wrong/correct. |
| `excise_in_prompt` | The same over `in_prompt` edges only (the planted item plus items bound to `derived_writes_in_prompt`). |
| `wipe` | Every live item: what wiping all memory removes. |
| `outputs.fraction_wrong_tracing` | Of the wrong outputs, the share whose `trace` reaches the planted item's write. |
| `outputs.fraction_wrong_tracing_in_prompt` | The same over `in_prompt` edges only. |
| `outputs.correct_tracing_to_planted` | Correct outputs whose trace still reaches it (over-inclusion). |

## `results.json`

IDs and counts only; no memory content.

```jsonc
{
  "schema_version": 1,
  "config": {"mode": "scripted", "steps": 60, "error_rate": 0.05, "facts": 6,
             "customers": 4, "prompt_k": 4, "max_notes": 4, "model": null},
  "runs": [{
    "seed": 0, "use_prob": 0.5,               // use_prob is null in llm mode
    "tasks": {"answer": 29, "note": 15, "summarize": 16},
    "ledger": {"writes": 67, "items": 57, "tombstoned_items": 3},
    "planted": [{                               // one entry per planted fact
      "store": "crewai", "record_id": "<id>",
      "derived_writes": 41, "derived_writes_in_prompt": 9,
      "derived_over_time": [0, 0, 0, 0, 1, 1, "... one per step"],
      "max_depth": 4, "stores_reached": {"crewai": 11, "langgraph": 30},
      "contaminating_edges": {"in_prompt": 139, "retrieved_in_scope": 34, "in_prompt_share": 0.8035},
      "deleted_derived_items": 0,
      "excise": {"items": 35, "wrong": 7, "correct": 28},
      "excise_in_prompt": {"items": 9, "wrong": 5, "correct": 4}
    }],
    "outputs": {"total": 29, "wrong": 3, "wrong_tracing_to_planted": 2,
                "wrong_tracing_to_planted_in_prompt": 1, "fraction_wrong_tracing": 0.6667,
                "fraction_wrong_tracing_in_prompt": 0.3333, "correct_tracing_to_planted": 18},
    "wipe": {"items": 54, "wrong": 9, "correct": 45},
    "headline": {"derived_writes": 41, "...": "flat numbers used by the table and summary"}
  }],
  "summary": [{                                 // one entry per use_prob
    "use_prob": 0.5, "runs": 5, "seeds": [0, 1, 2, 3, 4],
    "headline": {"derived_writes": {"mean": 39.2, "min": 25, "max": 51}, "...": null}
  }],
  "elapsed_s": 4.4
}
```

Fractions are `null` when their denominator is 0; a summary stat is `null`
when every run's value is. Record IDs are random per run, so compare runs
on everything except `record_id`.
