"""Propagation experiment runner.

    python -m experiments.propagation.run --mode scripted --steps 60 --seed 0 --out results.json
    python -m experiments.propagation.run --sweep --out sweep.json
    python -m experiments.propagation.run --mode llm --model <provider/model> --steps 20 --yes

See README.md for the scenario, metric definitions and the results schema.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from typing import List, Optional

from experiments.propagation import metrics
from experiments.propagation.scenario import Config

SCHEMA_VERSION = 1

# Flat per-run numbers that the summary table and the sweep aggregate.
HEADLINE = [
    ("derived_writes", "derived"),
    ("derived_writes_in_prompt", "in_prompt"),
    ("max_depth", "depth"),
    ("stores_reached", "stores"),
    ("in_prompt_share", "ip_share"),
    ("outputs_wrong", "wrong_out"),
    ("fraction_wrong_tracing", "traced"),
    ("fraction_wrong_tracing_in_prompt", "traced_ip"),
    ("excise_items", "excise"),
    ("excise_correct", "excise_ok"),
    ("excise_in_prompt_items", "excise_ip"),
    ("excise_in_prompt_correct", "excise_ip_ok"),
    ("wipe_items", "wipe"),
    ("wipe_correct", "wipe_ok"),
]


def headline(m: dict) -> dict:
    p, o = m["planted"][0], m["outputs"]
    return {
        "derived_writes": p["derived_writes"],
        "derived_writes_in_prompt": p["derived_writes_in_prompt"],
        "max_depth": p["max_depth"],
        "stores_reached": len(p["stores_reached"]),
        "in_prompt_share": p["contaminating_edges"]["in_prompt_share"],
        "outputs_wrong": o["wrong"],
        "fraction_wrong_tracing": o["fraction_wrong_tracing"],
        "fraction_wrong_tracing_in_prompt": o["fraction_wrong_tracing_in_prompt"],
        "excise_items": p["excise"]["items"],
        "excise_correct": p["excise"]["correct"],
        "excise_in_prompt_items": p["excise_in_prompt"]["items"],
        "excise_in_prompt_correct": p["excise_in_prompt"]["correct"],
        "wipe_items": m["wipe"]["items"],
        "wipe_correct": m["wipe"]["correct"],
    }


def run_one(cfg: Config, agent=None) -> dict:
    """One scenario run in a temporary directory; returns its results entry."""
    from dod.ledger import Ledger
    from experiments.propagation import scenario

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.db")
        ledger = Ledger(path)
        try:
            r = scenario.run(cfg, d, ledger, agent)
            m = metrics.compute(ledger, path, r.planted, r.wrong, cfg.steps)
        finally:
            ledger.close()
    use_prob = cfg.use_prob if cfg.mode == "scripted" else None
    return {"seed": cfg.seed, "use_prob": use_prob, "tasks": r.tasks, **m,
            "headline": headline(m)}


def summarize(runs: List[dict]) -> List[dict]:
    """Mean, min and max of each headline number, per use probability."""
    groups = {}
    for r in runs:
        groups.setdefault(r["use_prob"], []).append(r)
    out = []
    for p, rs in groups.items():
        stats = {}
        for key, _ in HEADLINE:
            vals = [r["headline"][key] for r in rs if r["headline"][key] is not None]
            stats[key] = ({"mean": round(sum(vals) / len(vals), 4), "min": min(vals),
                           "max": max(vals)} if vals else None)
        out.append({"use_prob": p, "runs": len(rs), "seeds": [r["seed"] for r in rs],
                    "headline": stats})
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def table(results: dict) -> str:
    cols = ["seed", "p_use"] + [short for _, short in HEADLINE]
    rows = [[_fmt(r["seed"]), _fmt(r["use_prob"])] + [_fmt(r["headline"][k]) for k, _ in HEADLINE]
            for r in results["runs"]]
    if len(results["runs"]) > 1:
        for s in results["summary"]:
            h = s["headline"]
            rows.append([f"mean({s['runs']})", _fmt(s["use_prob"])]
                        + [_fmt(h[k]["mean"]) if h[k] else "-" for k, _ in HEADLINE])
            rows.append(["range", ""]
                        + [f"{_fmt(h[k]['min'])}..{_fmt(h[k]['max'])}" if h[k] else "-"
                           for k, _ in HEADLINE])
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    line = lambda r: "  ".join(v.rjust(w) for v, w in zip(r, widths))  # noqa: E731
    return "\n".join([line(cols), line(["-" * w for w in widths])] + [line(r) for r in rows])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    d = Config()
    ap = argparse.ArgumentParser(prog="python -m experiments.propagation.run",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["scripted", "llm"], default=d.mode)
    ap.add_argument("--steps", type=int, default=d.steps)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--use-prob", type=float, default=d.use_prob,
                    help="scripted: probability a conflicting value in the prompt is used")
    ap.add_argument("--error-rate", type=float, default=d.error_rate,
                    help="scripted: probability of an unrelated mistake per step")
    ap.add_argument("--facts", type=int, default=d.facts, help="correct facts (topics) seeded")
    ap.add_argument("--customers", type=int, default=d.customers)
    ap.add_argument("--prompt-k", type=int, default=d.prompt_k)
    ap.add_argument("--max-notes", type=int, default=d.max_notes)
    ap.add_argument("--sweep", action="store_true",
                    help="run --sweep-seeds seeds from --seed, for each of --use-probs")
    ap.add_argument("--sweep-seeds", type=int, default=5)
    ap.add_argument("--use-probs", default="0.1,0.5,0.9")
    ap.add_argument("--model", default=os.environ.get("MODEL"),
                    help="llm: CrewAI LLM model string (default: $MODEL)")
    ap.add_argument("--yes", action="store_true", help="llm: confirm the LLM calls")
    ap.add_argument("--out", default="results.json")
    return ap.parse_args(argv)


def configs(a: argparse.Namespace) -> List[Config]:
    base = dict(mode=a.mode, steps=a.steps, error_rate=a.error_rate, facts=a.facts,
                customers=a.customers, prompt_k=a.prompt_k, max_notes=a.max_notes,
                model=a.model)
    if not a.sweep:
        return [Config(seed=a.seed, use_prob=a.use_prob, **base)]
    probs = [a.use_prob] if a.mode == "llm" else [float(p) for p in a.use_probs.split(",")]
    return [Config(seed=a.seed + i, use_prob=p, **base)
            for p in probs for i in range(a.sweep_seeds)]


def main(argv: Optional[List[str]] = None) -> int:
    a = parse_args(argv)
    try:
        cfgs = configs(a)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.mode == "llm":
        if not a.model:
            print("error: --mode llm needs --model (or $MODEL), e.g. anthropic/<model>",
                  file=sys.stderr)
            return 2
        calls = sum(c.steps for c in cfgs)
        print(f"llm mode: {len(cfgs)} run(s) x {a.steps} steps = ~{calls} LLM calls "
              f"to {a.model}, one per step.")
        if not a.yes:
            print("Not started. Re-run with --yes to proceed.")
            return 2

    t0 = time.perf_counter()
    runs = [run_one(c) for c in cfgs]
    results = {
        "schema_version": SCHEMA_VERSION,
        "config": {k: v for k, v in cfgs[0].public().items() if k not in ("seed", "use_prob")},
        "runs": runs,
        "summary": summarize(runs),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(table(results))
    print(f"\nWrote {a.out} ({len(runs)} run(s), {results['elapsed_s']} s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
