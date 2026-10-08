"""The propagation experiment harness (experiments/propagation).

Metrics are checked against a small hand-built ledger (core only). The
scenario tests need crewai and langgraph, run offline, and stay well under
30 s together.
"""

import json
import socket
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dod.capture import Capture  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from experiments.propagation import metrics, run  # noqa: E402
from experiments.propagation.scenario import Config  # noqa: E402


def _ticket(step):
    return json.dumps(["tickets", f"step-{step:04d}", "reply"])


@pytest.fixture
def hand_ledger(tmp_path):
    """F is a correct fact, P the planted one.

    step 1: note n1      <- P (in_prompt), F (in_prompt)         wrong
    step 2: output O1    <- n1 (in_prompt), F (in_prompt)        wrong
    step 3: output O2    <- P (retrieved_in_scope), F (in_prompt) correct
    step 3: output O3    <- F (in_prompt)                         wrong, unrelated
    step 4: note n2      <- O2 (retrieved_in_scope)               correct
    step 4: n1 is forgotten (tombstone)
    """
    path = str(tmp_path / "ledger.db")
    ledger = Ledger(path)
    cap = Capture(ledger)
    F, P, n1, n2 = ("crewai", "f"), ("crewai", "p"), ("crewai", "n1"), ("crewai", "n2")
    O1, O2, O3 = (("langgraph", _ticket(s)) for s in (2, 3, 33))

    def write(step, item, ip=(), ris=()):
        key = ("agent", step)
        cap.on_retrieve(key, ris)
        cap.on_retrieve(key, ip, "in_prompt")
        w = cap.on_write(key, item[1], item[0], "agent_inference", "agent",
                         f"step-{step:04d}" if isinstance(step, int) else step)
        cap.bind(w, item[1])
        cap.reset(key)

    write("seed", F)
    write("plant", P)
    write(1, n1, ip=[P, F])
    write(2, O1, ip=[n1, F])
    write(3, O2, ip=[F], ris=[P])
    write(3, O3, ip=[F])
    write(4, n2, ris=[O2])
    cap.on_delete("crewai", ["n1"], "forget", session_id="step-0004")
    wrong = {P, n1, O1, O3}
    yield ledger, path, P, wrong, (O1, O2, O3)
    ledger.close()


def test_metrics_match_hand_computed(hand_ledger):
    ledger, path, P, wrong, (O1, O2, O3) = hand_ledger
    m = metrics.compute(ledger, path, [P], wrong, steps=4)

    assert m["ledger"] == {"writes": 7, "items": 7, "tombstoned_items": 1}
    (p,) = m["planted"]
    assert (p["store"], p["record_id"]) == P
    assert p["derived_writes"] == 4  # n1, O1, O2, n2
    assert p["derived_writes_in_prompt"] == 2  # n1, O1
    assert p["derived_over_time"] == [1, 2, 3, 4]
    assert p["max_depth"] == 2
    assert p["stores_reached"] == {"crewai": 2, "langgraph": 2}
    assert p["contaminating_edges"] == {
        "in_prompt": 2, "retrieved_in_scope": 2, "in_prompt_share": 0.5}
    assert p["deleted_derived_items"] == 1  # n1
    # Live targets: P, O1, O2, n2 (n1 is already gone).
    assert p["excise"] == {"items": 4, "wrong": 2, "correct": 2}
    # Over in_prompt edges only: P, O1 (n1 is gone).
    assert p["excise_in_prompt"] == {"items": 2, "wrong": 2, "correct": 0}
    # Live items: F, P, O1, O2, O3, n2.
    assert m["wipe"] == {"items": 6, "wrong": 3, "correct": 3}
    assert m["outputs"] == {
        "total": 3,
        "wrong": 2,  # O1, O3
        "wrong_tracing_to_planted": 1,  # O1
        "wrong_tracing_to_planted_in_prompt": 1,
        "fraction_wrong_tracing": 0.5,
        "fraction_wrong_tracing_in_prompt": 0.5,
        "correct_tracing_to_planted": 1,  # O2, through a retrieved_in_scope edge
    }


def test_is_output():
    assert metrics.is_output(("langgraph", _ticket(1)))
    assert not metrics.is_output(("langgraph", json.dumps(["kb", "refunds"])))
    assert not metrics.is_output(("crewai", _ticket(1)))
    assert not metrics.is_output(("langgraph", "not json"))


def test_summarize_reports_mean_and_range():
    runs = [{"seed": s, "use_prob": 0.5,
             "headline": dict.fromkeys((k for k, _ in run.HEADLINE), s)} for s in (1, 3)]
    runs[0]["headline"]["fraction_wrong_tracing"] = None
    (s,) = run.summarize(runs)
    assert s["runs"] == 2 and s["seeds"] == [1, 3]
    assert s["headline"]["derived_writes"] == {"mean": 2.0, "min": 1, "max": 3}
    assert s["headline"]["fraction_wrong_tracing"] == {"mean": 3.0, "min": 3, "max": 3}


def test_llm_mode_needs_model_and_yes(monkeypatch, capsys, tmp_path):
    def never(*a, **k):
        raise AssertionError("llm mode must not run without --yes")

    monkeypatch.setattr(run, "run_one", never)
    monkeypatch.delenv("MODEL", raising=False)
    out = str(tmp_path / "r.json")
    assert run.main(["--mode", "llm", "--steps", "7", "--out", out]) == 2
    assert run.main(["--mode", "llm", "--model", "x/y", "--steps", "7", "--out", out]) == 2
    assert "~7 LLM calls" in capsys.readouterr().out
    assert run.main(["--mode", "llm", "--model", "x/y", "--steps", "7", "--sweep",
                     "--sweep-seeds", "3", "--out", out]) == 2
    assert "~21 LLM calls" in capsys.readouterr().out
    assert not Path(out).exists()


# Scenario runs: real crewai Memory and LangGraph InMemoryStore, offline.

scenario_marks = [pytest.mark.crewai, pytest.mark.langgraph]


@pytest.fixture
def offline(monkeypatch):
    pytest.importorskip("crewai")
    pytest.importorskip("langgraph.store.memory")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("scripted mode must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _strip_ids(results):
    for r in results["runs"]:
        for p in r["planted"]:
            p.pop("record_id")
    return results["runs"]


@pytest.mark.crewai
@pytest.mark.langgraph
def test_scripted_is_deterministic_and_content_free(offline, tmp_path, capsys):
    args = ["--steps", "20", "--seed", "7", "--facts", "3", "--customers", "3"]
    t0 = time.perf_counter()
    assert run.main(args + ["--out", str(tmp_path / "a.json")]) == 0
    assert run.main(args + ["--out", str(tmp_path / "b.json")]) == 0
    assert time.perf_counter() - t0 < 30
    a_text = (tmp_path / "a.json").read_text()
    a, b = json.loads(a_text), json.loads((tmp_path / "b.json").read_text())

    assert a["schema_version"] == 1
    assert a["config"]["mode"] == "scripted" and a["config"]["steps"] == 20
    assert _strip_ids(a) == _strip_ids(b)
    (r,) = a["runs"]
    assert sum(r["tasks"].values()) == 20
    assert len(r["planted"][0]["derived_over_time"]) == 20
    # IDs and counts only: no store content in the results.
    for word in ("refunds are", "Policy", "Told", "Reply to", "days after"):
        assert word not in a_text
    assert "derived" in capsys.readouterr().out


@pytest.mark.crewai
@pytest.mark.langgraph
def test_use_prob_bounds(offline):
    base = dict(steps=20, seed=1, facts=2, customers=2, error_rate=0.0)
    never = run.run_one(Config(use_prob=0.0, **base))
    always = run.run_one(Config(use_prob=1.0, **base))

    # Never using the false fact: no wrong output, but over-inclusion still
    # links outputs that only had it in scope.
    assert never["outputs"]["wrong"] == 0
    assert never["wipe"]["wrong"] == 1  # the planted fact itself
    assert never["planted"][0]["excise"]["wrong"] == 1

    # Always using it: every wrong output traces to it, over in_prompt edges too.
    o = always["outputs"]
    assert o["wrong"] > 0
    assert o["fraction_wrong_tracing"] == 1.0
    assert o["fraction_wrong_tracing_in_prompt"] == 1.0
    p = always["planted"][0]
    assert p["derived_writes"] >= p["derived_writes_in_prompt"] > 0
    assert p["derived_over_time"] == sorted(p["derived_over_time"])
    assert p["derived_over_time"][-1] == p["derived_writes"]
    # Excision keeps every correct item a wipe would lose, except over-inclusion.
    assert p["excise"]["wrong"] == always["wipe"]["wrong"]
    assert p["excise"]["items"] < always["wipe"]["items"]


@pytest.mark.crewai
@pytest.mark.langgraph
def test_sweep_aggregates_per_use_prob(offline, tmp_path):
    out = tmp_path / "s.json"
    assert run.main(["--sweep", "--sweep-seeds", "2", "--use-probs", "0,1", "--steps", "8",
                     "--facts", "2", "--customers", "2", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert [(r["seed"], r["use_prob"]) for r in res["runs"]] == [
        (0, 0.0), (1, 0.0), (0, 1.0), (1, 1.0)]
    assert [(s["use_prob"], s["runs"]) for s in res["summary"]] == [(0.0, 2), (1.0, 2)]


def test_llm_agent_sampling_by_provider(monkeypatch):
    pytest.importorskip("crewai")
    import crewai

    from experiments.propagation.scenario import Config, LLMAgent

    made = []
    monkeypatch.setattr(crewai, "LLM", lambda **kw: made.append(kw))
    LLMAgent(Config(mode="llm", model="anthropic/claude-sonnet-5-5"))
    LLMAgent(Config(mode="llm", model="openai/gpt-x"))
    assert made == [{"model": "anthropic/claude-sonnet-5-5"},
                    {"model": "openai/gpt-x", "temperature": 0}]
