"""Runs examples/false_memory_demo.py and checks its key numbers, so it can't silently rot."""

import runpy
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "false_memory_demo.py"
SUMMARY = "Summary: excised 5 items, preserved 12, stores touched: crewai, langgraph."

pytestmark = [pytest.mark.crewai, pytest.mark.langgraph]


def _run(workdir, capsys):
    workdir.mkdir(exist_ok=True)
    out = runpy.run_path(str(DEMO))["main"](str(workdir))
    return out, capsys.readouterr().out


@pytest.fixture
def offline(monkeypatch):
    pytest.importorskip("crewai")
    pytest.importorskip("langgraph.store.memory")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("the demo must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_demo_numbers(tmp_path, capsys, offline):
    out, printed = _run(tmp_path, capsys)
    text = out["text"]

    assert "90 days" in text[out["bad"]] and text[out["bad"]].startswith("Yes")
    assert "30 days" in text[out["fixed"]] and text[out["fixed"]].startswith("No")

    # The bad reply traces, through the macro (CrewAI) and the FAQ entry
    # (LangGraph), to two roots: the official policy and the poisoned message.
    t = out["trace"]
    assert sorted((w.store, w.source) for w in t.roots) == [
        ("crewai", "document"), ("crewai", "user")]
    assert [w.store for w in t.ancestors] == ["crewai", "langgraph", "crewai", "crewai"]
    assert out["picked"] == out["root"]

    # Four derived items across both stores; five excised with the root.
    assert out["blast"].store_counts == {"crewai": 2, "langgraph": 2}
    assert {s: len(r) for s, r in out["excise"].done.items()} == {"crewai": 3, "langgraph": 2}
    assert out["excise"].skipped == []
    assert len(out["excised"]) == 5
    assert out["root"] in out["excised"] and out["bad"] in out["excised"]

    # Every correct item is kept, including the ones learned after the root.
    assert len(out["correct"]) == 12
    assert len(out["after_root"]) == 3
    assert not set(out["correct"]) & set(out["excised"])

    # The re-run reply traces only to the official policy.
    assert [(w.store, w.source) for w in out["rerun_roots"]] == [("crewai", "document")]

    assert printed.rstrip().endswith(SUMMARY)
    # The output the README shows is the output the demo prints.
    demo = (ROOT / "README.md").read_text().split("## Demo", 1)[1]
    shown = "== " + demo.split("```\n== ", 1)[1].split("```", 1)[0]
    assert printed.rstrip().endswith(shown.rstrip())


def test_demo_output_is_identical_every_run(tmp_path, capsys, offline):
    _, first = _run(tmp_path / "a", capsys)
    _, second = _run(tmp_path / "b", capsys)
    assert first == second


def test_readme_shows_demo_summary():
    readme = (ROOT / "README.md").read_text()
    demo = readme.split("## Demo", 1)[1]
    assert "python examples/false_memory_demo.py" in demo
    assert SUMMARY in demo
