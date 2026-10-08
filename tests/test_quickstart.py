"""Runs examples/quickstart.py, the README's quickstart, so the README can't go stale."""

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "quickstart.py"


def _lines(text):
    # Indentation and the example's import-order noqa markers don't count.
    return [s.replace("  # noqa: E402", "") for s in map(str.strip, text.splitlines()) if s]


def test_readme_quickstart_matches_example():
    readme = (ROOT / "README.md").read_text()
    snippet = _lines(readme.split("```python\n", 1)[1].split("```", 1)[0])
    example = iter(_lines(EXAMPLE.read_text()))
    # Every README line appears in the example, in the same order.
    missing = [line for line in snippet if line not in example]
    assert not missing


@pytest.mark.crewai
@pytest.mark.langgraph
def test_quickstart_runs(tmp_path, monkeypatch, capsys):
    pytest.importorskip("crewai")
    pytest.importorskip("langgraph.store.memory")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # must run offline
    out = runpy.run_path(str(EXAMPLE))["main"](str(tmp_path))
    root, bad, t, b = out["root"], out["bad"], out["trace"], out["blast"]

    (root_write,) = t.roots
    assert (root_write.store, root_write.source) == ("crewai", "user")
    assert [w.write_id for w in b.items] == [root_write.write_id]
    assert [w.store for w in t.ancestors] == ["langgraph", "crewai"]

    assert b.store_counts == {"crewai": 1, "langgraph": 1}
    lg_id = '["users", "ana", "meal-plan"]'
    assert out["affected"] == sorted(
        [("crewai", root.id), ("crewai", bad.id), ("langgraph", lg_id)]
    )
    assert out["after"] == []
    assert "blast radius: {'crewai': 1, 'langgraph': 1}" in capsys.readouterr().out
