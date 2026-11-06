"""The TypeScript capture package writes ledgers the Python library reads.

Runs ts/test/golden.mjs (needs node >= 22.13 and a built ts/: ``npm ci &&
npm run build`` in ts/), then checks the file with the Python library.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TS = ROOT / "ts"
sys.path.insert(0, str(ROOT / "scripts"))

from export_ts_schema import OUT, render  # noqa: E402

from dod.capture import _content_hash  # noqa: E402
from dod.ledger import Ledger, _context_set  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402

needs_ts = pytest.mark.skipif(
    not shutil.which("node") or not (TS / "dist" / "index.js").exists(),
    reason="node or a built ts/ package (cd ts && npm ci && npm run build) is missing")


@pytest.mark.skipif(not TS.exists(), reason="no ts/ (an sdist ships the Python package only)")
def test_ts_schema_is_current():
    assert OUT.read_text() == render(), "run python scripts/export_ts_schema.py"


@pytest.fixture
def golden(tmp_path):
    path = tmp_path / "ts.db"
    out = subprocess.run(["node", "--no-warnings", str(TS / "test" / "golden.mjs"), str(path)],
                         check=True, capture_output=True, text=True).stdout
    return path, json.loads(out)


@needs_ts
def test_python_reads_a_ts_ledger(golden):
    path, g = golden
    with Ledger(path) as led:
        ids = g["ids"]
        derived = {i for its in blast_radius(led, "kb", "policy").bindings.values() for i in its}
        assert {("crm", "R1"), ("notes", "ana.md")} <= derived
        assert ("crm", "R2") not in derived
        only = {i for its in blast_radius(led, "kb", "policy", in_prompt_only=True)
                .bindings.values() for i in its}
        assert ("crm", "R1") in only and ("notes", "ana.md") not in only
        t = trace(led, "notes", "ana.md")
        assert {w.write_id for w in t.ancestors} == {ids["d1"], ids["bad"]}
        assert t.deleted[("notes", "ana.md")][0].reason == "forget"
        # Python keeps appending to the same file, and its IDs stay ordered.
        w = led.append_write(content="py", store="kb", source="user",
                             context=[("crm", "R1", "in_prompt")])
        assert w > max(ids.values())
        led.bind(w, "kb", "py")
        assert ("kb", "py") in {i for its in blast_radius(led, "crm", "R1").bindings.values()
                                for i in its}
        assert all(len(i) == 32 and int(i, 16) for i in ids.values())


@needs_ts
def test_ts_ledger_holds_no_content(golden):
    path, _ = golden
    for f in path.parent.glob(path.name + "*"):
        data = f.read_bytes()
        for text in ("refunds within 90 days", "told Ana 90 days", "Ana gets a refund"):
            assert text.encode() not in data, (f, text)


@needs_ts
def test_content_addresses_and_hashes_match(golden):
    _, g = golden
    vectors = [
        [("s", "x", "in_prompt"), ("s", "y", "retrieved_in_scope")],
        [("mem0", "é", "in_prompt"), ("mem0", "😀", "retrieved_in_scope"),
         ("mem0", "￿", "in_prompt")],
        [("a\0b", "c", "in_prompt"), ("d", "ü\0", "retrieved_in_scope")],
    ]
    assert g["sets"] == [_context_set(v)[1] for v in vectors]
    assert g["hashes"] == [_content_hash(s) for s in ("", "héllo", "😀")]


@needs_ts
def test_ts_refuses_audited_ledgers(tmp_path):
    path = tmp_path / "audited.db"
    with Ledger(path, audit=True) as led:
        led.append_write(content="x", store="s", source="user")
    r = subprocess.run(["node", "--no-warnings", str(TS / "test" / "golden.mjs"), str(path)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "audited" in r.stderr
