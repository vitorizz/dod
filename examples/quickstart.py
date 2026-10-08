"""Quickstart: one ledger, a CrewAI memory and a LangGraph store.

A false fact is saved, one step derives a LangGraph item from it, a second
step derives a CrewAI memory from that. Then: trace the last memory back to
the false fact, list everything derived from it, and quarantine all of it.

    pip install -e ".[crewai,langgraph]"
    python examples/quickstart.py

Runs offline: a toy embedder stands in for a real one, and every call below
skips crewai's LLM (explicit scope/categories/importance, shallow recall).
"""

import hashlib
import os
import sys
import tempfile

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")  # crewai's, not ours
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai.memory.unified_memory import Memory  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from dod.capture import Capture  # noqa: E402
from dod.integrations import crewai, langgraph  # noqa: E402
from dod.ledger import Ledger  # noqa: E402
from dod.query import blast_radius, trace  # noqa: E402
from dod.scope import agent_task_scope  # noqa: E402


def toy_embed(texts):
    # Bag of words hashed into 64 dims. Use a real embedder in production.
    out = []
    for text in texts:
        v = [0.0] * 64
        for word in text.lower().split():
            v[hashlib.sha256(word.encode()).digest()[0] % 64] += 1.0
        out.append(v)
    return out


def main(workdir):
    ledger = Ledger(os.path.join(workdir, "ledger.db"))
    cap = Capture(ledger)
    memory = crewai.instrument(
        Memory(storage=os.path.join(workdir, "crewai"), embedder=toy_embed), cap
    )
    store = langgraph.instrument(InMemoryStore(), cap)
    ns = ("users", "ana")

    def remember(text, source=None):
        return memory.remember(text, scope="/", categories=[], importance=0.5,
                               source=source)

    # A false fact gets saved.
    with agent_task_scope("assistant", "onboarding"):
        root = remember("Ana is allergic to peanuts", source="user")

    # Step 1: recall from CrewAI, write to LangGraph.
    with agent_task_scope("assistant", "meal-plan"):
        memory.recall("what is Ana allergic to", depth="shallow")
        store.put(ns, "meal-plan", {"text": "no peanut dishes this week"})

    # Step 2: read from LangGraph, write to CrewAI.
    with agent_task_scope("assistant", "groceries"):
        store.get(ns, "meal-plan")
        bad = remember("Removed peanut butter from the grocery list")

    # The bad output traces back to the false fact, across both stores.
    t = trace(ledger, "crewai", bad.id)
    print("trace roots:", [(w.store, w.write_id) for w in t.roots])
    print("path:", [(e.parent_store, e.parent_record_id) for e in t.edges])

    # Everything derived from the false fact.
    b = blast_radius(ledger, "crewai", root.id)
    print("blast radius:", b.store_counts)
    affected = sorted({item for items in b.bindings.values() for item in items})

    # Quarantine (reversible): instrumented CrewAI recall stops returning them.
    ledger.quarantine(affected, reason="false allergy", actor="oncall")
    with agent_task_scope("assistant", "after"):
        after = memory.recall("what is Ana allergic to", depth="shallow")
    print("recall after quarantine:", [m.record.content for m in after])

    memory.close()
    ledger.close()
    return {"root": root, "bad": bad, "trace": t, "blast": b,
            "affected": affected, "after": after}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        main(sys.argv[1] if len(sys.argv) > 1 else d)
