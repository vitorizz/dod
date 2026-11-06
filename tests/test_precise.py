import random

import pytest

import dod
from dod.ledger import Ledger
from dod.query import blast_radius, trace


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        yield led


def write(led, rid, context=(), parents=()):
    w = led.append_write(content=rid, store="mem", source="agent_inference",
                         context=list(context), parents=list(parents))
    led.bind(w, "mem", rid)
    return w


def derived(b):
    return {i for its in b.bindings.values() for i in its} - {(b.store, b.record_id)}


def test_a_write_that_read_an_earlier_version_is_left_out(ledger):
    write(ledger, "profile")                                     # profile v1: clean
    write(ledger, "early", context=[("mem", "profile", "in_prompt")])  # read v1
    write(ledger, "bad")
    # profile v2 is rewritten from the bad fact.
    write(ledger, "profile", context=[("mem", "bad", "in_prompt")],
          parents=[("mem", "profile", "in_prompt")])
    write(ledger, "late", context=[("mem", "profile", "in_prompt")])   # read v2
    loose = blast_radius(ledger, "mem", "bad")
    tight = blast_radius(ledger, "mem", "bad", precise=True)
    assert derived(loose) == {("mem", "profile"), ("mem", "early"), ("mem", "late")}
    assert derived(tight) == {("mem", "profile"), ("mem", "late")}
    assert tight.store_counts == {"mem": 2}
    assert all(e.child_write_id in {w.write_id for w in tight.descendants}
               for e in tight.edges)
    r = dod.excise(ledger, "mem", "bad", precise=True, dry_run=True)
    assert sorted(r.targets["mem"]) == ["bad", "late", "profile"]


def test_precise_is_a_subset_and_consistent_with_trace(ledger):
    rng = random.Random(3)
    items = []
    for i in range(400):
        ctx = [("mem", r, rng.choice(["in_prompt", "retrieved_in_scope"]))
               for r in rng.sample(items, min(len(items), rng.randrange(0, 4)))]
        rid = f"r{i}" if rng.random() < .75 or not items else rng.choice(items)
        write(ledger, rid, context=ctx)
        if rid not in items:
            items.append(rid)
    for rid in rng.sample(items, 30):
        for only in (False, True):
            loose = blast_radius(ledger, "mem", rid, in_prompt_only=only)
            tight = blast_radius(ledger, "mem", rid, in_prompt_only=only, precise=True)
            assert derived(tight) <= derived(loose)
            if only:
                continue
            # Every precisely derived write traces back to a version of rid.
            roots = {w.write_id for w in blast_radius(ledger, "mem", rid).items}
            for w in tight.descendants:
                (item,) = tight.bindings[w.write_id][:1] or [None]
                if item is None:
                    continue
                t = trace(ledger, *item)
                ancestors = {a.write_id for a in t.ancestors} | {a.write_id for a in t.items}
                assert ancestors & roots, (rid, item)
