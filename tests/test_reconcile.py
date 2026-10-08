import hashlib
import random
import sqlite3
import time

import pytest
from perf import SCALE

from dod.capture import Capture
from dod.ledger import Ledger
from dod.reconcile import KINDS, Finding, reconcile

STORE = "mem"


def h(content):
    return hashlib.sha256(content.encode()).hexdigest()


class FakeStore:
    """A store adapter over a dict ``record_id -> content``. ``hashes=False``
    makes it existence-only. ``on_get`` runs before each ``get``."""

    def __init__(self, items=None, hashes=True):
        self.items = dict(items or {})
        self.hashes = hashes
        self.on_get = None
        self.scans = self.gets = 0

    def _hash(self, content):
        return h(content) if self.hashes else None

    def scan(self):
        self.scans += 1
        return [(r, self._hash(c)) for r, c in self.items.items()]

    def get(self, record_id):
        self.gets += 1
        if self.on_get:
            self.on_get(record_id)
        if record_id not in self.items:
            return False, None
        return True, self._hash(self.items[record_id])


@pytest.fixture
def path(tmp_path):
    return tmp_path / "ledger.db"


@pytest.fixture
def ledger(path):
    with Ledger(path) as led:
        yield led


@pytest.fixture
def cap(ledger):
    return Capture(ledger)


def write(cap, store_name, record_id, content, key="k", parents=()):
    """Record a write through capture, as an integration would, and bind it."""
    if parents:
        cap.on_retrieve(key, [(store_name, p) for p in parents])
    wid = cap.on_write(key, content, store_name, "agent_inference")
    cap.bind(wid, record_id)
    return wid


def test_matching_store_is_ok(ledger, cap):
    store = FakeStore()
    for i in range(5):
        store.items[f"r{i}"] = f"c{i}"
        write(cap, STORE, f"r{i}", f"c{i}")
    report = reconcile(ledger, {STORE: store}, grace=0)
    assert report.ok and report.findings == []
    assert report.scanned == {STORE: 5} and report.live == {STORE: 5}
    assert report.content_checked == {STORE: True}
    assert store.gets == 0  # nothing to re-check


def test_each_kind_is_reported(ledger, cap):
    store = FakeStore()
    for r in ("kept", "changed", "gone", "deleted"):
        store.items[r] = r
        write(cap, STORE, r, r)
    # Outside capture: an insert, an edit, a deletion; and a recorded
    # deletion that the store didn't apply (or a re-insert).
    store.items["extra"] = "x"
    store.items["changed"] = "edited"
    del store.items["gone"]
    cap.on_delete(STORE, ["deleted"], "forget")

    report = reconcile(ledger, {STORE: store}, grace=0)
    kinds = {f.record_id: f.kind for f in report.findings}
    assert kinds == {"extra": "unrecorded", "changed": "modified", "gone": "missing",
                     "deleted": "resurrected"}
    by = {f.record_id: f for f in report.findings}
    assert by["changed"].ledger_hash == h("changed") and by["changed"].store_hash == h("edited")
    assert by["gone"].ledger_hash == h("gone") and by["gone"].store_hash is None
    assert by["extra"].ledger_hash is None and by["extra"].store_hash == h("x")
    assert by["deleted"].ledger_hash == h("deleted")
    assert set(report.by_kind()) == set(KINDS)
    assert report.live == {STORE: 3}  # kept, changed, gone
    assert not report.ok


def test_findings_are_sorted_and_typed(ledger, cap):
    store = FakeStore({"b": "1", "a": "2"})
    other = FakeStore({"z": "3"})
    report = reconcile(ledger, {"s2": other, "s1": store}, grace=0)
    assert [(f.store, f.record_id) for f in report.findings] == [
        ("s1", "a"), ("s1", "b"), ("s2", "z")]
    assert all(isinstance(f, Finding) for f in report.findings)


def test_existence_only_store_never_reports_modified(ledger, cap):
    store = FakeStore(hashes=False)
    store.items["a"] = "a"
    write(cap, STORE, "a", "a")
    store.items["a"] = "edited outside capture"
    report = reconcile(ledger, {STORE: store}, grace=0)
    assert report.ok and report.content_checked == {STORE: False}


def test_recompaction_is_compared_with_the_latest_write(ledger, cap):
    store = FakeStore({"a": "v1"})
    write(cap, STORE, "a", "v1")
    wid = cap.on_compaction("k", [(STORE, "a")], "v2", STORE, "agent_inference")
    cap.bind(wid, "a")
    store.items["a"] = "v2"
    assert reconcile(ledger, {STORE: store}, grace=0).ok
    store.items["a"] = "v1"  # rolled back outside capture
    (f,) = reconcile(ledger, {STORE: store}, grace=0).findings
    assert (f.kind, f.ledger_hash, f.store_hash) == ("modified", h("v2"), h("v1"))


def test_recreated_after_tombstone_is_live(ledger, cap):
    store = FakeStore({"a": "v1"})
    write(cap, STORE, "a", "v1")
    cap.on_delete(STORE, ["a"], "forget")
    time.sleep(0.001)
    write(cap, STORE, "a", "v2")
    store.items["a"] = "v2"
    report = reconcile(ledger, {STORE: store}, grace=0)
    assert report.ok and report.live == {STORE: 1}
    # And deleted again, through capture: nothing live, nothing found.
    cap.on_delete(STORE, ["a"], "forget")
    del store.items["a"]
    report = reconcile(ledger, {STORE: store}, grace=0)
    assert report.ok and report.live == {STORE: 0}


def test_unrecorded_item_lists_the_writes_it_was_read_into(ledger, cap):
    store = FakeStore({"seed": "s"})
    write(cap, STORE, "seed", "s")
    store.items["planted"] = "written outside capture"
    # The planted item is read (retrieved) and two writes derive from it; a
    # third write in another scope doesn't.
    w1 = write(cap, STORE, "d1", "d1", key="t1", parents=["planted", "seed"])
    w2 = write(cap, STORE, "d2", "d2", key="t1")  # same scope: context still holds it
    write(cap, STORE, "d3", "d3", key="t2", parents=["seed"])
    # As a direct parent of a compaction too.
    w4 = cap.on_compaction("t3", [(STORE, "planted")], "c", STORE, "agent_inference")
    cap.bind(w4, "d4")
    for r in ("d1", "d2", "d3", "d4"):
        store.items[r] = r if r != "d4" else "c"
    (f,) = reconcile(ledger, {STORE: store}, grace=0).findings
    assert (f.record_id, f.kind) == ("planted", "unrecorded")
    assert f.read_into == (w1, w2, w4)  # in ledger order


def test_resurrected_item_lists_only_writes_after_its_tombstone(ledger, cap):
    store = FakeStore({"a": "a"})
    write(cap, STORE, "a", "a")
    before = write(cap, STORE, "b", "b", key="t1", parents=["a"])
    cap.on_delete(STORE, ["a"], "forget")
    time.sleep(0.001)
    after = write(cap, STORE, "c", "c", key="t2", parents=["a"])  # read while "deleted"
    store.items.update(b="b", c="c")
    (f,) = reconcile(ledger, {STORE: store}, grace=0).findings
    assert (f.record_id, f.kind) == ("a", "resurrected")
    assert f.read_into == (after,) and before not in f.read_into


def test_stores_are_independent(ledger, cap):
    a, b = FakeStore({"x": "1"}), FakeStore()
    write(cap, "a", "x", "1")
    # The same record ID bound in store "a" says nothing about store "b".
    b.items["x"] = "1"
    report = reconcile(ledger, {"a": a, "b": b}, grace=0)
    assert [(f.store, f.kind) for f in report.findings] == [("b", "unrecorded")]
    assert report.unchecked == []
    # A store not passed in isn't checked, and is listed as such.
    write(cap, "c", "y", "2")
    report = reconcile(ledger, {"a": a}, grace=0)
    assert report.ok and report.unchecked == ["c"]


def test_in_flight_write_clears_within_grace(ledger, cap):
    store = FakeStore()
    wid = cap.on_write("k", "v", STORE, "agent_inference")
    store.items["a"] = "v"  # the store call returned; the bind hasn't run yet
    store.on_get = lambda r: cap.bind(wid, "a")  # ...and lands during the grace period
    report = reconcile(ledger, {STORE: store}, grace=0.01)
    assert report.ok and report.cleared == 1
    assert store.gets == 1


def test_in_flight_delete_clears_within_grace(ledger, cap):
    store = FakeStore({"a": "v"})
    write(cap, STORE, "a", "v")
    del store.items["a"]  # deleted; the tombstone hasn't been recorded yet
    store.on_get = lambda r: cap.on_delete(STORE, [r], "forget")
    report = reconcile(ledger, {STORE: store}, grace=0.01)
    assert report.ok and report.cleared == 1


def test_recheck_reports_the_current_kind(ledger, cap):
    store = FakeStore({"a": "v"})
    write(cap, STORE, "a", "v")
    store.items["a"] = "edited"  # modified on the first pass...

    def gone(record_id):
        store.items.pop(record_id, None)  # ...and deleted before the re-check

    store.on_get = gone
    report = reconcile(ledger, {STORE: store}, grace=0)
    assert [(f.record_id, f.kind) for f in report.findings] == [("a", "missing")]


def test_grace_is_waited(ledger, cap):
    store = FakeStore({"x": "1"})
    start = time.monotonic()
    reconcile(ledger, {STORE: store}, grace=0.2)
    assert time.monotonic() - start >= 0.2
    with pytest.raises(ValueError):
        reconcile(ledger, {STORE: store}, grace=-1)


def test_adapter_errors_propagate(ledger, cap):
    class Broken(FakeStore):
        def scan(self):
            raise ConnectionError("store unreachable")

    with pytest.raises(ConnectionError):
        reconcile(ledger, {STORE: Broken()}, grace=0)


def test_reconcile_is_read_only(path, ledger, cap):
    store = FakeStore({"a": "a", "extra": "x"})
    write(cap, STORE, "a", "a")
    write(cap, STORE, "b", "b", parents=["extra"])
    tables = ("writes", "direct_parents", "context_sets", "context_members", "context_chain",
              "bindings", "tombstones", "quarantine")
    conn = sqlite3.connect(str(path))

    def counts():
        return [conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables]

    before = counts()
    report = reconcile(ledger, {STORE: store}, grace=0)
    assert {f.kind for f in report.findings} == {"unrecorded", "missing"}
    assert counts() == before
    assert store.items == {"a": "a", "extra": "x"}
    # The walk connection is clean for the next query.
    report2 = reconcile(ledger, {STORE: store}, grace=0)
    assert report2.findings == report.findings


def test_in_memory_ledger():
    with Ledger(":memory:") as ledger:
        cap = Capture(ledger)
        store = FakeStore({"a": "a", "extra": "x"})
        write(cap, STORE, "a", "a")
        report = reconcile(ledger, {STORE: store}, grace=0)
        assert [(f.record_id, f.kind) for f in report.findings] == [("extra", "unrecorded")]
        # Its temp tables are gone: the ledger connection is shared.
        assert not ledger._conn.execute(
            "SELECT 1 FROM sqlite_temp_master WHERE name LIKE 'rec_%'").fetchone()


def test_reconcile_100k_items_is_linear(path):
    """100k recorded items plus planted findings: every finding is found, and
    the whole pass stays well under a few seconds."""
    rng = random.Random(1)
    n = 100_000
    store = FakeStore()
    with Ledger(path, background_index=False) as ledger:
        batch = []
        for i in range(n):
            parents = [(STORE, f"r{j}", "in_prompt") for j in rng.sample(range(max(1, i)), min(i, 2))]
            batch.append(dict(content=f"c{i}", store=STORE, source="agent_inference",
                              parents=parents, record_id=f"r{i}"))
            store.items[f"r{i}"] = f"c{i}"
            if len(batch) == 5_000:
                ledger.append_batch(batch)
                batch = []
        planted = {f"x{i}" for i in range(50)}
        for r in planted:
            store.items[r] = "outside"
        changed = {f"r{i}" for i in rng.sample(range(n), 50)}
        for r in changed:
            store.items[r] = "edited"
        gone = {f"r{i}" for i in rng.sample(range(n), 50)} - changed
        for r in gone:
            del store.items[r]

        start = time.perf_counter()
        report = reconcile(ledger, {STORE: store}, grace=0)
        elapsed = time.perf_counter() - start
    kinds = report.by_kind()
    assert {f.record_id for f in kinds["unrecorded"]} == planted
    assert {f.record_id for f in kinds["modified"]} == changed
    assert {f.record_id for f in kinds["missing"]} == gone
    assert report.live == {STORE: n}
    print(f"reconcile over {n} items: {elapsed * 1e3:.0f} ms")
    assert elapsed < 5 * SCALE
