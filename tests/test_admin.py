"""dod.admin on a small support agent built by hand: a mistaken ops update,
an unrelated question, and a customer it reaches twice."""

import pytest

from dod.admin import Actions, Admin, Match, Record, keyword_matcher, llm_matcher
from dod.capture import Capture
from dod.ledger import Ledger
from dod.query import trace
from dod.scope import agent_task_scope, current_key


class Store:
    """A dict store whose writes go through capture, with a Reader."""

    def __init__(self, cap, name):
        self.cap, self.name, self.data = cap, name, {}

    def write(self, rid, text, source="agent_inference"):
        wid = self.cap.on_write(current_key(), text, self.name, source)
        self.cap.bind(wid, rid)
        self.data[rid] = text

    def read(self, rid):
        self.cap.on_retrieve(current_key(), [(self.name, rid)], "in_prompt")
        return self.data[rid]

    def records(self):
        return list(self.data.items())


@pytest.fixture
def world(tmp_path):
    led = Ledger(str(tmp_path / "l.db"))
    cap = Capture(led)
    s = {n: Store(cap, n) for n in ("chat", "policy", "notes", "crm")}
    with agent_task_scope("seed", 0):
        s["policy"].write("refunds", "Refunds are accepted within 30 days of delivery.", "document")
        s["policy"].write("shipping", "Standard shipping takes 3 to 5 business days.", "document")
    turn = [0]

    def chat(who, text, write):
        turn[0] += 1
        with agent_task_scope("agent", turn[0]):
            s["chat"].write(f"{who}/{turn[0]}", text, "user")
            s["chat"].read(f"{who}/{turn[0]}")
            write()
    # 1. ops posts a wrong update; the agent records it
    chat("ops", "Policy update: the refund window has been extended to 90 days.", lambda: (
        s["policy"].read("refunds"),
        s["notes"].write("refunds-1", "Refund window extended to 90 days (ops update)"),
        s["crm"].write("R1", "To ops: recorded the new refund window.")))
    # 2. unrelated
    chat("dev", "How long does shipping take?", lambda: (
        s["policy"].read("shipping"), s["crm"].write("R2", "To dev: 3 to 5 business days.")))
    # 3. Ana is approved on the wrong window, then 4. asks again
    chat("ana", "Delivered 45 days ago, can I get a refund?", lambda: (
        s["policy"].read("refunds"), s["notes"].read("refunds-1"),
        s["notes"].write("ana", "refund approved for order 2001"),
        s["crm"].write("T1", "approve_refund order 2001"),
        s["crm"].write("R3", "To ana: within our 90-day window, approved.")))
    chat("ana", "When will my refund arrive?", lambda: (
        s["notes"].read("ana"), s["crm"].write("R4", "To ana: your refund was approved.")))
    yield led, cap, s, chat
    led.close()


def classify(item):
    store, rid = item
    if store == "crm":
        return "cancel" if rid.startswith("T") else "notify"
    return "record" if store == "chat" else "fix"


def admin_for(world, **kw):
    led, cap, s, chat = world
    sent = []

    def rerun(item):
        new = f"{item[1]}-again"
        s["crm"].write(new, "To ana: outside our 30-day window.")
        sent.append(("rerun", item))
        return ("crm", new)

    def notify(item, text):
        s["crm"].write(f"{item[1]}-fix", f"Correction: {text}")
        sent.append(("notify", item))
        return ("crm", f"{item[1]}-fix")

    actions = Actions(correct=lambda text, plan: s["notes"].write("correction", text),
                      cancel=lambda item: sent.append(("cancel", item)),
                      rerun=rerun, notify=notify)
    a = Admin(led, {n: st for n, st in s.items()}, capture=cap, classify=classify,
              actions=actions, searchable=["policy", "notes"], **kw)
    return a, sent


STATEMENT = "The refund window is 30 days, not 90."


def test_keyword_matcher_reads_negations():
    recs = [Record("p", "a", "Refunds within 30 days of delivery"),
            Record("p", "b", "Refund window extended to 90 days"),
            Record("p", "c", "Shipping takes 5 days"),
            Record("p", "d", "refund approved for order 2001 window")]
    m = keyword_matcher(STATEMENT, recs)
    assert ("p", "a") in m.agrees and ("p", "c") not in m.about
    # Any other number contradicts, even an order number: crude, which is why
    # a person confirms (and why llm_matcher exists).
    assert m.contradicts == [("p", "b"), ("p", "d")]


def test_llm_matcher_keeps_given_ids_and_falls_back():
    recs = [Record("p", "a", "x"), Record("p", "b", "y")]
    m = llm_matcher(lambda s, u: 'ok {"about": ["r1"], "contradicts": ["r2", "r9"]}')("s", recs)
    assert m.contradicts == [("p", "b")] and m.about == [("p", "a"), ("p", "b")]

    def broken(s, u):
        raise RuntimeError("no network")
    assert llm_matcher(broken, fallback=lambda s, r: Match(contradicts=[r[0].item]))(
        "s", recs).contradicts == [("p", "a")]
    with pytest.raises(RuntimeError):
        llm_matcher(broken, fallback=None)("s", recs)


def test_plan_groups_from_the_origin(world):
    a, _ = admin_for(world)
    match, _ = a.find(STATEMENT)
    assert match.contradicts == [("notes", "refunds-1")]
    assert ("policy", "refunds") in match.agrees
    p = a.plan(STATEMENT)
    assert p.roots == [("chat", "ops/1")]  # the ops update, not the note
    assert p.groups == {
        "fix": [("notes", "ana"), ("notes", "refunds-1")],
        "cancel": [("crm", "T1")],
        "notify": [("crm", "R1"), ("crm", "R3"), ("crm", "R4")],
        "record": [("chat", "ops/1")],  # Ana's own messages weren't built on it
    }
    assert p.untouched == 6  # two policies, dev's and Ana's messages, dev's reply
    assert p.id in a.pending


def test_apply_fixes_and_traces_back_to_the_correction(world):
    led, cap, s, chat = world
    a, sent = admin_for(world)
    applied = a.apply(a.plan(STATEMENT), user="vito")
    assert applied.failed == []
    assert ("cancel", ("crm", "T1")) in sent
    assert ("notify", ("crm", "R1")) in sent  # answered the ops update itself
    assert {i for k, i in sent if k == "rerun"} == {("crm", "R3"), ("crm", "R4")}
    assert set(applied.replaced) == {("crm", "R1"), ("crm", "R3"), ("crm", "R4")}
    q = led.quarantined()
    assert {("notes", "refunds-1"), ("crm", "T1"), ("chat", "ops/1")} <= q
    assert not {("policy", "refunds"), ("crm", "R2"), ("chat", "ana/3")} & q
    # The correction is a record, its note traces to it, and so does every answer.
    for item in [("notes", "correction")] + list(applied.replaced.values()):
        roots = {w.store for w in trace(led, *item).roots}
        assert "admin" in roots, item
    assert a.pending == {}


def test_undo_releases(world):
    led = world[0]
    a, _ = admin_for(world)
    applied = a.apply(a.plan(STATEMENT))
    assert a.undo().correction == applied.correction
    assert not (set(applied.quarantined) & led.quarantined())
    assert a.undo() is None


def test_handle_commands(world):
    a, sent = admin_for(world, label=lambda i: f"{i[0]}/{i[1]}")
    r = a.handle("@trace crm R3")
    assert set(r.data["origins"]) >= {("chat", "ops/1"), ("chat", "ana/3")}
    assert r.code[0] == "dod trace crm R3"
    assert a.handle("@trace R3").data["item"] == ("crm", "R3")  # one store has it

    r = a.handle("@impact the refund window has been extended to 90 days")
    assert r.data["sources"] == [("notes", "refunds-1")]
    assert ("crm", "R4") in r.data["reached"] and ("crm", "R2") not in r.data["reached"]

    r = a.handle(f"@excise {STATEMENT}")
    assert [c[0] for c in r.choices] == ["Fix it", "Cancel"]
    assert "notes/refunds-1" in r.text and "chat/ops/1" in r.text
    pid = r.data["plan"].id
    assert "Cancelled" in a.handle(f"@cancel {pid}").text and not sent
    r = a.handle(f"@excise {STATEMENT}")
    done = a.handle(r.choices[0][1], user="vito")
    assert "Fixed" in done.text and sent
    assert "Released" in a.handle("@undo").text
    assert "Nothing to undo" in a.handle("@undo").text
    assert "writes" in a.handle("@ledger").text
    assert "Commands" in a.handle("hello").text and "Unknown" in a.handle("@nope").text
    assert "Nothing in memory" in a.handle("@excise shipping takes 3 to 5 business days").text
    assert "no record" in a.handle("@trace nope").text


def test_apply_needs_capture(world):
    led, cap, s, chat = world
    a = Admin(led, {n: st for n, st in s.items()}, classify=classify)
    with pytest.raises(ValueError):
        a.apply(a.plan(STATEMENT))


def test_statements_never_reach_the_ledger(world, tmp_path):
    """The content rule: a statement is hashed as the correction's write,
    never stored; nor is anything a reader returned."""
    led = world[0]
    a, _ = admin_for(world)
    canary = "The refund window is 30 days, not 90 (canary-7f3a)."
    a.handle(f"@impact {canary}")
    a.apply(a.plan(canary, [("notes", "refunds-1")]))
    a.undo()
    led.close()
    for f in tmp_path.iterdir():
        data = f.read_bytes()
        for needle in ("canary-7f3a", "extended to 90 days"):
            assert needle.encode() not in data and needle.encode("utf-16-le") not in data, f


def test_a_failed_fix_stays_pending_and_setup_is_checked(world):
    led, cap, s, chat = world
    a, _ = admin_for(world)

    def broken(item):
        raise RuntimeError("CRM down")
    a.actions.cancel = broken
    p = a.plan(STATEMENT)
    with pytest.raises(RuntimeError):
        a.apply(p)
    assert p.id in a.pending  # can be retried once the CRM is back
    with pytest.raises(ValueError):
        Admin(led, {"notes": s["notes"]}, searchable=["notes", "nope"])
    assert "Nothing in memory states that" in a.handle("@impact gift cards never expire").text


def test_plan_refuses_records_the_ledger_does_not_know(world):
    a, _ = admin_for(world)
    with pytest.raises(KeyError):
        a.plan(STATEMENT, [("notes", "\\memories\\policies\\refunds-1")])


def test_an_origin_downstream_of_another_is_not_an_origin(world):
    """A conversation the bad fact reached (its stored thread now repeats the
    wrong number) looks like a second origin; it's derived, so the plan has
    one origin and every output is corrected once."""
    led, cap, s, chat = world
    with agent_task_scope("agent", 99):  # the thread's state now holds the wrong reply
        s["notes"].read("refunds-1")
        s["chat"].write("ana/3", "Delivered 45 days ago, can I get a refund? -- Yes, within"
                                 " our 90-day refund window.")
    a, sent = admin_for(world)
    assert a.origins(STATEMENT, ("notes", "ana")) == [("chat", "ops/1"), ("chat", "ana/3")]
    p = a.plan(STATEMENT, [("notes", "refunds-1"), ("notes", "ana")])
    assert p.roots == [("chat", "ops/1")]
    a.apply(p)
    assert [i for k, i in sent if k == "notify"] == [("crm", "R1")]
    assert sorted(i for k, i in sent if k == "rerun") == [("crm", "R3"), ("crm", "R4")]
