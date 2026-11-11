"""Tracing and fixing agent memory by hand: from a statement, or @commands.

The query functions (``trace``, ``blast_radius``, ``excise``, ``regenerate``,
``Ledger.release``) take record IDs. People don't have those: they have a bad
answer, or a fact they know is wrong ("the refund window is 30 days, not 90").
``Admin`` goes from there:

- ``find(statement)``: read the records in the stores (through a ``Reader``
  per store) and match them against the statement: which are about it, which
  contradict it, which agree. A matcher decides: ``keyword_matcher`` (words
  and numbers, no model) or ``llm_matcher`` (any model). It only proposes; a
  person confirms. Nothing read is written to the ledger.
- ``plan(statement, wrong)``: trace the confirmed wrong records to where they
  came from (an origin that contradicts the statement too, else the record
  itself) and group everything built on that by what to do with it, through
  ``classify``: ``fix`` (beliefs: quarantined), ``cancel`` (pending actions:
  quarantined and passed to ``Actions.cancel``), ``notify`` (outputs already
  sent: kept on record and answered again) and ``record`` (history:
  quarantined, kept).
- ``apply(plan)``: record the correction as a write (store ``admin``, source
  ``user``), let the app write it where the agent will read it
  (``Actions.correct``) and cancel what's pending, then ``regenerate`` from
  each origin: what it reached is quarantined (reversible), and each sent
  output is re-answered (``Actions.rerun``) or, if it answered the origin
  itself, sent the correction (``Actions.notify``). Everything these write
  has the correction in its context, so it traces back to it.
- ``undo()``: release what the last ``apply`` quarantined.
- ``handle("@excise ...")``: the same as commands, for a chat, Slack bot or
  script: ``@trace``, ``@impact``, ``@excise`` (then ``@confirm``/``@cancel``),
  ``@undo``, ``@ledger``, ``@help``. Returns a ``Response``: text to post,
  choices (buttons as data), structured data, and the equivalent library
  calls.

The side effects (writing the correction, cancelling, re-answering) are the
app's: ``Admin`` never touches a store except through ``Actions``.
"""

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from dod.capture import Capture
from dod.ledger import Ledger
from dod.query import blast_radius, trace
from dod.regenerate import RegenerateTask, regenerate
from dod.scope import agent_task_scope, current_key

Item = Tuple[str, str]
GROUPS = ("fix", "cancel", "notify", "record")
ACTOR = "dod-admin"


class Reader(Protocol):
    """Reads one store's records, for matching statements against them."""

    def records(self) -> Iterable[Tuple[str, str]]:
        """``(record_id, text)`` for each record in the store."""


@dataclass
class Record:
    store: str
    record_id: str
    text: str

    @property
    def item(self) -> Item:
        return (self.store, self.record_id)


@dataclass
class Match:
    """What a matcher found for a statement. ``about`` holds every record
    on the same fact, including those in ``contradicts`` and ``agrees``."""

    about: List[Item] = field(default_factory=list)
    contradicts: List[Item] = field(default_factory=list)
    agrees: List[Item] = field(default_factory=list)


Matcher = Callable[[str, List[Record]], Match]

_STOP = set("""about after again also been before being could does from have here into
just like more most only other over said same should since some still such than that
their them then there these they this those through very want were what when where
which while will with would your actually please today""".split())


def _words(text: str) -> set:
    """Significant words, crudely singular ("refunds" is "refund")."""
    out = set()
    for w in re.findall(r"[a-z]{4,}", text.lower()):
        if w not in _STOP:
            out.add(w[:-1] if w.endswith("s") and not w.endswith("ss") else w)
    return out


def _numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _asserted(statement: str) -> set:
    """Numbers the statement asserts: not those it negates ("not 90")."""
    return _numbers(re.sub(r"\b(?:not|instead of|rather than)\s+\d+(?:\.\d+)?", " ",
                           statement, flags=re.I))


def keyword_matcher(statement: str, records: List[Record]) -> Match:
    """No model: a record is about the statement when it shares two of its
    words (or all, if it has fewer); it contradicts it when it states numbers
    and none is one the statement asserts, and agrees when one is."""
    words, nums = _words(statement), _asserted(statement)
    need = min(2, len(words))
    out = Match()
    if not need:
        return out
    for r in records:
        if len(words & _words(r.text)) < need:
            continue
        out.about.append(r.item)
        rn = _numbers(r.text)
        if nums and rn:
            (out.agrees if rn & nums else out.contradicts).append(r.item)
    return out


MATCH_SYSTEM = """Someone is correcting a fact an AI agent got wrong, or asking what a
fact affected. You get their statement and records from the agent's memory, each
with an id. Answer with one JSON object and nothing else:
{"about": [ids of records about the same fact],
 "contradicts": [ids of records that conflict with the statement],
 "agrees": [ids of records that say the same as the statement]}
Only use ids you were given. If you're unsure, leave an id out of "contradicts"
and "agrees": a person reviews your answer before anything changes."""


def llm_matcher(complete: Callable[[str, str], str],
                fallback: Optional[Matcher] = keyword_matcher) -> Matcher:
    """A matcher that asks a model: ``complete(system, user)`` returns its
    text. Ids it invents are dropped; if the call or the parse fails,
    ``fallback`` answers instead (``None`` to raise)."""

    def match(statement: str, records: List[Record]) -> Match:
        ids = {f"r{i}": r.item for i, r in enumerate(records, 1)}
        user = f"STATEMENT: {statement}\n\nRECORDS:\n" + "\n".join(
            f"- r{i}: {r.text}" for i, r in enumerate(records, 1))
        try:
            text = complete(MATCH_SYSTEM, user)
            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0) if m else text)
            pick = lambda k: [ids[i] for i in data.get(k) or [] if i in ids]  # noqa: E731
            contradicts, agrees = pick("contradicts"), pick("agrees")
            agrees = [i for i in agrees if i not in contradicts]
            about = list(dict.fromkeys(pick("about") + contradicts + agrees))
            return Match(about, contradicts, agrees)
        except Exception:
            if fallback is None:
                raise
            return fallback(statement, records)

    return match


@dataclass
class Actions:
    """The app's side of a fix. Each runs with the correction in context, so
    what it writes through capture traces back to the correction."""

    correct: Optional[Callable[[str, "Plan"], None]] = None
    """Write the corrected fact where the agent will read it."""
    cancel: Optional[Callable[[Item], None]] = None
    """Cancel a pending action (a ticket, a scheduled job) in its system."""
    rerun: Optional[Callable[[Item], Optional[Item]]] = None
    """Answer again what a sent output answered; return the new output's item."""
    notify: Optional[Callable[[Item, str], Optional[Item]]] = None
    """For an output that answered the origin itself (re-running would repeat
    the mistake): send the correction (the ``str``); return its item."""


@dataclass
class Plan:
    id: str
    statement: str
    wrong: List[Item]
    """What the person confirmed is wrong."""
    roots: List[Item]
    """Where it came from: what ``apply`` regenerates from."""
    groups: Dict[str, List[Item]]
    """Every reached live item, by what will happen to it (``GROUPS``)."""
    untouched: int
    """Live items outside the plan."""

    @property
    def affected(self) -> int:
        return sum(len(v) for v in self.groups.values())


@dataclass
class Applied:
    plan: Plan
    correction: Item
    cancelled: List[Item]
    replaced: Dict[Item, Item]
    quarantined: List[Item]
    failed: List[Tuple[Item, str]]


@dataclass
class Response:
    """What ``handle`` answers: text to post, ``choices`` as buttons
    (``(label, command)``: send ``command`` back to ``handle`` when picked),
    structured ``data``, and the equivalent library calls."""

    text: str
    choices: List[Tuple[str, str]] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    code: List[str] = field(default_factory=list)


HELP = """Commands:
@trace <store> <record_id>   where a record came from
@impact <fact>               what records stating it affected (or: @impact <store> <record_id>)
@excise <the right fact>     find what contradicts it and plan the fix; then @confirm or @cancel
@undo                        release what the last fix quarantined
@ledger                      what the ledger holds, and its health"""


class Admin:
    """See the module docs. ``readers`` maps store name to a ``Reader``;
    ``searchable`` limits which of them ``find`` searches (default: all; the
    rest are still read to tell an origin's content). ``classify(item)``
    returns one of ``GROUPS`` (default ``fix``). ``label(item)`` names a
    record in responses. ``capture`` records the correction; ``apply`` needs
    it."""

    def __init__(
        self,
        ledger: Ledger,
        readers: Mapping[str, Reader],
        capture: Optional[Capture] = None,
        matcher: Matcher = keyword_matcher,
        classify: Optional[Callable[[Item], str]] = None,
        actions: Optional[Actions] = None,
        searchable: Optional[Sequence[str]] = None,
        label: Optional[Callable[[Item], str]] = None,
        store: str = "admin",
    ):
        self.ledger, self.readers, self.capture = ledger, dict(readers), capture
        self.matcher = matcher
        self.classify = classify or (lambda item: "fix")
        self.actions = actions or Actions()
        self.searchable = list(searchable) if searchable is not None else list(self.readers)
        missing = [s for s in self.searchable if s not in self.readers]
        if missing:
            raise ValueError(f"searchable stores without a reader: {missing}")
        self.label = label or (lambda item: f"{item[0]}:{item[1]}")
        self.store = store
        # In memory only: a restart forgets pending plans and what @undo would
        # release (the quarantine rows stay in the ledger: dod release works).
        self.pending: Dict[str, Plan] = {}
        self.history: List[Applied] = []

    # -- reading -----------------------------------------------------------

    def _live(self) -> set:
        """Every item with a binding, minus quarantined ones."""
        with self.ledger._lock:
            rows = self.ledger._conn.execute(
                "SELECT DISTINCT store, record_id FROM bindings").fetchall()
        return set(rows) - self.ledger.quarantined()

    def records(self, stores: Optional[Iterable[str]] = None) -> List[Record]:
        """Live records of ``stores`` (default: the searchable ones)."""
        q = self.ledger.quarantined()
        out = []
        for s in (self.searchable if stores is None else stores):
            for rid, text in self.readers[s].records():
                if (s, rid) not in q:
                    out.append(Record(s, rid, text))
        return out

    def find(self, statement: str) -> Tuple[Match, List[Record]]:
        """Match ``statement`` against the searchable records."""
        recs = self.records()
        return self.matcher(statement, recs), recs

    def _items_of(self, write_ids: Iterable[str]) -> List[Item]:
        ids = list(write_ids)
        if not ids:
            return []
        with self.ledger._lock:
            rows = self.ledger._conn.execute(
                f"SELECT DISTINCT store, record_id FROM bindings WHERE write_id IN"
                f" ({','.join('?' * len(ids))})", ids).fetchall()
        return [tuple(r) for r in rows]

    def resolve(self, ref: str) -> Item:
        """``"<store> <record_id>"``, or a record ID bound in one store only."""
        store, _, rid = ref.strip().partition(" ")
        if rid:
            with self.ledger._lock:
                hit = self.ledger._conn.execute(
                    "SELECT 1 FROM bindings WHERE store = ? AND record_id = ? LIMIT 1",
                    (store, rid.strip())).fetchone()
            if not hit:
                raise KeyError(f"no record {rid.strip()!r} in store {store!r}")
            return (store, rid.strip())
        with self.ledger._lock:
            rows = self.ledger._conn.execute(
                "SELECT DISTINCT store FROM bindings WHERE record_id = ?", (store,)).fetchall()
        if len(rows) != 1:
            raise KeyError(f"{ref!r} is in {len(rows)} stores; name the store too"
                           if rows else f"no record {ref!r} in the ledger")
        return (rows[0][0], store)

    # -- planning ----------------------------------------------------------

    def origins(self, statement: str, item: Item) -> List[Item]:
        """Where ``item`` came from, as far as it conflicts with
        ``statement``: its trace's roots that contradict it too, else the
        item itself."""
        tr = trace(self.ledger, *item)
        roots = [i for i in self._items_of(w.write_id for w in tr.roots) if i != item]
        text = {}
        for s in {r[0] for r in roots} & set(self.readers):
            text.update({(s, rid): t for rid, t in self.readers[s].records()})
        recs = [Record(s, rid, text[(s, rid)]) for s, rid in roots if (s, rid) in text]
        bad = self.matcher(statement, recs).contradicts if recs else []
        return bad or [item]

    def plan(self, statement: str, wrong: Optional[List[Item]] = None) -> Plan:
        """The fix for ``wrong`` (default: whatever contradicts the
        statement). Kept in ``pending`` until applied or dropped."""
        if wrong is None:
            wrong = self.find(statement)[0].contradicts
        wrong = list(dict.fromkeys(tuple(i) for i in wrong))
        if not wrong:
            raise ValueError("nothing to fix: no record contradicts the statement")
        for store, rid in wrong:
            self.resolve(f"{store} {rid}")  # KeyError for a record the ledger doesn't know
        roots: List[Item] = []
        for item in wrong:
            for r in self.origins(statement, item):
                if r not in roots:
                    roots.append(r)
        live = self._live()
        reached = set(wrong) | set(roots)
        for r in roots:
            b = blast_radius(self.ledger, *r)
            reached |= {i for its in b.bindings.values() for i in its}
        groups: Dict[str, List[Item]] = {g: [] for g in GROUPS}
        for item in sorted(reached & live):
            g = self.classify(item)
            groups[g if g in groups else "fix"].append(item)
        p = Plan(uuid.uuid4().hex[:8], statement, wrong, roots, groups,
                 untouched=len(live - reached))
        self.pending[p.id] = p
        return p

    # -- applying ----------------------------------------------------------

    def apply(self, plan, user: Optional[str] = None) -> Applied:
        """Apply a plan (or its id); see the module docs."""
        if isinstance(plan, str):
            plan = self.pending[plan]
        if self.capture is None:
            raise ValueError("Admin needs a capture to record the correction")
        cap, a = self.capture, self.actions
        cid = f"correction:{plan.id}"
        correction = (self.store, cid)
        with agent_task_scope(user or ACTOR, cid):
            key = current_key()
            wid = cap.on_write(key, plan.statement, self.store, "user")
            cap.bind(wid, cid)
            cap.on_retrieve(key, [correction], "in_prompt")
            if a.correct:
                a.correct(plan.statement, plan)
            cancelled = []
            for item in plan.groups["cancel"]:
                if a.cancel:
                    a.cancel(item)
                    cancelled.append(item)
        notify = set(plan.groups["notify"])

        def rewrite(task: RegenerateTask) -> Optional[Item]:
            if task.item not in notify:
                return None
            cap.on_retrieve(current_key(), [correction], "in_prompt")
            if any(p.status == "bad" for p in task.parents):  # answered the origin itself
                return a.notify(task.item, plan.statement) if a.notify else None
            return a.rerun(task.item) if a.rerun else None

        replaced: Dict[Item, Item] = {}
        quarantined: List[Item] = []
        failed: List[Tuple[Item, str]] = []
        for root in plan.roots:
            r = regenerate(self.ledger, *root, rewrite, agent_id=ACTOR)
            replaced.update(r.replaced)
            quarantined += [(s, rid) for s, rids in r.done.items() for rid in rids]
            failed += [((f.store, f.record_id), f.reason) for f in r.failed]
        out = Applied(plan, correction, cancelled, replaced, quarantined, failed)
        self.pending.pop(plan.id, None)  # only now: if an action raised, it can be retried
        self.history.append(out)
        return out

    def undo(self, user: Optional[str] = None) -> Optional[Applied]:
        """Release what the last ``apply`` quarantined (sent corrections and
        cancellations stay: they happened). ``None`` if nothing to undo."""
        if not self.history:
            return None
        last = self.history.pop()
        if last.quarantined:
            self.ledger.release(last.quarantined, reason=f"undo {last.correction[1]}",
                                actor=user or ACTOR)
        return last

    def status(self) -> dict:
        """Counts per store, and the quarantined total."""
        with self.ledger._lock:
            c = self.ledger._conn
            writes = dict(c.execute("SELECT store, count(*) FROM writes GROUP BY store").fetchall())
            items = dict(c.execute("SELECT store, count(DISTINCT record_id) FROM bindings"
                                   " GROUP BY store").fetchall())
        return {"writes": writes, "items": items, "quarantined": len(self.ledger.quarantined())}

    # -- commands ----------------------------------------------------------

    def handle(self, text: str, user: Optional[str] = None) -> Response:
        """Run one ``@command``; see ``HELP``."""
        m = re.match(r"\s*@(\w+)\b[:,]?\s*(.*)$", text, re.S)
        if not m:
            return Response("Commands start with @.\n" + HELP)
        cmd, arg = m.group(1).lower(), m.group(2).strip()
        try:
            fn = {"trace": self._trace, "impact": self._impact, "excise": self._excise,
                  "confirm": self._confirm, "cancel": self._cancel, "undo": self._undo,
                  "ledger": self._ledger, "status": self._ledger, "help": self._help}.get(cmd)
            if fn is None:
                return Response(f"Unknown command @{cmd}.\n" + HELP)
            return fn(arg, user)
        except (KeyError, ValueError) as exc:
            return Response(str(exc.args[0] if exc.args else exc))

    def _help(self, arg, user) -> Response:
        return Response(HELP)

    def _trace(self, arg, user) -> Response:
        if not arg:
            return Response("Which record? @trace <store> <record_id>")
        item = self.resolve(arg)
        tr = trace(self.ledger, *item)
        roots = [i for i in self._items_of(w.write_id for w in tr.roots) if i != item]
        names = self._names(roots) or "nothing: it has no recorded inputs"
        return Response(f"{self.label(item)} came from {names} "
                        f"({len(tr.ancestors)} write(s) upstream).",
                        data={"item": item, "origins": roots,
                              "ancestors": len(tr.ancestors)},
                        code=[f"dod trace {item[0]} {item[1]}",
                              f"dod.query.trace(ledger, {item[0]!r}, {item[1]!r})"])

    def _names(self, items: List[Item], most: int = 5) -> str:
        names = [self.label(i) for i in items]
        more = f" and {len(names) - most} more" if len(names) > most else ""
        return ", ".join(names[:most]) + more

    def _impact(self, arg, user) -> Response:
        if not arg:
            return Response("Of what? @impact <fact> or @impact <store> <record_id>")
        try:
            sources = [self.resolve(arg)]
        except KeyError:
            match, _ = self.find(arg)
            sources = match.agrees
            if not sources:
                return Response("Nothing in memory states that." + (
                    f" {len(match.about)} record(s) are about it but say otherwise;"
                    " to correct them: @excise <the right fact>" if match.about else ""))
        reached: Dict[Item, None] = {}
        for s in sources:
            b = blast_radius(self.ledger, *s)
            for its in b.bindings.values():
                for i in its:
                    if i not in sources:
                        reached[i] = None
        per: Dict[str, int] = {}
        for s, _ in reached:
            per[s] = per.get(s, 0) + 1
        what = ", ".join(f"{n} in {s}" for s, n in sorted(per.items())) or "nothing else"
        return Response(
            f"{len(sources)} record(s) state it ({self._names(sources)});"
            f" built on them: {what}.",
            data={"sources": sources, "reached": list(reached), "per_store": per},
            code=[f"dod blast-radius {s[0]} {s[1]}" for s in sources])

    def _excise(self, arg, user) -> Response:
        if not arg:
            return Response("What's the right fact? e.g. @excise the refund window is 30 days")
        match, recs = self.find(arg)
        if not match.contradicts:
            return Response("Nothing in memory contradicts that; nothing to fix."
                            if match.about else "Nothing in memory is about that.",
                            data={"match": match})
        p = self.plan(arg, match.contradicts)
        return self._plan_response(p, match, recs)

    def _plan_response(self, p: Plan, match: Match, recs: List[Record]) -> Response:
        lines = [f"Contradicts you: {', '.join(self.label(i) for i in p.wrong)}.",
                 f"Where it came from: {', '.join(self.label(i) for i in p.roots)}."]
        verbs = {"fix": "quarantine", "cancel": "cancel", "notify": "answer again",
                 "record": "keep on record (quarantined)"}
        for g in GROUPS:
            if p.groups[g]:
                lines.append(f"{verbs[g].capitalize()}: "
                             + ", ".join(self.label(i) for i in p.groups[g]))
        lines.append(f"Untouched: {p.untouched} record(s). Nothing is deleted; @undo reverses it.")
        return Response("\n".join(lines),
                        choices=[("Fix it", f"@confirm {p.id}"), ("Cancel", f"@cancel {p.id}")],
                        data={"plan": p, "match": match,
                              "records": {r.item: r.text for r in recs if r.item in match.about}},
                        code=[f"plan = admin.plan({p.statement!r})", "admin.apply(plan)"]
                        + [f"# = dod.regenerate(ledger, {r[0]!r}, {r[1]!r}, rewrite)"
                           for r in p.roots])

    def _confirm(self, arg, user) -> Response:
        if arg not in self.pending:
            return Response("No pending fix with that id (it may have been applied or cancelled).")
        a = self.apply(arg, user=user)
        return Response(
            f"Fixed: {len(a.quarantined)} record(s) quarantined, {len(a.cancelled)} cancelled,"
            f" {len(a.replaced)} answered again." + (f" {len(a.failed)} failed." if a.failed else ""),
            data={"applied": a}, code=[f"admin.apply({a.plan.id!r})"])

    def _cancel(self, arg, user) -> Response:
        return Response("Cancelled; nothing changed." if self.pending.pop(arg, None)
                        else "No pending fix with that id.")

    def _undo(self, arg, user) -> Response:
        a = self.undo(user=user)
        if a is None:
            return Response("Nothing to undo.")
        return Response(f"Released {len(a.quarantined)} record(s) the fix had quarantined."
                        " Corrections already sent and cancellations stay.",
                        data={"applied": a},
                        code=["admin.undo()", "dod release <store> <record_id> ..."])

    def _ledger(self, arg, user) -> Response:
        s = self.status()
        per = ", ".join(f"{st} {n}" for st, n in sorted(s["items"].items()))
        return Response(f"{sum(s['writes'].values())} writes, {sum(s['items'].values())} records"
                        f" across {len(s['items'])} store(s) ({per}); {s['quarantined']} quarantined.",
                        data=s, code=["dod stats", "dod doctor"])
