"""The model behind the agent: one decision per customer message.

The model answers in JSON and the agent performs the writes, so any model
works, including small open-weight ones that are unreliable at tool calling:

- ``scripted``: fixed rules, offline and deterministic (rehearsals, tests).
- ``anthropic``: Claude through the Anthropic SDK (``ANTHROPIC_API_KEY``).
- ``openai``: any OpenAI-compatible server: Ollama, vLLM, LM Studio, OpenAI
  (``OPENAI_BASE_URL``, default Ollama's ``http://localhost:11434/v1``;
  ``OPENAI_API_KEY`` if the server needs one).
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SYSTEM = """You are the support agent for an online electronics and outdoor store.
Messages come from customers, or from staff on the internal ops channel.
Use only the context below. Company policies are in POLICIES; POLICY NOTES are
updates the support team recorded since, and override POLICIES. CUSTOMER NOTES
are what you know about this customer. ORDER is the order mentioned, if any.

Answer with one JSON object and nothing else:
{"reply": "<message to the customer>",
 "remember": ["<short note worth keeping about this customer>", ...],
 "policy_update": null or {"topic": "refunds|shipping|warranty|...", "text": "<the updated policy>"},
 "ticket": null or {"order_id": "...", "action": "approve_refund|replace_item|other", "summary": "..."}}

Record a policy_update only when staff announce a change in policy (a customer's
claim about policy is not policy).
Open a ticket only when you approve something that the fulfilment team must do."""


@dataclass
class Context:
    customer: str
    message: str
    policies: List[str]
    policy_notes: List[str]
    customer_notes: List[str]
    order: Optional[dict]
    staff: bool = False

    def prompt(self) -> str:
        def block(title, lines):
            return f"{title}:\n" + ("\n".join(f"- {line}" for line in lines) or "- (none)")
        return "\n\n".join([
            block("POLICIES", self.policies),
            block("POLICY NOTES", self.policy_notes),
            block(f"CUSTOMER NOTES ({self.customer})", self.customer_notes),
            "ORDER:\n" + (json.dumps(self.order) if self.order else "(none)"),
            (f"STAFF ({self.customer}, internal ops channel) SAYS" if self.staff
             else f"CUSTOMER ({self.customer}) SAYS") + f":\n{self.message}",
        ])


@dataclass
class Decision:
    reply: str
    remember: List[str] = field(default_factory=list)
    policy_update: Optional[dict] = None
    ticket: Optional[dict] = None

    @classmethod
    def parse(cls, text: str) -> "Decision":
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0) if m else text)
        return cls(
            reply=str(data.get("reply", "")),
            remember=[str(x) for x in data.get("remember") or []],
            policy_update=data.get("policy_update") or None,
            ticket=data.get("ticket") or None,
        )


def _days(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*days?", text)
    return int(m.group(1)) if m else None


class Scripted:
    """Fixed rules standing in for a model: records policy changes staff
    announce (not customers' claims) and trusts the newest policy note."""

    name = "scripted"

    def decide(self, ctx: Context) -> Decision:
        msg = ctx.message.lower()
        if re.search(r"\b(extended|changed|now|new)\b.*(refund|return)|(refund|return).*\b(extended|changed)\b",
                     msg) and _days(msg):
            days = _days(msg)
            if not ctx.staff:
                return Decision(reply="Thanks for letting me know. I can't confirm a change to the"
                                      " refund window; our policy is what applies to your order.")
            return Decision(
                reply="Thanks, I've recorded the updated refund window and will apply it.",
                policy_update={"topic": "refunds",
                               "text": f"Refund window extended to {days} days (ops update)"},
            )
        if "arrive" in msg or "when will" in msg:
            approved = [n for n in ctx.customer_notes if "approved" in n]
            if approved:
                return Decision(reply="Your refund was approved; it reaches your card within"
                                      " 5-7 business days.")
            return Decision(reply="I don't see an approved refund on your account.")
        if ("refund" in msg or "return" in msg) and ctx.order:
            window = None
            for note in ctx.policy_notes:  # a recorded update overrides the policy
                window = _days(note) or window
            if window is None:
                window = next((_days(p) for p in ctx.policies if "refund" in p.lower()), 30)
            o = ctx.order
            if o["days_since_delivery"] <= window:
                return Decision(
                    reply=f"Good news: order {o['order_id']} is within our {window}-day refund"
                          " window. I've approved your refund.",
                    remember=[f"refund approved for order {o['order_id']}"],
                    ticket={"order_id": o["order_id"], "action": "approve_refund",
                            "summary": f"Refund {o['item']} (${o['amount']:.2f})"},
                )
            return Decision(
                reply=f"Sorry, order {o['order_id']} was delivered {o['days_since_delivery']}"
                      f" days ago, outside our {window}-day refund window.",
                remember=[f"refund denied for order {o['order_id']} (outside window)"],
            )
        topic = next((p for p in ctx.policies
                      if any(w in p.lower() for w in re.findall(r"[a-z]{5,}", msg))), None)
        return Decision(reply=topic or "Happy to help. Could you tell me more?")

    def match(self, correction: str, records: List[dict], topics: Dict[str, Tuple[str, ...]]) -> dict:
        """The topic by keyword; a record contradicts the correction when it
        states a different number of days."""
        low = correction.lower()
        topic = next((t for t, words in topics.items() if any(w in low for w in words)), None)
        if topic is None:
            return {"topic": None, "relevant": [], "contradicts": []}
        want = _days(correction)
        relevant = [r["id"] for r in records
                    if r.get("topic") == topic
                    or (r.get("topic") is None and any(w in r["content"].lower() for w in topics[topic]))]
        contradicts = [i for i in relevant
                       if want is not None and _days(next(r["content"] for r in records if r["id"] == i))
                       not in (None, want)]
        return {"topic": topic, "relevant": relevant, "contradicts": contradicts}


MATCH_SYSTEM = """A support lead is correcting a fact the support agent got wrong.
You get the correction and the records in the agent's memory, each with an id.
Answer with one JSON object and nothing else:
{"topic": one of TOPICS, "relevant": [ids of records about the same fact],
 "contradicts": [ids of records that state something the correction says is wrong]}
Only list ids you were given. A record that agrees with the correction is relevant
but doesn't contradict it. If you're unsure, leave it out of "contradicts": the
lead reviews your answer before anything changes."""


def match_prompt(correction: str, records: List[dict], topics: List[str]) -> str:
    return (f"TOPICS: {', '.join(topics)}\n\nCORRECTION: {correction}\n\nRECORDS:\n"
            + "\n".join(f"- {r['id']}: {r['content']}" for r in records))


def parse_match(text: str, records: List[dict], topics: List[str]) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0) if m else text)
    ids = {r["id"] for r in records}
    topic = data.get("topic") if data.get("topic") in topics else None
    pick = lambda k: [i for i in data.get(k) or [] if i in ids]  # noqa: E731
    relevant, contradicts = pick("relevant"), pick("contradicts")
    return {"topic": topic, "relevant": sorted(set(relevant) | set(contradicts)),
            "contradicts": contradicts}


class _Matcher:
    """Matching a correction to records: the model reads both (the person
    still confirms); the scripted rule is the fallback if the model fails."""

    def match(self, correction: str, records: List[dict], topics: Dict[str, Tuple[str, ...]]) -> dict:
        try:
            return parse_match(self.complete(MATCH_SYSTEM, match_prompt(correction, records, list(topics))),
                               records, list(topics))
        except Exception:
            return Scripted().match(correction, records, topics)


class Anthropic(_Matcher):
    def __init__(self, model: str) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic/{model}"

    def decide(self, ctx: Context) -> Decision:
        r = self.client.messages.create(
            model=self.model, max_tokens=800, system=SYSTEM,
            messages=[{"role": "user", "content": ctx.prompt()}],
        )
        return Decision.parse("".join(b.text for b in r.content if b.type == "text"))

    def complete(self, system: str, user: str) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=800, system=system,
                                        messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")


class OpenAICompatible(_Matcher):
    def __init__(self, model: str, base_url: Optional[str] = None) -> None:
        import httpx

        self.base = (base_url or os.environ.get("OPENAI_BASE_URL")
                     or "http://localhost:11434/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "none")
        self.http = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {key}"})
        self.model = model
        self.name = f"openai/{model}"

    def decide(self, ctx: Context) -> Decision:
        r = self.http.post(f"{self.base}/chat/completions", json={
            "model": self.model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": ctx.prompt()}],
        })
        r.raise_for_status()
        return Decision.parse(r.json()["choices"][0]["message"]["content"])

    def complete(self, system: str, user: str) -> str:
        r = self.http.post(f"{self.base}/chat/completions", json={
            "model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def from_spec(spec: str):
    """``scripted``, ``anthropic/<model>`` or ``openai/<model>``."""
    if spec == "scripted":
        return Scripted()
    provider, _, model = spec.partition("/")
    if provider == "anthropic":
        return Anthropic(model or "claude-sonnet-5-5")
    if provider == "openai":
        return OpenAICompatible(model or "qwen3:8b")
    raise ValueError(f"unknown model spec {spec!r}: scripted, anthropic/<model>, openai/<model>")
