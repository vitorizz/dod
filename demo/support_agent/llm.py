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
from typing import List, Optional

SYSTEM = """You are the support agent for an online electronics and outdoor store.
Use only the context below. Company policies are in POLICIES; POLICY NOTES are
updates the support team recorded since. CUSTOMER NOTES are what you know about
this customer. ORDER is the order the customer mentions, if any.

Answer with one JSON object and nothing else:
{"reply": "<message to the customer>",
 "remember": ["<short note worth keeping about this customer>", ...],
 "policy_update": null or {"topic": "refunds|shipping|warranty|...", "text": "<the updated policy>"},
 "ticket": null or {"order_id": "...", "action": "approve_refund|replace_item|other", "summary": "..."}}

Record a policy_update only when the customer reports a change in policy.
Open a ticket only when you approve something that the fulfilment team must do."""


@dataclass
class Context:
    customer: str
    message: str
    policies: List[str]
    policy_notes: List[str]
    customer_notes: List[str]
    order: Optional[dict]

    def prompt(self) -> str:
        def block(title, lines):
            return f"{title}:\n" + ("\n".join(f"- {line}" for line in lines) or "- (none)")
        return "\n\n".join([
            block("POLICIES", self.policies),
            block("POLICY NOTES", self.policy_notes),
            block(f"CUSTOMER NOTES ({self.customer})", self.customer_notes),
            "ORDER:\n" + (json.dumps(self.order) if self.order else "(none)"),
            f"CUSTOMER ({self.customer}) SAYS:\n{self.message}",
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
    """Fixed rules standing in for a model: trusts the newest policy note."""

    name = "scripted"

    def decide(self, ctx: Context) -> Decision:
        msg = ctx.message.lower()
        if re.search(r"\b(extended|changed|now|new)\b.*(refund|return)|(refund|return).*\b(extended|changed)\b",
                     msg) and _days(msg):
            days = _days(msg)
            return Decision(
                reply="Thanks for letting us know. I've noted the updated refund window.",
                remember=[f"{ctx.customer} reported that the refund window is {days} days"],
                policy_update={"topic": "refunds",
                               "text": f"Refund window extended to {days} days (reported by"
                                       f" {ctx.customer})"},
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


class Anthropic:
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


class OpenAICompatible:
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
