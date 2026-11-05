"""Confirm what reached the model: upgrade retrieved items to ``in_prompt``.

Integrations tag a retrieval ``retrieved_in_scope``: they see what the agent
read, not what it put in the prompt. ``confirm(prompt)`` closes that gap.
Call it with the text sent to the model, inside the agent's scope; every
item retrieved in the scope whose text appears in the prompt is upgraded to
``in_prompt`` for the writes that follow::

    from dod import prompt

    prompt.confirm(rendered_prompt)          # any model, any client
    response = my_llm(rendered_prompt)

or wrap a client once (``dod.integrations.llm.instrument_anthropic`` /
``instrument_openai``) and every call is confirmed for you.

Matching is by text: the start of each retrieved string (whitespace
normalized, see ``dod.capture.probes``) must appear in the prompt. A
paraphrased or summarized item isn't confirmed and stays
``retrieved_in_scope``: still a parent. Item text is held in memory only,
while the scope lives; the ledger stays content-free.
"""

from typing import Iterable, List, Optional

from dod import scope as _scope
from dod.capture import Capture, Item


def confirm(prompt: str, captures: Optional[Iterable[Capture]] = None,
            key=None) -> List[Item]:
    """Upgrade the current scope's retrieved items that appear in ``prompt``.

    ``captures`` defaults to every ``Capture`` an integration was
    instrumented with; ``key`` to the current scope's. Returns the items
    upgraded. Outside any scope, does nothing.
    """
    key = key if key is not None else _scope.current_key()
    if key is None:
        return []
    if captures is None:
        with _scope._captures_lock:
            captures = list(_scope._captures)
    out: List[Item] = []
    for cap in captures:
        out += cap.confirm(key, prompt)
    return out
