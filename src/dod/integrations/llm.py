"""Model clients: confirm, on every call, what the agent put in the prompt.

``instrument_anthropic(client)`` and ``instrument_openai(client)`` wrap a
client instance so each request's text is passed to ``dod.prompt.confirm``
before it is sent: retrieved items whose text is in the request are upgraded
to ``in_prompt`` for the writes that follow. Nothing about the request or
response changes, and nothing is sent anywhere but the model.

- Anthropic (anthropic 1.12.1): ``client.messages.create`` and
  ``client.messages.stream``, sync or async client; the text is ``system``
  plus every string in ``messages`` (text blocks, tool results, tool inputs).
- OpenAI and OpenAI-compatible servers (any SDK whose client has
  ``chat.completions.create``, and ``responses.create`` if present): the
  text is every string in ``messages``, or ``instructions`` plus ``input``.

For any other client, call ``dod.prompt.confirm(text)`` yourself with the
rendered prompt. Outside an ``agent_task_scope`` (or another integration's
scope) confirmation does nothing.
"""

import functools
import inspect
import json
from typing import Any, Iterable, List

from dod import prompt as _prompt

_MARK = "_dod_instrumented"


def request_text(*parts: Any) -> str:
    """Every string in ``parts`` (nested dicts, lists, SDK objects), joined:
    the request as the model will read it, for matching."""
    out: List[str] = []

    def walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, bytes):
            return  # images, files: no text to match
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif hasattr(v, "model_dump"):
            walk(v.model_dump())
        elif isinstance(v, (int, float, bool)):
            return
        else:
            try:
                out.append(json.dumps(v, default=str))
            except Exception:
                pass

    for p in parts:
        walk(p)
    return "\n".join(out)


def _wrap(obj: Any, name: str, text_of) -> None:
    orig = getattr(obj, name, None)
    if orig is None or getattr(orig, _MARK, False):
        return
    if inspect.iscoroutinefunction(orig):
        @functools.wraps(orig)
        async def wrapper(*args, **kwargs):
            _prompt.confirm(text_of(kwargs))
            return await orig(*args, **kwargs)
    else:
        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            _prompt.confirm(text_of(kwargs))
            return orig(*args, **kwargs)
    setattr(wrapper, _MARK, True)
    object.__setattr__(obj, name, wrapper)


def instrument_anthropic(client: Any) -> Any:
    """Confirm each ``messages.create`` / ``messages.stream`` request's text.
    Instrumenting twice is a no-op."""
    messages = getattr(client, "messages", None)
    if messages is None or not callable(getattr(messages, "create", None)):
        raise TypeError(f"{type(client).__name__} has no messages.create()")

    def text_of(kw: dict) -> str:
        return request_text(kw.get("system"), kw.get("messages"))

    for name in ("create", "stream"):
        _wrap(messages, name, text_of)
    return client


def instrument_openai(client: Any) -> Any:
    """Confirm each ``chat.completions.create`` (and ``responses.create``)
    request's text. Instrumenting twice is a no-op."""
    completions = getattr(getattr(client, "chat", None), "completions", None)
    if completions is None or not callable(getattr(completions, "create", None)):
        raise TypeError(f"{type(client).__name__} has no chat.completions.create()")
    _wrap(completions, "create", lambda kw: request_text(kw.get("messages")))
    responses = getattr(client, "responses", None)
    if responses is not None and callable(getattr(responses, "create", None)):
        _wrap(responses, "create",
              lambda kw: request_text(kw.get("instructions"), kw.get("input")))
    return client


__all__: Iterable[str] = ["instrument_anthropic", "instrument_openai", "request_text"]
