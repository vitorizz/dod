"""The context scope shared by every integration.

The context key is ``(agent_id, task_id)``, held in a ``ContextVar``.
Retrievals and writes are linked only under the same key, whichever store
they touch, so a write in one store gets parents retrieved from another.
Enter it with ``agent_task_scope(agent_id, task_id)``.

- asyncio tasks copy the current context, so the scope carries through.
- Plain threads and ``ThreadPoolExecutor`` workers start with an empty
  context: enter the scope inside the worker, or submit the work through
  ``contextvars.copy_context().run``.
- Outside any scope, integrations don't track retrievals, and record writes
  with zero parents. There is no shared fallback key.
"""

import contextvars
import threading
import weakref
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

from dod.capture import Capture

Key = Tuple[str, str]

_key: "contextvars.ContextVar[Optional[Key]]" = contextvars.ContextVar(
    "dod_scope_key", default=None
)

# Every Capture an integration instruments; entering or leaving a scope resets
# its key in each.
_captures: "weakref.WeakSet[Capture]" = weakref.WeakSet()
_captures_lock = threading.Lock()


def current_key() -> Optional[Key]:
    """Return the current ``(agent_id, task_id)``, or None outside any scope."""
    return _key.get()


def register(capture: Capture) -> None:
    """Have scope entry and exit reset their key in ``capture``. Idempotent."""
    with _captures_lock:
        _captures.add(capture)


@contextmanager
def agent_task_scope(agent_id: Any, task_id: Any) -> Iterator[Key]:
    """Make ``(agent_id, task_id)`` the current context key.

    Resets the key's context on entry and on exit, so a finished scope holds
    no memory, and restores the previous key on exit, so scopes nest.
    """
    key = (str(agent_id), str(task_id))
    reset(key)
    token = _key.set(key)
    try:
        yield key
    finally:
        _key.reset(token)
        reset(key)


def reset(key: Key) -> None:
    """Clear ``key``'s context in every registered ``Capture``."""
    with _captures_lock:
        captures = list(_captures)
    for cap in captures:
        cap.reset(key)
