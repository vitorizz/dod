"""CrewAI integration: capture lineage from a ``crewai.Memory`` (crewai 1.15.22).

``instrument(memory, capture)`` wraps the instance's ``recall`` and
``remember``. Both run synchronously in the caller and see exact record IDs
(``MemoryMatch.record.id`` and the returned ``MemoryRecord.id``). Memory
events are not used: save events carry no record ID, and the event bus runs
sync handlers in a thread pool, so they can arrive out of order.

Scoping
    The context key is ``(agent_id, task_id)``, held in a ``ContextVar``.
    Recalls and writes are linked only under the same key. Enter it with
    ``agent_task_scope(agent_id, task_id)``.

    - asyncio tasks copy the current context, so the scope carries through.
    - Plain threads and ``ThreadPoolExecutor`` workers start with an empty
      context: enter the scope inside the worker, or submit the work through
      ``contextvars.copy_context().run``.
    - Outside any scope, recalls are not tracked, and a write is recorded with
      zero parents and a one-time warning. There is no shared fallback key.

    In a crew run, call ``install_task_scopes()`` once. It registers crewai
    ``PRE_STEP``/``POST_STEP`` interception hooks, which ``Task._execute_core``
    and ``Task._aexecute_core`` dispatch synchronously in the task's own
    thread or coroutine, around ``agent.execute_task``. A ``TaskStartedEvent``
    handler can't enter the scope: the event bus runs it in a thread pool on a
    copy of the context, so the key never reaches the agent. A task callback
    runs after the task, too late. If a task raises, ``POST_STEP`` is not
    dispatched and that task's key stays current in its thread until the next
    task starts.

Edge tags
    Recall results are ``retrieved_in_scope`` unless the caller (looked up a
    few frames above the wrapper, to see through ``MemoryScope``) is a
    crewai 1.15.22 path that puts every returned match into the prompt:

    - ``Agent._retrieve_memory_context``: appended to the task prompt.
    - ``Agent._prepare_kickoff``: appended to the kickoff messages.
    - ``LiteAgent._inject_memory_context``: appended to the system message.
    - ``RecallMemoryTool._run``: returned to the model as the tool result.

    Those are tagged ``in_prompt``. Other callers (user code, ``Flow.recall``,
    human-feedback lessons) stay ``retrieved_in_scope``.

Compaction
    crewai 1.15.22 consolidates on save: ``EncodingFlow`` searches for similar
    records and may rewrite (``storage.update``) or delete
    (``storage.delete``) them. During a wrapped ``remember`` the storage's
    ``search``, ``save``, ``update`` and ``delete`` are observed; the save pool
    runs on a copy of the caller's context, so the observation follows it. If
    any record was rewritten or deleted, each rewritten or newly saved record
    is recorded with ``capture.on_compaction``, replacing every record the
    consolidation consulted, rewrote or deleted. ``Memory.update()`` called
    directly writes outside ``remember`` and is not captured.

``remember_many`` (background, non-blocking) raises ``NotImplementedError``.
A crew's automatic post-task save goes through it; crewai logs and swallows
the error, so an instrumented crew doesn't auto-save until it is supported.
"""

import contextvars
import functools
import inspect
import logging
import sys
import threading
import weakref
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dod.capture import Capture
from dod.ledger import SOURCES

logger = logging.getLogger(__name__)

Key = Tuple[str, str]

_MARK = "_dod_instrumented"

_key: "contextvars.ContextVar[Optional[Key]]" = contextvars.ContextVar(
    "dod_crewai_key", default=None
)
_observed: "contextvars.ContextVar[Optional[_Consolidation]]" = contextvars.ContextVar(
    "dod_crewai_consolidation", default=None
)
# (task_id, previous key) per task scope entered by the PRE_STEP hook.
_hook_stack: "contextvars.ContextVar[Tuple[Tuple[str, Optional[Key]], ...]]" = (
    contextvars.ContextVar("dod_crewai_hook_stack", default=())
)

# Every Capture passed to instrument(); entering a scope resets its key in each.
_captures: "weakref.WeakSet[Capture]" = weakref.WeakSet()
_captures_lock = threading.Lock()

# Code objects of callers that put every recall result into the prompt.
_in_prompt_callers: set = set()
# Frames above the wrapper to search: covers MemoryScope/MemorySlice and
# arecall delegating to recall.
_CALLER_DEPTH = 4


class _Consolidation:
    """Storage calls observed during one ``remember``."""

    __slots__ = ("consulted", "deleted", "updated", "saved")

    def __init__(self) -> None:
        self.consulted: Dict[str, None] = {}
        self.deleted: Dict[str, None] = {}
        self.updated: Dict[str, Any] = {}
        self.saved: List[Any] = []


def instrument(memory: Any, capture: Capture, store: str = "crewai") -> Any:
    """Wrap ``memory``'s ``recall``, ``remember`` and ``remember_many`` in place.

    Signatures, return values and exceptions of ``recall`` and ``remember``
    are unchanged. Instrumenting an already instrumented memory is a no-op.
    """
    if getattr(getattr(memory, "recall", None), _MARK, False):
        return memory
    storage = getattr(memory, "_storage", None)
    if storage is None:
        raise TypeError(f"{type(memory).__name__} has no _storage to observe")
    if type(memory).__module__.startswith("crewai."):
        _load_in_prompt_callers()
    _observe_storage(storage)
    with _captures_lock:
        _captures.add(capture)

    orig_recall = memory.recall
    orig_remember = memory.remember
    orig_remember_many = memory.remember_many
    warned = []

    @functools.wraps(orig_recall)
    def recall(*args, **kwargs):
        results = orig_recall(*args, **kwargs)
        key = _key.get()
        if key is not None and results:
            capture.on_retrieve(
                key, [(store, m.record.id) for m in results], _tag(sys._getframe(1))
            )
        return results

    @functools.wraps(orig_remember)
    def remember(*args, **kwargs):
        content = _arg(args, kwargs, 0, "content")
        source = _source(_arg(args, kwargs, 5, "source"))
        key = _key.get()
        if key is None:
            if not warned:
                warned.append(True)
                logger.warning(
                    "dod: remember() outside agent_task_scope; recorded with no parents"
                )
            # A fresh key has no context, and no other write can share it.
            ctx_key, agent_id, session_id = object(), None, None
        else:
            ctx_key, (agent_id, session_id) = key, key
        if not isinstance(content, (str, bytes)):
            content = str(content)
        write_id = capture.on_write(ctx_key, content, store, source, agent_id, session_id)

        obs = _Consolidation()
        token = _observed.set(obs)
        try:
            record = orig_remember(*args, **kwargs)
        finally:
            _observed.reset(token)
            _record_compaction(capture, ctx_key, obs, store, source, agent_id, session_id)
        if record is not None:
            capture.bind(write_id, record.id)
        return record

    @functools.wraps(orig_remember_many)
    def remember_many(*args, **kwargs):
        raise NotImplementedError(
            "dod: remember_many() saves in the background and isn't captured yet; "
            "use remember() per item"
        )

    for name, fn in (
        ("recall", recall),
        ("remember", remember),
        ("remember_many", remember_many),
    ):
        setattr(fn, _MARK, True)
        # Memory is a pydantic model, which rejects assigning non-field names.
        object.__setattr__(memory, name, fn)
    return memory


@contextmanager
def agent_task_scope(agent_id: Any, task_id: Any) -> Iterator[Key]:
    """Make ``(agent_id, task_id)`` the current context key.

    Resets the key's context on entry and restores the previous key on exit,
    so scopes nest.
    """
    key = (str(agent_id), str(task_id))
    _reset(key)
    token = _key.set(key)
    try:
        yield key
    finally:
        _key.reset(token)


def install_task_scopes() -> None:
    """Enter ``agent_task_scope(agent.id, task.id)`` around every crewai task.

    Registers global ``PRE_STEP``/``POST_STEP`` interception hooks. Idempotent.
    """
    from crewai.hooks.dispatch import InterceptionPoint, get_hooks, register

    if _enter_task_scope not in get_hooks(InterceptionPoint.PRE_STEP):
        register(InterceptionPoint.PRE_STEP, _enter_task_scope)
    if _exit_task_scope not in get_hooks(InterceptionPoint.POST_STEP):
        register(InterceptionPoint.POST_STEP, _exit_task_scope)


def _enter_task_scope(ctx: Any) -> None:
    agent, task = getattr(ctx, "agent", None), getattr(ctx, "task", None)
    if getattr(ctx, "kind", None) != "task" or agent is None or task is None:
        return None
    key = (str(agent.id), str(task.id))
    _reset(key)
    _hook_stack.set(_hook_stack.get() + ((key[1], _key.get()),))
    _key.set(key)
    return None  # a non-None return would replace the step's payload


def _exit_task_scope(ctx: Any) -> None:
    task = getattr(ctx, "task", None)
    if getattr(ctx, "kind", None) != "task" or task is None:
        return None
    stack, task_id = _hook_stack.get(), str(task.id)
    for i in range(len(stack) - 1, -1, -1):
        if stack[i][0] == task_id:
            _key.set(stack[i][1])
            _hook_stack.set(stack[:i])
            break
    return None


def _reset(key: Key) -> None:
    with _captures_lock:
        captures = list(_captures)
    for cap in captures:
        cap.reset(key)


def _arg(args: tuple, kwargs: dict, index: int, name: str) -> Any:
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


def _source(value: Any) -> str:
    return value if value in SOURCES else "agent_inference"


def _tag(frame: Any) -> str:
    for _ in range(_CALLER_DEPTH):
        if frame is None:
            break
        if frame.f_code in _in_prompt_callers:
            return "in_prompt"
        frame = frame.f_back
    return "retrieved_in_scope"


@functools.lru_cache(maxsize=None)
def _load_in_prompt_callers() -> None:
    from crewai.agent.core import Agent
    from crewai.lite_agent import LiteAgent
    from crewai.tools.memory_tools import RecallMemoryTool

    for fn in (
        Agent._retrieve_memory_context,
        Agent._prepare_kickoff,
        LiteAgent._inject_memory_context,
        RecallMemoryTool._run,
    ):
        _in_prompt_callers.add(inspect.unwrap(fn).__code__)


def _observe_storage(storage: Any) -> None:
    def on_search(obs, args, kwargs, result):
        for record, _score in result or ():
            obs.consulted[record.id] = None

    def on_save(obs, args, kwargs, result):
        obs.saved.extend(_arg(args, kwargs, 0, "records") or ())

    def on_update(obs, args, kwargs, result):
        record = _arg(args, kwargs, 0, "record")
        obs.updated[record.id] = record

    def on_delete(obs, args, kwargs, result):
        for record_id in _arg(args, kwargs, 2, "record_ids") or ():
            obs.deleted[record_id] = None

    for name, observe in (
        ("search", on_search),
        ("save", on_save),
        ("update", on_update),
        ("delete", on_delete),
    ):
        orig = getattr(storage, name)
        if getattr(orig, _MARK, False):
            continue

        def wrapper(*args, _orig=orig, _observe=observe, **kwargs):
            result = _orig(*args, **kwargs)
            obs = _observed.get()
            if obs is not None:
                _observe(obs, args, kwargs, result)
            return result

        functools.update_wrapper(wrapper, orig)
        setattr(wrapper, _MARK, True)
        object.__setattr__(storage, name, wrapper)


def _record_compaction(capture, key, obs, store, source, agent_id, session_id) -> None:
    touched = list(obs.deleted) + list(obs.updated)
    if not touched:
        return
    replaced = [(store, rid) for rid in dict.fromkeys(list(obs.consulted) + touched)]
    # Rewritten content comes from the consolidation LLM; saved content is the
    # caller's own.
    merged = [(r, "agent_inference") for r in obs.updated.values()]
    merged += [(r, source) for r in obs.saved if r.id not in obs.updated]
    for record, src in merged:
        write_id = capture.on_compaction(
            key, replaced, record.content, store, src, agent_id, session_id
        )
        capture.bind(write_id, record.id)
