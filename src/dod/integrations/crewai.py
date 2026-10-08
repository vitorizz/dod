"""CrewAI integration: capture lineage from a ``crewai.Memory`` (crewai 1.15.22).

``instrument(memory, capture)`` wraps the instance's ``recall`` and
``remember``. Both run synchronously in the caller and see exact record IDs
(``MemoryMatch.record.id`` and the returned ``MemoryRecord.id``). Memory
events are not used: save events carry no record ID, and the event bus runs
sync handlers in a thread pool, so they can arrive out of order.

Scoping
    The context key and ``agent_task_scope`` live in ``dod.scope``, shared
    with every other integration (re-exported here). Outside any scope,
    recalls are not tracked, and a write is recorded with zero parents and a
    one-time warning.

    In a crew run, call ``install_task_scopes()`` once. It registers crewai
    ``PRE_STEP``/``POST_STEP`` interception hooks, which ``Task._execute_core``
    and ``Task._aexecute_core`` dispatch synchronously in the task's own
    thread or coroutine, around ``agent.execute_task``. A ``TaskStartedEvent``
    handler can't enter the scope: the event bus runs it in a thread pool on a
    copy of the context, so the key never reaches the agent. A task callback
    runs after the task, too late. A key's context is cleared when its scope
    ends, so finished tasks hold no memory. If a task raises, ``POST_STEP`` is
    not dispatched: that task's key stays current in its thread, and its
    context stays held, until the next task starts.

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
    is recorded as a compaction, replacing every record the consolidation
    consulted, rewrote or deleted. If records were only deleted, the
    compaction is bound to the record ``remember`` returned (crewai returns
    the first similar record, which may be one it just deleted). The
    compactions and the write's bind go to the ledger in one transaction.

Direct edits and deletions
    ``update(record_id, ...)`` is recorded as a compaction of that record, with
    the scope's context, before the store write, and bound to it after. A
    metadata-only update hashes the record's unchanged content.
    ``forget(record_ids=...)`` is recorded as an unbound compaction whose
    parents are the given IDs (with filters too, they over-include the deleted
    set). Both are tagged ``agent_inference``: the method takes no source.
    ``forget`` without ``record_ids``, ``reset`` and ``reset_all`` delete by
    filter, and the storage API doesn't report which IDs it deleted, so they
    pass through and log a warning on every call.

``remember_many``
    Saves in the background and returns ``[]``. Each item is recorded as a
    write when ``remember_many`` is called, with the scope's context as
    parents, so the edge doesn't depend on when the save runs. The wrapped
    ``_background_encode_batch`` observes storage like ``remember`` does and,
    when the save finishes, binds each write to the record inserted with its
    content and records compactions, in one transaction. Items merged into an
    existing record or dropped as in-batch duplicates stay unbound (a merge is
    still a compaction); a one-item batch binds to the returned record.
"""

import contextvars
import functools
import inspect
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

from dod import scope as _scope
from dod.capture import Capture
from dod.ledger import SOURCES
from dod.scope import Key, _key, agent_task_scope

__all__ = ["agent_task_scope", "install_task_scopes", "instrument"]

logger = logging.getLogger(__name__)

_MARK = "_dod_instrumented"

_observed: "contextvars.ContextVar[Optional[_Consolidation]]" = contextvars.ContextVar(
    "dod_crewai_consolidation", default=None
)
# (key, previous key) per task scope entered by the PRE_STEP hook.
_hook_stack: "contextvars.ContextVar[Tuple[Tuple[Key, Optional[Key]], ...]]" = (
    contextvars.ContextVar("dod_crewai_hook_stack", default=())
)
_pending_batch: "contextvars.ContextVar[Optional[_Batch]]" = contextvars.ContextVar(
    "dod_crewai_batch", default=None
)

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


class _Batch:
    """One ``remember_many`` call, handed to its background save."""

    __slots__ = ("key", "writes", "source", "agent_id", "session_id")

    def __init__(self, key, writes, source, agent_id, session_id) -> None:
        self.key = key  # a snapshot: the caller's scope may end before the save
        self.writes: List[Tuple[str, str]] = writes  # (content, write_id)
        self.source = source
        self.agent_id = agent_id
        self.session_id = session_id


def instrument(memory: Any, capture: Capture, store: str = "crewai") -> Any:
    """Wrap ``memory``'s ``recall``, ``remember``, ``remember_many``, ``update``,
    ``forget``, ``reset`` and ``reset_all`` in place.

    Signatures, return values and exceptions are unchanged. Instrumenting an already instrumented memory is a no-op.
    """
    if getattr(getattr(memory, "recall", None), _MARK, False):
        return memory
    storage = getattr(memory, "_storage", None)
    if storage is None:
        raise TypeError(f"{type(memory).__name__} has no _storage to observe")
    if type(memory).__module__.startswith("crewai."):
        _load_in_prompt_callers()
    _observe_storage(storage)
    _scope.register(capture)

    orig_recall = memory.recall
    orig_remember = memory.remember
    orig_remember_many = memory.remember_many
    orig_background = getattr(memory, "_background_encode_batch", None)
    if orig_background is None:
        raise TypeError(
            f"{type(memory).__name__} has no _background_encode_batch to observe"
        )
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

    def scope(method):
        key = _key.get()
        if key is not None:
            return key, key[0], key[1]
        if not warned:
            warned.append(True)
            logger.warning(
                "dod: %s() outside agent_task_scope; recorded with no parents", method
            )
        # A fresh key has no context, and no other write can share it.
        return object(), None, None

    @functools.wraps(orig_remember)
    def remember(*args, **kwargs):
        content = _text(_arg(args, kwargs, 0, "content"))
        source = _source(_arg(args, kwargs, 5, "source"))
        ctx_key, agent_id, session_id = scope("remember")
        write_id = capture.on_write(ctx_key, content, store, source, agent_id, session_id)

        obs = _Consolidation()
        token = _observed.set(obs)
        record = None
        try:
            record = orig_remember(*args, **kwargs)
        finally:
            _observed.reset(token)
            returned = [record] if record is not None else []
            binds = [(write_id, record.id)] if record is not None else []
            _settle(capture, ctx_key, obs, store, source, agent_id, session_id,
                    binds, returned, content)
        return record

    @functools.wraps(orig_remember_many)
    def remember_many(*args, **kwargs):
        contents = [_text(c) for c in _arg(args, kwargs, 0, "contents") or ()]
        if not contents:
            return orig_remember_many(*args, **kwargs)
        source = _source(_arg(args, kwargs, 5, "source"))
        ctx_key, agent_id, session_id = scope("remember_many")
        # Recorded now, in the caller's scope; bound once the background save
        # has run. A read-only memory never saves, so its writes stay unbound.
        write_ids = capture.on_batch(
            ctx_key, store, [((), c, source, None) for c in contents],
            agent_id=agent_id, session_id=session_id,
        )
        if getattr(memory, "read_only", False):
            return orig_remember_many(*args, **kwargs)
        batch = _Batch(
            capture.snapshot(ctx_key), list(zip(contents, write_ids)),
            source, agent_id, session_id,
        )
        token = _pending_batch.set(batch)
        try:
            return orig_remember_many(*args, **kwargs)
        except BaseException:
            capture.reset(batch.key)
            raise
        finally:
            _pending_batch.reset(token)

    @functools.wraps(orig_background)
    def background_encode_batch(*args, **kwargs):
        # Runs on the save pool, in a copy of remember_many's context.
        batch = _pending_batch.get()
        if batch is None:
            return orig_background(*args, **kwargs)
        obs = _Consolidation()
        token = _observed.set(obs)
        records = None
        try:
            records = orig_background(*args, **kwargs)
        finally:
            _observed.reset(token)
            try:
                returned = list(records or ())
                _settle(capture, batch.key, obs, store, batch.source, batch.agent_id,
                        batch.session_id, _batch_binds(batch, obs, returned), returned,
                        batch.writes[0][0])
            finally:
                capture.reset(batch.key)
        return records

    wrapped = [
        ("recall", recall),
        ("remember", remember),
        ("remember_many", remember_many),
        ("_background_encode_batch", background_encode_batch),
    ]

    orig_update = getattr(memory, "update", None)
    if orig_update is not None:

        @functools.wraps(orig_update)
        def update(*args, **kwargs):
            if getattr(memory, "read_only", False):
                return orig_update(*args, **kwargs)  # returns the record unchanged
            record_id = _arg(args, kwargs, 0, "record_id")
            content = _arg(args, kwargs, 1, "content")
            if content is None:
                # Metadata-only update: the record keeps its content.
                existing = storage.get_record(record_id)
                if existing is None:
                    return orig_update(*args, **kwargs)  # raises: not found
                content = existing.content
            ctx_key, agent_id, session_id = scope("update")
            # Recorded before the store write, like remember; bound after it.
            (write_id,) = capture.on_batch(
                ctx_key, store,
                [([(store, record_id)], _text(content), "agent_inference", None)],
                agent_id=agent_id, session_id=session_id,
            )
            record = orig_update(*args, **kwargs)
            capture.bind(write_id, record.id)
            return record

        wrapped.append(("update", update))

    orig_forget = getattr(memory, "forget", None)
    if orig_forget is not None:

        @functools.wraps(orig_forget)
        def forget(*args, **kwargs):
            record_ids = list(_arg(args, kwargs, 4, "record_ids") or ())
            if not record_ids:
                _warn_uncaptured("forget")
                return orig_forget(*args, **kwargs)
            ctx_key, agent_id, session_id = scope("forget")
            # Nothing survives a deletion, so the compaction stays unbound.
            capture.on_batch(
                ctx_key, store,
                [([(store, r) for r in record_ids], "", "agent_inference", None)],
                agent_id=agent_id, session_id=session_id,
            )
            return orig_forget(*args, **kwargs)

        wrapped.append(("forget", forget))

    for name in ("reset", "reset_all"):
        orig = getattr(memory, name, None)
        if orig is None:
            continue

        def reset(*args, _orig=orig, _name=name, **kwargs):
            _warn_uncaptured(_name)
            return _orig(*args, **kwargs)

        wrapped.append((name, functools.update_wrapper(reset, orig)))

    for name, fn in wrapped:
        setattr(fn, _MARK, True)
        # Memory is a pydantic model, which rejects assigning non-field names.
        object.__setattr__(memory, name, fn)
    return memory


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
    _scope.reset(key)
    _hook_stack.set(_hook_stack.get() + ((key, _key.get()),))
    _key.set(key)
    return None  # a non-None return would replace the step's payload


def _exit_task_scope(ctx: Any) -> None:
    task = getattr(ctx, "task", None)
    if getattr(ctx, "kind", None) != "task" or task is None:
        return None
    stack, task_id = _hook_stack.get(), str(task.id)
    for i in range(len(stack) - 1, -1, -1):
        key, previous = stack[i]
        if key[1] == task_id:
            _key.set(previous)
            _hook_stack.set(stack[:i])
            _scope.reset(key)  # the task is done; free its context
            break
    return None


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


def _settle(capture, key, obs, store, source, agent_id, session_id, binds, returned,
            content) -> None:
    """Record a save's compactions and binds in one ledger transaction."""
    compactions = []
    touched = list(obs.deleted) + list(obs.updated)
    if touched:
        replaced = [(store, rid) for rid in dict.fromkeys(list(obs.consulted) + touched)]
        # Rewritten content comes from the consolidation LLM; saved content is
        # the caller's own.
        merged = [(r, "agent_inference") for r in obs.updated.values()]
        merged += [(r, source) for r in obs.saved if r.id not in obs.updated]
        if not merged:
            # Delete-only: bind the compaction to the record(s) the call
            # returned, so the deletion is never missing from the ledger.
            merged = [(r, "agent_inference") for r in returned]
        compactions = [(replaced, r.content, src, r.id) for r, src in merged]
        if not compactions:  # the save failed after deleting; nothing to bind
            compactions = [(replaced, content, "agent_inference", None)]
    if compactions or binds:
        capture.on_batch(key, store, compactions, binds, agent_id, session_id)


def _batch_binds(batch: _Batch, obs: _Consolidation, returned: list) -> list:
    """Bind each ``remember_many`` write to the record inserted for it.

    Inserted records carry their item's content. An item merged into an
    existing record, or dropped as an in-batch duplicate, stays unbound; a
    merge is still recorded as a compaction. A one-item batch binds to the
    returned record, as ``remember`` does.
    """
    saved: Dict[str, List[str]] = {}
    for r in obs.saved:
        saved.setdefault(r.content, []).append(r.id)
    binds = []
    for content, write_id in batch.writes:
        ids = saved.get(content)
        if ids:
            binds.append((write_id, ids.pop(0)))
    if not binds and len(batch.writes) == 1 and len(returned) == 1:
        binds.append((batch.writes[0][1], returned[0].id))
    return binds


def _warn_uncaptured(method: str) -> None:
    # Every call: these are rare, and each one is a gap in the ledger.
    logger.warning(
        "dod: %s() deletes by filter; the deleted record IDs are unknown, so the "
        "deletion is not in the ledger. Pass record_ids to forget() to record it.",
        method,
    )


def _text(content: Any) -> Any:
    return content if isinstance(content, (str, bytes)) else str(content)
