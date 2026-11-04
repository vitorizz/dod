"""Pydantic AI integration: capture lineage from an agent's tool calls
(pydantic-ai-slim 2.66.0).

Pydantic AI has no memory store of its own: agents remember through tools
(a function that reads or writes a database, a CRM, a vector store) and
through the message history passed back into each run. ``CaptureToolset``
wraps any toolset and captures both::

    from dod.integrations.pydantic_ai import CaptureToolset

    agent = Agent(model, toolsets=[CaptureToolset(my_tools, capture,
                                                  read_only=["lookup"])])

Scoping
    Each run is one unit of work, keyed ``(agent name, run_id)`` (agent name
    ``"pydantic_ai"`` if the agent has none). Every tool call runs under
    that key, so a store instrumented with another dod integration and used
    inside a tool links its writes to the run's context, across stores. The
    key's context is cleared when the toolset is exited after the last
    active run.

Context
    A tool result goes back to the model, so it is in the prompt from then
    on, including in later runs that pass the history back. Before each
    call, every read tool result in the message history (``ToolReturnPart``
    of one of this toolset's read tools) is added to the context as
    ``in_prompt``. Its record ID is ``"call:<run_id>:<tool_call_id>"``, from
    the request that carried it, so a later run's writes trace back to
    results an earlier run fetched.

Reads and writes
    Tools named in ``read_only`` are reads; tools named in ``passthrough``
    are only scoped (their own stores are instrumented, so recording them
    again would duplicate each write); every other tool is a write (a missed
    write breaks lineage; an extra one costs a row).

    - A read tool's result is recorded after the call as a write (source
      ``tool``, content the canonical JSON of tool name and result) bound to
      its call ID in one transaction.
    - A write tool is recorded before the call (source ``agent_inference``,
      content the canonical JSON of tool name and arguments), so a ledger
      failure stops it, and bound after success to ``record_id(tool, args,
      result)`` or its call ID. Its result is then added to the run's
      context as ``in_prompt``. A call that raises (including
      ``ModelRetry``) stays unbound.
    - ``deleted(tool, args, result)`` names records a successful call
      deleted; each gets a tombstone (reason ``forget``, or ``excision``
      inside ``dod.excise``).

    Once a call has succeeded, a failure recording it is logged, never
    raised.

Quarantine
    A quarantined call item is left out of the context. Tools that read
    instrumented stores get those stores' own quarantine filtering.

Store
    All calls go to one store, ``store`` (default ``"pydantic_ai"``).
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.toolsets import WrapperToolset

from dod import scope as _scope
from dod.capture import Capture
from dod.excise import deletion_reason
from dod.quarantine import QuarantineCache, count_blocked

logger = logging.getLogger(__name__)

RecordIdFn = Callable[[str, Dict[str, Any], Any], Optional[str]]
DeletedFn = Callable[[str, Dict[str, Any], Any], Iterable[str]]


def call_record_id(run_id: Optional[str], tool_call_id: Optional[str]) -> str:
    """The record ID of the call ``tool_call_id`` made in run ``run_id``."""
    return f"call:{run_id}:{tool_call_id}"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _plain(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if dump else value


@dataclass
class CaptureToolset(WrapperToolset):
    """Wrap ``wrapped`` so every tool call is captured; see the module docs."""

    capture: Capture = field(default=None, kw_only=True)  # type: ignore[assignment]
    store: str = field(default="pydantic_ai", kw_only=True)
    read_only: Iterable[str] = field(default=(), kw_only=True)
    passthrough: Iterable[str] = field(default=(), kw_only=True)
    record_id: Optional[RecordIdFn] = field(default=None, kw_only=True)
    deleted: Optional[DeletedFn] = field(default=None, kw_only=True)

    def __init__(self, wrapped: Any, capture: Capture, *, store: str = "pydantic_ai",
                 read_only: Iterable[str] = (), passthrough: Iterable[str] = (),
                 record_id: Optional[RecordIdFn] = None,
                 deleted: Optional[DeletedFn] = None) -> None:
        if not isinstance(capture, Capture):
            raise TypeError("capture must be a dod Capture")
        self.wrapped = wrapped
        self.capture = capture
        self.store = store
        self.read_only = frozenset(read_only)
        self.passthrough = frozenset(passthrough)
        overlap = self.read_only & self.passthrough
        if overlap:
            raise ValueError(f"tools both read_only and passthrough: {sorted(overlap)}")
        self.record_id = record_id
        self.deleted = deleted
        self._state = _State(QuarantineCache(capture._ledger))
        _scope.register(capture)

    async def for_run(self, ctx):
        new = await self.wrapped.for_run(ctx)
        return self if new is self.wrapped else self._with(new)

    async def for_run_step(self, ctx):
        new = await self.wrapped.for_run_step(ctx)
        return self if new is self.wrapped else self._with(new)

    def _with(self, wrapped: Any) -> "CaptureToolset":
        out = object.__new__(type(self))
        out.__dict__.update(self.__dict__)
        out.wrapped = wrapped
        return out  # shares _state: one set of run keys per toolset

    async def __aenter__(self):
        with self._state.lock:
            self._state.active += 1
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args):
        try:
            return await self.wrapped.__aexit__(*args)
        finally:
            with self._state.lock:
                self._state.active -= 1
                keys = self._state.keys if self._state.active <= 0 else []
                if self._state.active <= 0:
                    self._state.keys = []
                    self._state.known.clear()
            for key in keys:
                _scope.reset(key)

    def _key(self, ctx) -> tuple:
        name = getattr(getattr(ctx, "agent", None), "name", None) or "pydantic_ai"
        key = (str(name), str(ctx.run_id))
        with self._state.lock:
            if key not in self._state.known:
                self._state.known.add(key)
                self._state.keys.append(key)
        return key

    def _history(self, ctx) -> List[tuple]:
        items = []
        for message in ctx.messages:
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name in self.read_only:
                    items.append((self.store, call_record_id(message.run_id, part.tool_call_id)))
        blocked = self._state.quarantine.current()
        if blocked:
            dropped = [i for i in items if i in blocked]
            if dropped:
                count_blocked(dropped)
                items = [i for i in items if i not in blocked]
        return items

    async def call_tool(self, name: str, tool_args: Dict[str, Any], ctx, tool) -> Any:
        key = self._key(ctx)
        with _scope.use_key(*key):
            history = self._history(ctx)
            if history:
                self.capture.on_retrieve(key, history, "in_prompt")
            if name in self.passthrough:
                return await self.wrapped.call_tool(name, tool_args, ctx, tool)
            if name in self.read_only:
                result = await self.wrapped.call_tool(name, tool_args, ctx, tool)
                self._safely(name, self._record_read, key, name, ctx, result)
                return result
            (write_id,) = self.capture.on_batch(
                key, self.store,
                [((), _canonical({"tool": name, "arguments": tool_args}), "agent_inference", None)],
                agent_id=key[0], session_id=key[1],
            )
            result = await self.wrapped.call_tool(name, tool_args, ctx, tool)
            self._safely(name, self._bind_write, key, write_id, name, tool_args, ctx, result)
            return result

    def _safely(self, name, fn, *args) -> None:
        try:
            fn(*args)
        except Exception:
            logger.exception("dod: recording tool call %r failed after it succeeded;"
                             " its lineage is incomplete", name)

    def _record_read(self, key, name, ctx, result) -> None:
        rid = call_record_id(ctx.run_id, ctx.tool_call_id)
        self.capture.on_batch(
            key, self.store,
            [((), _canonical({"tool": name, "result": _plain(result)}), "tool", rid)],
            agent_id=key[0], session_id=key[1],
        )
        self.capture.on_retrieve(key, [(self.store, rid)], "in_prompt")

    def _bind_write(self, key, write_id, name, tool_args, ctx, result) -> None:
        rid = None
        if self.record_id is not None:
            try:
                rid = self.record_id(name, tool_args, result)
            except Exception:
                logger.exception("dod: record_id(%r) failed; using the call ID", name)
        rid = str(rid) if rid is not None else call_record_id(ctx.run_id, ctx.tool_call_id)
        gone: List[str] = []
        if self.deleted is not None:
            try:
                gone = [str(r) for r in self.deleted(name, tool_args, result) or ()]
            except Exception:
                logger.exception("dod: deleted(%r) failed; no tombstones recorded", name)
        self.capture.on_batch(
            key, self.store, binds=[(write_id, rid)], agent_id=key[0], session_id=key[1],
            deletes=[(r, deletion_reason(), write_id) for r in gone],
        )
        if (self.store, rid) not in self._state.quarantine.current():
            self.capture.on_retrieve(key, [(self.store, rid)], "in_prompt")


class _State:
    """Per-toolset state shared by its per-run copies."""

    __slots__ = ("lock", "active", "keys", "known", "quarantine")

    def __init__(self, quarantine: QuarantineCache) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.keys: List[tuple] = []  # run keys since the toolset was last idle
        self.known: set = set()
        self.quarantine = quarantine
