"""OpenAI Agents SDK integration: capture lineage from an agent's function
tools (openai-agents 0.20.0).

Agents built with the SDK remember through tools (a database, a CRM, a
vector store) and through the conversation passed back into later runs
(``result.to_input_list()`` or a ``Session``). ``instrument_tools`` wraps
``FunctionTool``s in place::

    from dod.integrations.openai_agents import instrument_tools

    agent = Agent(name="support", tools=instrument_tools([lookup, save], capture,
                                                         read_only=["lookup"]))

Scoping
    Inside ``agent_task_scope`` every tool call uses that scope. Outside
    one, each run is its own unit of work, keyed ``(agent name,
    "run-<n>")``: the SDK shares one ``Usage`` object between a run's tool
    calls, which identifies the run, and the key's context is cleared when
    the run's ``Usage`` is garbage collected. Every tool call runs under the
    key, so a store instrumented with another dod integration and used
    inside a tool links its writes to the run's context.

Context
    A tool result goes back to the model, so it is in the prompt from then
    on. Results of this run's read tools are added as ``in_prompt``; so are
    read results from earlier runs that the run's input carries (a
    ``function_call_output`` whose ``function_call`` named a read tool).
    Its record ID is ``"call:<call_id>"``.

Reads and writes
    As in ``dod.integrations.pydantic_ai``: tools in ``read_only`` are
    reads (the result is recorded after the call, source ``tool``, bound to
    its call item); tools in ``passthrough`` are only scoped; every other
    tool is a write, recorded before the call (source ``agent_inference``,
    content the tool name and arguments) and bound after it to
    ``record_id(tool, args, result)`` or its call item. ``deleted(tool,
    args, result)`` names records the call deleted (tombstones).

    The SDK's default failure handler turns a tool's exception into an
    error message returned to the model, so the wrapper sees a successful
    call: a failed write is then bound like a successful one. Create tools
    with ``failure_error_function=None`` to let failures raise (the write
    then stays unbound).

Quarantine
    Quarantined call items are left out of the context.
"""

import json
import logging
import threading
import weakref
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, List, Optional

from agents import FunctionTool

from dod import scope as _scope
from dod.capture import Capture
from dod.excise import deletion_reason
from dod.quarantine import QuarantineCache, count_blocked

logger = logging.getLogger(__name__)

RecordIdFn = Callable[[str, Dict[str, Any], Any], Optional[str]]
DeletedFn = Callable[[str, Dict[str, Any], Any], Iterable[str]]

_MARK = "_dod_instrumented"


def call_record_id(call_id: Optional[str]) -> str:
    """The record ID of the tool call ``call_id``."""
    return f"call:{call_id}"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _plain(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if dump else value


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


class _Runs:
    """Run keys for tool calls made outside ``agent_task_scope``."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.keys: Dict[int, tuple] = {}  # id(Usage) -> key; Usage isn't hashable
        self.n = 0

    def key(self, ctx) -> tuple:
        usage = ctx.usage
        with self.lock:
            key = self.keys.get(id(usage))
            if key is None:
                self.n += 1
                name = getattr(getattr(ctx, "agent", None), "name", None) or "openai_agents"
                key = self.keys[id(usage)] = (str(name), f"run-{self.n}")
                # Dropped (and the context cleared) before the id can be reused.
                weakref.finalize(usage, self._done, id(usage), key)
        return key

    def _done(self, ident: int, key: tuple) -> None:
        with self.lock:
            self.keys.pop(ident, None)
        _scope.reset(key)


def instrument_tools(tools: Iterable[Any], capture: Capture, *, store: str = "openai_agents",
                     read_only: Iterable[str] = (), passthrough: Iterable[str] = (),
                     record_id: Optional[RecordIdFn] = None,
                     deleted: Optional[DeletedFn] = None) -> List[Any]:
    """Wrap each ``FunctionTool`` in ``tools`` in place (others are returned
    unchanged); return the list. Instrumenting a tool twice is a no-op."""
    if not isinstance(capture, Capture):
        raise TypeError("capture must be a dod Capture")
    read_only, passthrough = frozenset(read_only), frozenset(passthrough)
    overlap = read_only & passthrough
    if overlap:
        raise ValueError(f"tools both read_only and passthrough: {sorted(overlap)}")
    _scope.register(capture)
    quarantine = QuarantineCache(capture._ledger)
    runs = _Runs()
    tools = list(tools)

    def allowed(items: List[tuple]) -> List[tuple]:
        blocked = quarantine.current()
        if blocked:
            dropped = [i for i in items if i in blocked]
            if dropped:
                count_blocked(dropped)
                items = [i for i in items if i not in blocked]
        return items

    def history(ctx) -> List[tuple]:
        items = ctx.turn_input if isinstance(getattr(ctx, "turn_input", None), list) else []
        reads = {_field(i, "call_id") for i in items
                 if _field(i, "type") == "function_call" and _field(i, "name") in read_only}
        return allowed([(store, call_record_id(_field(i, "call_id"))) for i in items
                        if _field(i, "type") == "function_call_output"
                        and _field(i, "call_id") in reads])

    def safely(name: str, fn, *args) -> None:
        try:
            fn(*args)
        except Exception:
            logger.exception("dod: recording tool call %r failed after it succeeded;"
                             " its lineage is incomplete", name)

    def record_read(key, name, ctx, result) -> None:
        rid = call_record_id(ctx.tool_call_id)
        capture.on_batch(key, store,
                         [((), _canonical({"tool": name, "result": _plain(result)}), "tool", rid)],
                         agent_id=key[0], session_id=key[1])
        capture.on_retrieve(key, [(store, rid)], "in_prompt")

    def bind_write(key, write_id, name, args, ctx, result) -> None:
        rid = None
        if record_id is not None:
            try:
                rid = record_id(name, args, result)
            except Exception:
                logger.exception("dod: record_id(%r) failed; using the call ID", name)
        rid = str(rid) if rid is not None else call_record_id(ctx.tool_call_id)
        gone: List[str] = []
        if deleted is not None:
            try:
                gone = [str(r) for r in deleted(name, args, result) or ()]
            except Exception:
                logger.exception("dod: deleted(%r) failed; no tombstones recorded", name)
        capture.on_batch(key, store, binds=[(write_id, rid)], agent_id=key[0], session_id=key[1],
                         deletes=[(r, deletion_reason(), write_id) for r in gone])
        if (store, rid) not in quarantine.current():
            capture.on_retrieve(key, [(store, rid)], "in_prompt")

    def wrap(tool: FunctionTool) -> None:
        invoke, name = tool.on_invoke_tool, tool.name

        async def on_invoke_tool(ctx, raw: str):
            key = _scope.current_key()
            scoped = key is None
            if scoped:
                key = runs.key(ctx)
            with _scope.use_key(*key) if scoped else nullcontext():
                past = history(ctx)
                if past:
                    capture.on_retrieve(key, past, "in_prompt")
                if name in passthrough:
                    return await invoke(ctx, raw)
                try:
                    args = json.loads(raw) if raw else {}
                except ValueError:
                    args = {"_raw": raw}
                if name in read_only:
                    result = await invoke(ctx, raw)
                    safely(name, record_read, key, name, ctx, result)
                    return result
                (write_id,) = capture.on_batch(
                    key, store,
                    [((), _canonical({"tool": name, "arguments": args}), "agent_inference", None)],
                    agent_id=key[0], session_id=key[1])
                result = await invoke(ctx, raw)
                safely(name, bind_write, key, write_id, name, args, ctx, result)
                return result

        tool.on_invoke_tool = on_invoke_tool
        setattr(tool, _MARK, True)

    for t in tools:
        if isinstance(t, FunctionTool) and not getattr(t, _MARK, False):
            wrap(t)
    return tools
