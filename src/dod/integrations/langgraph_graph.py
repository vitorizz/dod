"""LangGraph agents: capture lineage from graph runs (langgraph 1.2.14).

``dod.integrations.langgraph`` captures a ``BaseStore`` (long-term memory),
but leaves scoping to the caller and doesn't see the graph's own state (the
thread's short-term memory). ``instrument_graph`` adds both to a compiled
graph::

    from dod.integrations import langgraph, langgraph_graph

    store = langgraph.instrument(InMemoryStore(), cap)
    app = langgraph_graph.instrument_graph(
        builder.compile(checkpointer=saver, store=store), cap)
    app.invoke({"messages": [...]}, {"configurable": {"thread_id": "t1"}})

Scoping
    ``invoke``, ``ainvoke``, ``stream`` and ``astream`` run in
    ``agent_task_scope(<graph name>, <thread_id>)`` (thread ``"default"``
    without one), so every instrumented store a node uses links its writes
    to the run's context. A call made inside an existing scope (the
    caller's, or a parent graph's, for a subgraph) keeps that scope, and its
    state is recorded under it. Nodes run in the caller's context, or a copy
    of it, so the key reaches them.

State
    Each state channel of a thread is an item in store ``store`` (default
    ``"langgraph_state"``), record ID ``json.dumps([thread_id,
    checkpoint_ns, channel])``. Channels starting with ``__`` or
    ``branch:`` are internal and skipped.

    - With a checkpointer, the thread's saved state is read once when a run
      starts (``get_state``), and every channel in it is added to the run's
      context as ``retrieved_in_scope``: nodes see the whole state, and
      which channels a node reads isn't visible.
    - The run's input (source ``user``) and each top-level node's returned
      update (source ``agent_inference``) are recorded per channel, bound to
      the channel's item in one transaction, and added to the context, so
      later nodes take them as parents. A channel already in the context is
      being overwritten: a compaction of its previous value.

    These are observed with a LangChain callback handler added to the run's
    ``config["callbacks"]``, which LangGraph calls synchronously when the
    graph starts and as each top-level node finishes, before the next step
    runs. (The checkpointer can't be used: LangGraph saves writes in the
    background, after later nodes may have run, and graph input bypasses
    it.) A subgraph's output is recorded as its parent node's update.
    Non-dict updates (``Command``, lists) are not recorded.

    The content hashed is the written value as sorted-key JSON (non-JSON
    values by ``str``).

Quarantine and excision
    A graph needs its whole state to run, so quarantined state items are not
    filtered from it; ``trace``, ``blast_radius`` and quarantine of what was
    derived from them (in instrumented stores) work as usual. To remove a
    bad value from a thread, update its state (``update_state``).
"""

import contextvars
import functools
import json
import logging
from typing import Any, List, Optional, Tuple

from langchain_core.callbacks import BaseCallbackHandler

from dod import scope as _scope
from dod.capture import Capture

__all__ = ["instrument_graph", "record_id"]

logger = logging.getLogger(__name__)

_MARK = "_dod_instrumented"
# Graphs (by id) with a wrapped call in progress in this context: invoke
# calls stream internally, and only the outer call records.
_active: "contextvars.ContextVar[frozenset]" = contextvars.ContextVar(
    "dod_langgraph_active", default=frozenset()
)


def record_id(thread_id: Any, checkpoint_ns: str, channel: str) -> str:
    """The record ID of ``channel`` in a thread's state."""
    return json.dumps([str(thread_id), checkpoint_ns or "", channel])


def _internal(channel: str) -> bool:
    return channel.startswith("__") or channel.startswith("branch:")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _configurable(config: Optional[dict]) -> dict:
    return (config or {}).get("configurable") or {}


class _StateRecorder(BaseCallbackHandler):
    """Records one run's input and top-level node updates."""

    run_inline = True  # called in the node's own thread or task, in order
    raise_error = True  # a failed record stops the run: no write without lineage

    def __init__(self, capture: Capture, store: str, key: Tuple[str, str], thread: str,
                 ns: str) -> None:
        self.capture = capture
        self.store = store
        self.key = key
        self.thread = thread
        self.ns = ns
        self.root: Optional[Any] = None
        self.nodes: set = set()  # run IDs of top-level node runs

    def item(self, channel: str) -> Tuple[str, str]:
        return (self.store, record_id(self.thread, self.ns, channel))

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                       metadata=None, **kwargs) -> None:
        # The handler is added for this call only, so the first chain it
        # sees is the graph's own run, whatever outer run (if any) it has.
        if self.root is None:
            self.root = run_id
            self.record(inputs, "user")
        elif parent_run_id is not None and parent_run_id == self.root and (
            (metadata or {}).get("langgraph_node") is not None
        ):
            self.nodes.add(run_id)

    def on_chain_end(self, outputs, *, run_id, **kwargs) -> None:
        if run_id in self.nodes:
            self.nodes.discard(run_id)
            self.record(outputs, "agent_inference")

    def on_chain_error(self, error, *, run_id, **kwargs) -> None:
        self.nodes.discard(run_id)

    def record(self, update: Any, source: str) -> None:
        if not isinstance(update, dict):
            return
        channels = [c for c in update if isinstance(c, str) and not _internal(c)]
        if not channels:
            return
        key = self.key
        items = [self.item(c) for c in channels]
        with self.capture._lock:
            ctx = self.capture._context.get(key, {})
            seen = [i in ctx for i in items]
        batch = [([i] if s else [], _canonical(update[c]), source, i[1])
                 for i, c, s in zip(items, channels, seen)]
        self.capture.on_batch(key, self.store, batch, agent_id=key[0], session_id=key[1])
        self.capture.on_retrieve(key, items)


def _with_handler(config: Optional[dict], handler: BaseCallbackHandler) -> dict:
    config = dict(config or {})
    callbacks = config.get("callbacks")
    if callbacks is None:
        config["callbacks"] = [handler]
    elif isinstance(callbacks, list):
        config["callbacks"] = [*callbacks, handler]
    else:  # a callback manager
        manager = callbacks.copy()
        manager.add_handler(handler, inherit=True)
        config["callbacks"] = manager
    return config


def instrument_graph(graph: Any, capture: Capture, store: str = "langgraph_state") -> Any:
    """Scope ``graph``'s runs and capture its state; see the module docs.

    ``graph`` is a compiled LangGraph graph (a ``Pregel``). ``invoke``,
    ``ainvoke``, ``stream`` and ``astream`` are wrapped in place;
    signatures, results and exceptions are unchanged. Instrumenting twice is
    a no-op.
    """
    if getattr(getattr(graph, "invoke", None), _MARK, False):
        return graph
    for name in ("invoke", "ainvoke", "stream", "astream", "get_state"):
        if not callable(getattr(graph, name, None)):
            raise TypeError(f"{type(graph).__name__} has no {name}(); pass a compiled graph")
    _scope.register(capture)
    agent = str(getattr(graph, "name", None) or "langgraph")
    has_saver = getattr(graph, "checkpointer", None) not in (None, False)

    def prepare(args, kwargs):
        """The call's args and kwargs with our handler added, the run's key
        and its handler."""
        if "config" in kwargs or len(args) < 2:
            config, where = kwargs.get("config"), "kw"
        else:
            config, where = args[1], "pos"
        c = _configurable(config)
        thread = str(c.get("thread_id", "default"))
        key = _scope.current_key() or (agent, thread)
        handler = _StateRecorder(capture, store, key, thread, str(c.get("checkpoint_ns", "")))
        config = _with_handler(config, handler)
        if where == "pos":
            args = (args[0], config, *args[2:])
        else:
            kwargs = {**kwargs, "config": config}
        return args, kwargs, key, handler, c

    def enter(handler: _StateRecorder, c: dict, key):
        """Enter the run's scope (none if one is already active) and load the
        thread's state into its context. Returns the function that leaves it."""
        active = _active.set(_active.get() | {id(graph)})
        if _scope.current_key() is not None:
            scope = None
        else:
            scope = _scope.agent_task_scope(*key)
            scope.__enter__()
        if has_saver and "thread_id" in c:
            try:
                snap = graph.get_state({"configurable": dict(c)})
                values = getattr(snap, "values", None) or {}
                items = [handler.item(ch) for ch in values
                         if isinstance(ch, str) and not _internal(ch)]
                if items:
                    capture.on_retrieve(key, items)
            except Exception:
                logger.exception("dod: couldn't read the thread's state; its channels"
                                 " aren't parents of this run")

        def leave() -> None:
            try:
                _active.reset(active)
            except ValueError:
                pass  # another context: it ends with the generator anyway
            if scope is None:
                return
            try:
                scope.__exit__(None, None, None)
            except ValueError:
                # A generator closed from another context can't reset the
                # ContextVar there; the run is over, so just free its context.
                _scope.reset(key)

        return leave

    def wrap_call(orig):
        @functools.wraps(orig)
        def call(*args, **kwargs):
            if id(graph) in _active.get():
                return orig(*args, **kwargs)
            args, kwargs, key, handler, c = prepare(args, kwargs)
            leave = enter(handler, c, key)
            try:
                return orig(*args, **kwargs)
            finally:
                leave()
        return call

    def wrap_acall(orig):
        @functools.wraps(orig)
        async def call(*args, **kwargs):
            if id(graph) in _active.get():
                return await orig(*args, **kwargs)
            args, kwargs, key, handler, c = prepare(args, kwargs)
            leave = enter(handler, c, key)
            try:
                return await orig(*args, **kwargs)
            finally:
                leave()
        return call

    def wrap_stream(orig):
        def body(*args, **kwargs):
            if id(graph) in _active.get():
                yield from orig(*args, **kwargs)
                return
            args, kwargs, key, handler, c = prepare(args, kwargs)
            leave = enter(handler, c, key)
            try:
                yield from orig(*args, **kwargs)
            finally:
                leave()

        @functools.wraps(orig)
        def stream(*args, **kwargs):
            # A generator runs in its caller's context, so the scope and
            # ``_active`` it sets would leak to the caller between chunks: a
            # run started while this one is suspended would take this run's
            # key and context. Each step runs in the stream's own copy of
            # the context instead.
            ctx = contextvars.copy_context()
            gen = body(*args, **kwargs)
            try:
                while True:
                    try:
                        chunk = ctx.run(next, gen)
                    except StopIteration:
                        return
                    yield chunk
            finally:
                ctx.run(gen.close)
        return stream

    def wrap_astream(orig):
        @functools.wraps(orig)
        async def astream(*args, **kwargs):
            if id(graph) in _active.get():
                async for chunk in orig(*args, **kwargs):
                    yield chunk
                return
            args, kwargs, key, handler, c = prepare(args, kwargs)
            leave = enter(handler, c, key)
            try:
                async for chunk in orig(*args, **kwargs):
                    yield chunk
            finally:
                leave()
        return astream

    wrapped: List[Tuple[str, Any]] = [
        ("invoke", wrap_call), ("ainvoke", wrap_acall),
        ("stream", wrap_stream), ("astream", wrap_astream),
    ]
    for name, wrap in wrapped:
        fn = wrap(getattr(graph, name))
        setattr(fn, _MARK, True)
        object.__setattr__(graph, name, fn)
    return graph
