"""MCP integration: capture lineage from Model Context Protocol tool calls
and resource reads (mcp 1.28.1).

Two ways in, sharing one implementation:

- ``instrument(session, capture, server)`` wraps an ``mcp.ClientSession``
  instance's ``call_tool``, ``read_resource`` and ``list_tools`` in place,
  for a harness that talks to MCP servers itself.
- ``Gateway(capture, {server: session})`` is an MCP server that proxies one
  or more upstream sessions, instrumented the same way. Point any MCP client
  (a desktop app, an IDE, another agent) at the gateway instead of the
  servers, and every call through it is captured without changing the
  client.

Store and record IDs
    Each upstream server is its own store, ``"mcp:<server>"``. A resource's
    record ID is its URI. A tool call's record ID is ``"call:<hex>"``
    (unique per call) unless ``record_id(tool, arguments, result)`` returns
    one, for tools whose result names the record they wrote (a ticket or CRM
    ID). Returning ``None`` falls back to the call ID.

Reads and writes
    A tool named in ``read_only``, or annotated ``readOnlyHint: true`` (as
    listed by ``list_tools``) and not named in ``writes``, is a read; every
    other tool is a write. ``writes`` overrides a server's hint, which is
    the server's claim, not a guarantee. Unknown tools are writes: recording
    a write for a read costs one row, while missing a write breaks lineage.

    - ``read_resource(uri)`` adds ``(store, uri)`` to the scope's context as
      ``retrieved_in_scope``: the harness, not the model, decides whether the
      contents reach the prompt.
    - A read tool's result enters the model's context, but it has no record
      ID of its own, so the call is recorded as a write (source ``tool``,
      content the result's canonical JSON) bound to its call ID in one
      transaction, and that item is added to the context as ``in_prompt``.
      What the model later writes from that result traces back to the call.
    - A write tool call is recorded before the call (source
      ``agent_inference``, content the canonical JSON of the tool name and
      arguments), so a ledger failure stops the call. After it succeeds, the
      write is bound to its record ID and the result is added to the context
      as ``in_prompt``. A call that raises or returns ``isError`` stays
      unbound.
    - ``deleted(tool, arguments, result)``, if given, names records a
      successful call deleted; each gets a tombstone (reason ``forget``, or
      ``excision`` inside ``dod.excise``).

Scoping
    ``instrument`` reads ``dod.scope``'s key, like every integration. The
    gateway can't see its clients' scopes (they run in another process), so
    each client connection is one unit of work: its key is
    ``("mcp-client", <connection id>)``, kept for the connection's lifetime
    and cleared when the connection's session is garbage-collected. A client
    can name its own unit per request with ``_meta`` keys ``dod/agent_id``
    and ``dod/task_id``.

Quarantine
    Reading a quarantined resource raises ``McpError`` with the
    "resource not found" code (-32002) without calling the server; a
    quarantined call item is dropped before it reaches the context. Listings
    are unchanged.

Excision and reconciliation
    MCP has no generic delete or list-everything call, so there is no
    ``DeleteAdapter`` or ``ReconcileAdapter`` here; use the store's own
    integration (or quarantine, which needs none) for those.
"""

import json
import logging
import threading
import uuid
import weakref
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from mcp import types
from mcp.shared.exceptions import McpError

from dod import scope as _scope
from dod.capture import Capture
from dod.excise import deletion_reason
from dod.quarantine import QuarantineCache, count_blocked

logger = logging.getLogger(__name__)

_MARK = "_dod_instrumented"
# MCP's "resource not found" error code.
RESOURCE_NOT_FOUND = -32002
META_AGENT = "dod/agent_id"
META_TASK = "dod/task_id"

RecordIdFn = Callable[[str, Dict[str, Any], Any], Optional[str]]
DeletedFn = Callable[[str, Dict[str, Any], Any], Iterable[str]]


def store_name(server: str) -> str:
    """The store name for upstream ``server``."""
    return f"mcp:{server}"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _dump(result: Any) -> Any:
    dump = getattr(result, "model_dump", None)
    return dump(mode="json", by_alias=True, exclude_none=True) if dump else result


class _Recorder:
    """Capture for one upstream server's tool calls and resource reads."""

    def __init__(
        self,
        capture: Capture,
        server: str,
        read_only: Iterable[str] = (),
        record_id: Optional[RecordIdFn] = None,
        deleted: Optional[DeletedFn] = None,
        writes: Iterable[str] = (),
    ) -> None:
        self.capture = capture
        self.store = store_name(server)
        self.read_only = set(read_only)
        self.writes = set(writes)
        overlap = self.read_only & self.writes
        if overlap:
            raise ValueError(f"tools both read_only and writes: {sorted(overlap)}")
        self.annotated_read_only: set = set()
        self.record_id = record_id
        self.deleted = deleted
        self.quarantine = QuarantineCache(capture._ledger)
        self._warned = False
        self._lock = threading.Lock()
        _scope.register(capture)

    def learn(self, tools: Iterable[types.Tool]) -> None:
        with self._lock:
            for tool in tools:
                hint = getattr(tool.annotations, "readOnlyHint", None) if tool.annotations else None
                if hint:
                    self.annotated_read_only.add(tool.name)
                else:
                    self.annotated_read_only.discard(tool.name)

    def is_read(self, tool: str) -> bool:
        if tool in self.read_only:
            return True
        return tool not in self.writes and tool in self.annotated_read_only

    def _key(self):
        key = _scope.current_key()
        if key is None and not self._warned:
            self._warned = True
            logger.warning("dod: MCP call outside agent_task_scope; recorded with no parents")
        return key

    def check_resource(self, uri: str) -> None:
        if (self.store, uri) in self.quarantine.current():
            count_blocked([(self.store, uri)])
            raise McpError(types.ErrorData(code=RESOURCE_NOT_FOUND,
                                           message=f"Resource not found: {uri}"))

    def after_resource(self, uri: str, result: Any = None) -> None:
        key = _scope.current_key()
        if key is not None:
            texts = [getattr(c, "text", None) for c in getattr(result, "contents", ()) or ()]
            self.capture.on_retrieve(key, [(self.store, uri)],
                                     texts=[[t for t in texts if t]])

    def before_call(self, tool: str, arguments: Dict[str, Any]) -> Optional[tuple]:
        """Record a write tool's call before it runs; None for a read tool."""
        if self.is_read(tool):
            return None
        key = self._key()
        agent_id, session_id = key if key is not None else (None, None)
        ctx = key if key is not None else object()
        (write_id,) = self.capture.on_batch(
            ctx, self.store, [((), _canonical({"tool": tool, "arguments": arguments}),
                               "agent_inference", None)],
            agent_id=agent_id, session_id=session_id,
        )
        return ctx, agent_id, session_id, write_id

    def after_call(self, tool: str, arguments: Dict[str, Any], pending: Optional[tuple],
                   result: Any) -> None:
        """Record what a successful call produced. Never raises: the call
        already happened."""
        if getattr(result, "isError", False):
            return
        try:
            if pending is None:
                rid = self._read_item(tool, result)
            else:
                rid = self._bind_write(tool, arguments, pending, result)
            key = _scope.current_key()
            if key is not None and rid is not None:
                if (self.store, rid) in self.quarantine.current():
                    count_blocked([(self.store, rid)])
                else:
                    self.capture.on_retrieve(key, [(self.store, rid)], "in_prompt")
        except Exception:
            logger.exception("dod: recording MCP call %r on %r failed after the call"
                             " succeeded; its lineage is incomplete", tool, self.store)

    def _read_item(self, tool: str, result: Any) -> str:
        key = self._key()
        agent_id, session_id = key if key is not None else (None, None)
        rid = f"call:{uuid.uuid4().hex}"
        self.capture.on_batch(
            key if key is not None else object(), self.store,
            [((), _canonical({"tool": tool, "result": _dump(result)}), "tool", rid)],
            agent_id=agent_id, session_id=session_id,
        )
        return rid

    def _bind_write(self, tool, arguments, pending, result) -> str:
        ctx, agent_id, session_id, write_id = pending
        rid = None
        if self.record_id is not None:
            try:
                rid = self.record_id(tool, arguments, result)
            except Exception:
                logger.exception("dod: record_id(%r) failed; using the call ID", tool)
        rid = str(rid) if rid is not None else f"call:{write_id}"
        gone = []
        if self.deleted is not None:
            try:
                gone = [str(r) for r in self.deleted(tool, arguments, result) or ()]
            except Exception:
                logger.exception("dod: deleted(%r) failed; no tombstones recorded", tool)
        self.capture.on_batch(
            ctx, self.store, binds=[(write_id, rid)], agent_id=agent_id,
            session_id=session_id, deletes=[(r, deletion_reason(), write_id) for r in gone],
        )
        return rid


def instrument(
    session: Any,
    capture: Capture,
    server: str,
    read_only: Iterable[str] = (),
    record_id: Optional[RecordIdFn] = None,
    deleted: Optional[DeletedFn] = None,
    writes: Iterable[str] = (),
) -> Any:
    """Wrap ``session``'s ``call_tool``, ``read_resource`` and ``list_tools``
    in place; see the module docs. ``session`` is an ``mcp.ClientSession``
    (or anything with those coroutine methods). Signatures, results and
    exceptions are unchanged. Instrumenting twice is a no-op.
    """
    if getattr(getattr(session, "call_tool", None), _MARK, False):
        return session
    for name in ("call_tool", "read_resource", "list_tools"):
        if not callable(getattr(session, name, None)):
            raise TypeError(f"{type(session).__name__} has no {name}() to wrap")
    rec = _Recorder(capture, server, read_only, record_id, deleted, writes)
    orig_call, orig_read, orig_list = session.call_tool, session.read_resource, session.list_tools

    async def call_tool(name, arguments=None, *args, **kwargs):
        args_ = dict(arguments or {})
        pending = rec.before_call(name, args_)
        result = await orig_call(name, arguments, *args, **kwargs)
        rec.after_call(name, args_, pending, result)
        return result

    async def read_resource(uri, *args, **kwargs):
        rec.check_resource(str(uri))
        result = await orig_read(uri, *args, **kwargs)
        rec.after_resource(str(uri), result)
        return result

    async def list_tools(*args, **kwargs):
        result = await orig_list(*args, **kwargs)
        rec.learn(getattr(result, "tools", ()))
        return result

    for fn, orig in ((call_tool, orig_call), (read_resource, orig_read), (list_tools, orig_list)):
        fn.__name__, fn.__doc__ = orig.__name__, orig.__doc__
        fn.__wrapped__ = orig
        setattr(fn, _MARK, True)
        object.__setattr__(session, fn.__name__, fn)
    session._dod_recorder = rec
    return session


class Gateway:
    """An MCP server that proxies upstream sessions and captures every call.

    ``upstreams`` maps a server name to a connected ``ClientSession``; each is
    instrumented with ``capture`` (``options`` maps a server name to
    ``instrument``'s keyword arguments). Tools are exposed as
    ``<server>__<tool>``; resources keep their URIs. ``server`` is the
    low-level ``mcp.server.lowlevel.Server``; run it with any transport, for
    example ``await gateway.run_stdio()``.
    """

    SEP = "__"

    def __init__(
        self,
        capture: Capture,
        upstreams: Mapping[str, Any],
        name: str = "dod-gateway",
        options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        from mcp.server.lowlevel import Server

        options = options or {}
        for server in upstreams:
            if self.SEP in server:
                raise ValueError(f"server name {server!r} can't contain {self.SEP!r}")
        self.upstreams = {
            s: instrument(sess, capture, s, **dict(options.get(s, {})))
            for s, sess in upstreams.items()
        }
        self.capture = capture
        self._resources: Dict[str, str] = {}  # uri -> server
        self._keys: "weakref.WeakKeyDictionary[Any, tuple]" = weakref.WeakKeyDictionary()
        self._keys_lock = threading.Lock()
        self.server = server = Server(name)

        @server.list_tools()
        async def list_tools() -> list:
            out = []
            for s, sess in self.upstreams.items():
                for tool in (await sess.list_tools()).tools:
                    out.append(tool.model_copy(update={"name": f"{s}{self.SEP}{tool.name}"}))
            return out

        @server.call_tool(validate_input=False)
        async def call_tool(name: str, arguments: dict) -> types.CallToolResult:
            s, sep, tool = name.partition(self.SEP)
            if not sep or s not in self.upstreams:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
                    isError=True,
                )
            with self._scope():
                return await self.upstreams[s].call_tool(tool, arguments)

        @server.list_resources()
        async def list_resources() -> list:
            out = []
            for s, sess in self.upstreams.items():
                try:
                    resources = (await sess.list_resources()).resources
                except McpError:
                    continue  # the server has no resources
                for r in resources:
                    self._resources[str(r.uri)] = s
                    out.append(r)
            return out

        @server.read_resource()
        async def read_resource(uri) -> list:
            from mcp.server.lowlevel.helper_types import ReadResourceContents

            s = self._resources.get(str(uri))
            if s is None:
                await list_resources()  # a client may read a URI it didn't list
                s = self._resources.get(str(uri))
            if s is None:
                raise McpError(types.ErrorData(code=RESOURCE_NOT_FOUND,
                                               message=f"Resource not found: {uri}"))
            with self._scope():
                result = await self.upstreams[s].read_resource(uri)
            out = []
            for c in result.contents:
                if isinstance(c, types.TextResourceContents):
                    out.append(ReadResourceContents(c.text, c.mimeType))
                else:
                    import base64

                    out.append(ReadResourceContents(base64.b64decode(c.blob), c.mimeType))
            return out

    def _scope(self):
        """The current request's key: its ``_meta`` names, else its connection's."""
        ctx = self.server.request_context
        meta = ctx.meta.model_dump() if ctx.meta is not None else {}
        if meta.get(META_AGENT) is not None or meta.get(META_TASK) is not None:
            return _scope.use_key(meta.get(META_AGENT) or "mcp-client",
                                  meta.get(META_TASK) or "default")
        session = ctx.session
        with self._keys_lock:
            key = self._keys.get(session)
            if key is None:
                key = ("mcp-client", uuid.uuid4().hex)
                self._keys[session] = key
                weakref.finalize(session, _scope.reset, key)
        return _scope.use_key(*key)

    async def run_stdio(self) -> None:
        """Serve over stdio until the client disconnects."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read, write):
            await self.server.run(read, write, self.server.create_initialization_options())
