"""Anthropic memory tool integration: capture lineage from the client-side,
file-based memory tool (``memory_20250818``, anthropic 1.12.1).

``instrument(tool, capture)`` wraps a ``BetaAbstractMemoryTool`` or
``BetaAsyncAbstractMemoryTool`` instance's ``execute`` and
``clear_all_memory`` in place (instance attributes; the class is untouched).

The funnel
    In anthropic 1.12.1, ``call(input)`` builds the command and calls
    ``self.execute(command)``, which dispatches to ``view``, ``create``,
    ``str_replace``, ``insert``, ``delete`` and ``rename``. The tool runner
    calls ``call``, so wrapping ``execute`` covers every command Claude
    sends. ``instrument`` raises ``TypeError`` for a class that overrides
    ``call``, which could bypass ``execute``. Calling ``view``, ``create``
    and the other command methods directly bypasses capture.

Record IDs
    A file's record ID is its normalized memory path
    (``posixpath.normpath``, e.g. ``/memories/notes/user.md``). Directories
    have no record ID; their files do.

Scoping
    The context key is ``dod.scope``'s, shared with every integration, so a
    write in another store gets parents viewed here, and the reverse.
    Outside any scope, views are not tracked, and a write is recorded with
    zero parents and a one-time warning.

Reads
    ``view`` of a file returns its text into Claude's context as the tool
    result, so the file is added to the scope's context tagged
    ``in_prompt`` (also for a ``view_range``: part of it was in the prompt).
    ``view`` of a directory lists names, not content: each listed file is
    added as ``retrieved_in_scope``. A failed view records nothing.

Probing the store
    The abstract tool has no "exists" or "list" call, so the wrapper asks
    the tool's own (uninstrumented) ``view`` when it must know what a path
    is: whether ``create`` overwrites a file, and which files a ``delete``
    or ``rename`` of a directory, or ``clear_all_memory``, touches. A result
    is a directory listing if, as in the SDK's tool and the memory tool's
    documented format, its lines after the header are ``<size>\\t<path>``
    with the directory itself first; directories deeper than the listing
    shows are viewed in turn. Anything else that succeeds is a file; a
    ``ToolError`` means the path doesn't exist. A probe that raises anything
    else is logged and treated as "doesn't exist", which can miss a parent
    or a tombstone, never the host's call.

Writes
    Recorded before the tool call, tagged ``agent_inference`` (Claude chose
    the command), and bound after it succeeds. If recording fails, the
    command doesn't run; if the command raises, the write stays unbound.

    - ``create`` of a new path is a write; of an existing file, a
      compaction with the old file as parent.
    - ``str_replace`` and ``insert`` are compactions of the file (parent:
      the file itself, ``in_prompt``, plus the scope's context).
    - ``rename`` is, per file moved, a compaction bound to the new path with
      the old path as parent, plus a tombstone for the old path (reason
      ``other``) naming that write.
    - ``delete`` of a file records a tombstone (reason ``forget``, or
      ``excision`` inside ``dod.excise``). ``delete`` of a directory and
      ``clear_all_memory`` list the files first and record a tombstone per
      file. Tombstones are recorded after the command succeeds.

    The content hashed is what the command writes, not the whole file: the
    file reached after an edit is the previous file plus the edit, and the
    previous file is already a parent. So ``create`` hashes ``file_text``,
    ``str_replace`` hashes ``new_str``, ``insert`` hashes ``insert_text``,
    and ``rename`` (which writes no text) hashes the empty string.

    Once the command has succeeded, a failure recording its binds or
    tombstones is logged, never raised. The tool's own ``ToolError``\\ s
    propagate unchanged.

Quarantine
    ``view`` of a quarantined file raises the tool's usual
    ``ToolError("The path ... does not exist. ...")`` without calling the
    tool, so Claude never sees it and it never becomes a parent; a directory
    listing omits quarantined files (``dod.quarantine``). As in the LangGraph
    integration, editing or creating over a quarantined file still sees it,
    so the edit is a compaction with it as parent.

Excision
    ``DeleteAdapter(tool)`` lets ``dod.excise`` delete files through the
    instrumented ``execute``, so each deletion is tombstoned.
"""

import asyncio
import functools
import logging
import posixpath
from typing import Any, Callable, List, Optional, Tuple

from anthropic.lib.tools import ToolError
from anthropic.tools.memory import BetaAbstractMemoryTool, BetaAsyncAbstractMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818ViewCommand,
)

from dod import scope as _scope
from dod.capture import Capture
from dod.excise import deletion_reason
from dod.quarantine import QuarantineCache, count_blocked

logger = logging.getLogger(__name__)

_MARK = "_dod_instrumented"
_SOURCE = "agent_inference"
_ROOT = "/memories"


def record_id(path: str) -> str:
    """Return the record ID for the memory file at ``path``."""
    return posixpath.normpath(path) if path else path


def instrument(tool: Any, capture: Capture, store: str = "anthropic_memory") -> Any:
    """Wrap ``tool``'s ``execute`` and ``clear_all_memory`` in place.

    Signatures, return values and exceptions are unchanged. Instrumenting an
    already instrumented tool is a no-op.
    """
    if getattr(getattr(tool, "execute", None), _MARK, False):
        return tool
    if isinstance(tool, BetaAbstractMemoryTool):
        base, is_async = BetaAbstractMemoryTool, False
    elif isinstance(tool, BetaAsyncAbstractMemoryTool):
        base, is_async = BetaAsyncAbstractMemoryTool, True
    else:
        raise TypeError(f"{type(tool).__name__} is not an Anthropic memory tool")
    if type(tool).call is not base.call:
        raise TypeError(
            f"{type(tool).__name__} overrides call; only tools whose call routes "
            "through execute can be captured"
        )
    _scope.register(capture)
    rec = _Recorder(capture, store)
    execute, clear_all = tool.execute, tool.clear_all_memory

    if is_async:
        @functools.wraps(execute)
        async def wrapped_execute(command):
            if command.command == "view":
                rec.check_view(command)
                return rec.after_view(command, await execute(command))
            probe = _probe_for(command)
            found = (await _afiles(tool.view, *probe)) if probe else []
            pending = rec.record(command, found)
            result = await execute(command)
            rec.settle(pending)
            return result

        @functools.wraps(clear_all)
        async def wrapped_clear_all():
            found = await _afiles(tool.view, _ROOT, True)
            pending = rec.record_clear(found)
            result = await clear_all()
            rec.settle(pending)
            return result
    else:
        @functools.wraps(execute)
        def wrapped_execute(command):
            if command.command == "view":
                rec.check_view(command)
                return rec.after_view(command, execute(command))
            probe = _probe_for(command)
            found = _files(tool.view, *probe) if probe else []
            pending = rec.record(command, found)
            result = execute(command)
            rec.settle(pending)
            return result

        @functools.wraps(clear_all)
        def wrapped_clear_all():
            found = _files(tool.view, _ROOT, True)
            pending = rec.record_clear(found)
            result = clear_all()
            rec.settle(pending)
            return result

    for fn in (wrapped_execute, wrapped_clear_all):
        setattr(fn, _MARK, True)
    tool.execute = wrapped_execute
    tool.clear_all_memory = wrapped_clear_all
    return tool


class DeleteAdapter:
    """Delete adapter for ``dod.excise``: deletes one file through the
    instrumented ``execute``, which records its tombstone.

    For an async tool, each delete runs in a new event loop
    (``asyncio.run``), so it can't be used from inside a running loop.
    """

    def __init__(self, tool: Any) -> None:
        if not getattr(getattr(tool, "execute", None), _MARK, False):
            raise TypeError("tool must be instrumented with dod's anthropic_memory.instrument")
        self.tool = tool

    def delete(self, record_id: str) -> bool:
        """Delete the file ``record_id`` names; return False if it wasn't there."""
        if isinstance(self.tool, BetaAsyncAbstractMemoryTool):
            return asyncio.run(self._adelete(record_id))
        # Probe through the uninstrumented view: not a retrieval, and it must
        # see quarantined files too.
        if _classify(record_id, _try_view(self.tool.view, record_id))[0] != "file":
            return False
        self.tool.execute(_delete_command(record_id))
        return True

    async def _adelete(self, record_id: str) -> bool:
        result = await _atry_view(self.tool.view, record_id)
        if _classify(record_id, result)[0] != "file":
            return False
        await self.tool.execute(_delete_command(record_id))
        return True


class _Recorder:
    """What the wrapper records around each command, shared by the sync and
    async tools."""

    def __init__(self, capture: Capture, store: str) -> None:
        self.capture = capture
        self.store = store
        self.quarantine = QuarantineCache(capture._ledger)
        self.warned = False

    def check_view(self, command: Any) -> None:
        """Raise the tool's not-found error if the viewed file is quarantined."""
        item = (self.store, record_id(command.path))
        if item in self.quarantine.current():
            count_blocked([item])
            raise ToolError(
                f"The path {command.path} does not exist. Please provide a valid path."
            )

    def after_view(self, command: Any, result: Any) -> Any:
        """Drop quarantined files from a listing and track what was viewed."""
        listing = _listing(command.path, result)
        if listing is not None:
            blocked = self.quarantine.current()
            dropped = [i for i, (e, is_dir) in enumerate(listing.entries)
                       if not is_dir and (self.store, e) in blocked]
            if dropped:
                count_blocked((self.store, listing.entries[i][0]) for i in dropped)
                result = listing.without(dropped)
                listing = _listing(command.path, result)
        key = _scope.current_key()
        if key is None:
            return result
        if listing is None:
            self.capture.on_retrieve(key, [(self.store, record_id(command.path))], "in_prompt")
        else:
            files = [(self.store, e) for e, is_dir in listing.entries if not is_dir]
            if files:
                self.capture.on_retrieve(key, files)
        return result

    def record(self, command: Any, found: List[str]) -> Optional[tuple]:
        """Record a command's writes before it runs; return what ``settle``
        records after it succeeds."""
        kind = command.command
        if kind == "create":
            rid = record_id(command.path)
            replaced = [(self.store, rid)] if rid in found else []
            return self._writes([(replaced, command.file_text, rid)])
        if kind in ("str_replace", "insert"):
            rid = record_id(command.path)
            payload = command.new_str if kind == "str_replace" else command.insert_text
            return self._writes([([(self.store, rid)], payload, rid)])
        if kind == "rename":
            old, new = record_id(command.old_path), record_id(command.new_path)
            moved = [f for f in found if f == old or f.startswith(old + "/")] or [old]
            return self._writes(
                [([(self.store, f)], "", new + f[len(old):]) for f in moved],
                tombstone_old=True,
            )
        if kind == "delete":
            return self._deletes(found)
        return None

    def record_clear(self, found: List[str]) -> Optional[tuple]:
        return self._deletes(found)

    def _writes(self, writes: List[Tuple[list, str, str]], tombstone_old: bool = False):
        key = _scope.current_key()
        agent_id, session_id = key if key is not None else (None, None)
        # A fresh key has no context, and no other write can share it.
        ctx_key = key if key is not None else object()
        if key is None and not self.warned:
            self.warned = True
            logger.warning("dod: memory write outside agent_task_scope; recorded with no parents")
        ids = self.capture.on_batch(
            ctx_key, self.store, [(r, c, _SOURCE, None) for r, c, _ in writes],
            agent_id=agent_id, session_id=session_id,
        )
        binds = [(w, rid) for w, (_, _, rid) in zip(ids, writes)]
        deletes = []
        if tombstone_old:
            deletes = [(r[0][1], "other", w) for w, (r, _, _) in zip(ids, writes)]
        return ctx_key, agent_id, session_id, binds, deletes

    def _deletes(self, found: List[str]) -> Optional[tuple]:
        if not found:
            return None
        key = _scope.current_key()
        agent_id, session_id = key if key is not None else (None, None)
        reason = deletion_reason()
        return object(), agent_id, session_id, [], [(f, reason, None) for f in found]

    def settle(self, pending: Optional[tuple]) -> None:
        """Record binds and tombstones after the command succeeded."""
        if pending is None:
            return
        ctx_key, agent_id, session_id, binds, deletes = pending
        # The command already succeeded: a ledger failure here must not
        # surface as a failed memory command. Log it; unbound writes stay
        # pending in the Capture and can be bound later.
        try:
            self.capture.on_batch(ctx_key, self.store, binds=binds, agent_id=agent_id,
                                  session_id=session_id, deletes=deletes)
        except Exception:
            logger.exception(
                "dod: recording %d bind(s) and %d tombstone(s) for store %r failed "
                "after the memory command succeeded; lineage for these files is incomplete",
                len(binds), len(deletes), self.store,
            )


class _Listing:
    """A parsed directory listing: entries are ``(record_id, is_dir)``, and
    ``lines[i + 2]`` is entry ``i``'s line."""

    __slots__ = ("lines", "entries")

    def __init__(self, lines: List[str], entries: List[Tuple[str, bool]]) -> None:
        self.lines = lines
        self.entries = entries

    def without(self, drop: List[int]) -> str:
        skip = {i + 2 for i in drop}
        return "\n".join(line for n, line in enumerate(self.lines) if n not in skip)


def _listing(path: str, result: Any) -> Optional[_Listing]:
    """Parse ``result`` as a ``view`` listing of directory ``path``, or None
    if it isn't one (a file's content, or another format)."""
    if not isinstance(result, str) or result.startswith("Here's the content of"):
        return None
    lines = result.split("\n")
    if len(lines) < 2 or lines[1].partition("\t")[2] != path:
        return None
    prefix = path + "/"
    entries = []
    for line in lines[2:]:
        entry = line.partition("\t")[2]
        if not entry.startswith(prefix):
            return None
        entries.append((record_id(entry), entry.endswith("/")))
    return _Listing(lines, entries)


def _classify(path: str, result: Any) -> Tuple[Optional[str], List[Tuple[str, bool]]]:
    """``("file", [])``, ``("dir", entries)``, or ``(None, [])`` if the view
    failed."""
    if result is None:
        return None, []
    listing = _listing(path, result)
    if listing is None:
        return "file", []
    return "dir", listing.entries


def _probe_for(command: Any) -> Optional[Tuple[str, bool]]:
    """The path a command must probe before it runs, and whether to list a
    directory's files recursively."""
    kind = command.command
    if kind == "create":
        return command.path, False
    if kind == "delete":
        return command.path, True
    if kind == "rename":
        return command.old_path, True
    return None


def _view_command(path: str) -> BetaMemoryTool20250818ViewCommand:
    return BetaMemoryTool20250818ViewCommand(command="view", path=path)


def _delete_command(path: str) -> BetaMemoryTool20250818DeleteCommand:
    return BetaMemoryTool20250818DeleteCommand(command="delete", path=path)


def _try_view(view: Callable, path: str) -> Any:
    try:
        return view(_view_command(path))
    except ToolError:
        return None
    except Exception:
        logger.warning("dod: probing memory path %r failed; treating it as missing", path,
                       exc_info=True)
        return None


async def _atry_view(view: Callable, path: str) -> Any:
    try:
        return await view(_view_command(path))
    except ToolError:
        return None
    except Exception:
        logger.warning("dod: probing memory path %r failed; treating it as missing", path,
                       exc_info=True)
        return None


def _expand(path: str, result: Any, deep: bool, out: List[str]) -> List[str]:
    """Add ``path``'s files to ``out``; return directories still to view."""
    kind, entries = _classify(path, result)
    if kind == "file":
        out.append(record_id(path))
        return []
    if kind is None or not deep:
        return []
    out.extend(e for e, is_dir in entries if not is_dir)
    names = [e for e, _ in entries]
    # A directory whose children aren't listed is past the listing's depth
    # (or empty): view it in turn.
    return [d for d, is_dir in entries
            if is_dir and not any(n.startswith(d + "/") for n in names)]


def _files(view: Callable, path: str, deep: bool) -> List[str]:
    """Record IDs of the files at ``path``: the file itself, or (if
    ``deep``) every file under the directory."""
    out: List[str] = []
    todo, seen = [path], set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        todo += _expand(p, _try_view(view, p), deep, out)
    return list(dict.fromkeys(out))


async def _afiles(view: Callable, path: str, deep: bool) -> List[str]:
    out: List[str] = []
    todo, seen = [path], set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        todo += _expand(p, await _atry_view(view, p), deep, out)
    return list(dict.fromkeys(out))
