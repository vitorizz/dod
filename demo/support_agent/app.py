"""The demo's web app: a chat with the support agent, next to the dod console.

    python -m demo.support_agent.app                       # scripted model, in-memory store
    python -m demo.support_agent.app --model anthropic/claude-sonnet-5-5 \\
        --pg "postgresql://postgres@localhost:5432/demo"
    python -m demo.support_agent.app --model openai/qwen3:8b   # Ollama on localhost

Then open http://localhost:8000.
"""

import argparse
import asyncio
import contextlib
import os
import tempfile
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

import sqlite3

import dod
from dod import audit

from .lineage import Lineage
from .world import STORE_LABELS, STORY, World

STATIC = Path(__file__).parent / "static"


def create_app(model: str = "scripted", pg_uri: str = None) -> Starlette:
    state = {"world": None}
    lock = asyncio.Lock()

    async def fresh() -> World:
        if state["world"] is not None:
            await state["world"].stop()
        world = World(tempfile.mkdtemp(prefix="dod-demo-"), model, pg_uri)
        await world.start()
        state["world"] = world
        return world

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await fresh()
        yield
        await state["world"].stop()

    def snapshot() -> dict:
        w = state["world"]
        v = audit.verify(w.ledger)
        return {**Lineage(w).view(), "log": w.log, "model": w.llm.name, "backend": w.backend,
                "audit": {"ok": v.ok, "entries": v.entries, "head": v.head},
                "stores": STORE_LABELS, "customers": ["ana", "ben", "chloe", "dev", "eli"],
                "story": [{"customer": c, "message": m} for c, m in STORY]}

    async def index(request: Request):
        return FileResponse(STATIC / "index.html")

    async def get_state(request: Request):
        return JSONResponse(snapshot())

    async def chat(request: Request):
        body = await request.json()
        async with lock:
            entry = await state["world"].chat(body["customer"], body["message"])
        return JSONResponse({"entry": entry, **snapshot()})

    async def story(request: Request):
        async with lock:
            await fresh()
            await state["world"].run_story()
        return JSONResponse(snapshot())

    async def reset(request: Request):
        async with lock:
            await fresh()
        return JSONResponse(snapshot())

    async def box_action(request: Request):
        body = await request.json()
        action, box = request.path_params["action"], body["box"]
        w = state["world"]
        async with lock:
            lin = Lineage(w)
            if box not in lin.boxes:
                return JSONResponse({"error": f"unknown box {box}"}, status_code=404)
            if action == "trace":
                result = lin.trace(box)
            elif action == "blast":
                result = lin.blast(box)
            elif action == "compare":
                result = lin.compare(box)
            elif action == "excise":
                before = lin.compare(box)
                for item in lin._items_of(box):
                    dod.excise(w.ledger, *item)
                result = {"compare": before}
            elif action == "regenerate":
                before = lin.compare(box)
                pairs = []
                for item in lin._items_of(box):
                    r = await w.regenerate(*item)
                    for old, new in r["replaced"]:
                        pairs.append(_pair(w, old[1], new[1]))
                result = {"compare": before, "pairs": pairs}
            elif action == "report":
                result = _report(w, lin, box)
            else:
                return JSONResponse({"error": f"unknown action {action}"}, status_code=404)
        return JSONResponse({"result": result, **snapshot()})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/state", get_state),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/story", story, methods=["POST"]),
            Route("/api/reset", reset, methods=["POST"]),
            Route("/api/box/{action}", box_action, methods=["POST"]),
        ],
        lifespan=lifespan,
    )


def _report(w: World, lin: Lineage, box: str) -> dict:
    """Proof of removal for a box: every item it reached, with what happened
    to it and the audit entry that recorded it, plus the chain's check."""
    lin.view()  # fills each box's label
    labels = {tuple(b["item"]): b["label"] for b in lin.boxes.values() if b["item"]}
    items, docs = {}, []
    for item in lin._items_of(box):
        rep = audit.report(w.ledger, *item)
        docs.append(audit.to_markdown(rep))
        for i in rep["items"]:
            key = (i["store"], i["record_id"])
            acts = [{"what": a["action"], "ts": a["ts"], "entry": (a["chain"] or {}).get("seq")}
                    for a in i["quarantine"]]
            acts += [{"what": "deleted", "ts": d["ts"], "entry": (d["chain"] or {}).get("seq")}
                     for d in i["deletions"]]
            items[key] = {"store": i["store"], "record_id": i["record_id"],
                          "label": labels.get(key) or ("Lookup result the agent read"
                                                       if i["record_id"].startswith("call:")
                                                       else i["record_id"]),
                          "status": i["status"],
                          "actions": acts}
    return {"items": sorted(items.values(), key=lambda i: (i["store"], i["label"])),
            "chain": rep["chain"], "markdown": "\n".join(docs)}


def _pair(w: World, old: str, new: str) -> dict:
    """A regenerated reply next to the one it replaces."""
    db = sqlite3.connect(w._crm_db)
    try:
        rows = {r: db.execute("SELECT customer, question, text FROM replies WHERE id = ?",
                              (r,)).fetchone() for r in (old, new)}
    finally:
        db.close()
    customer, question, before = rows[old]
    return {"customer": customer, "question": question, "before": before,
            "after": rows[new][2], "old": old, "new": new}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=os.environ.get("DEMO_MODEL", "scripted"),
                   help="scripted, anthropic/<model> or openai/<model>")
    p.add_argument("--pg", default=os.environ.get("DEMO_PG_URI"),
                   help="Postgres URI (pgvector); in-memory store if omitted")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(create_app(args.model, args.pg), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
