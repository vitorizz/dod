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

from dod import audit

from . import correction
from .lineage import Lineage
from .world import CORRECTION, CUSTOMERS, STORE_LABELS, STORY, World

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
                "stores": STORE_LABELS, "customers": CUSTOMERS,
                "story": [{"customer": c, "message": m} for c, m in STORY],
                "correction_example": CORRECTION, "corrections": w.corrections}

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
            elif action == "report":
                result = _report(w, lin, box)
            else:
                return JSONResponse({"error": f"unknown action {action}"}, status_code=404)
        return JSONResponse({"result": result, **snapshot()})

    async def correct(request: Request):
        """The correction flow: candidates, then a plan, then apply."""
        body = await request.json()
        step, text = request.path_params["step"], str(body.get("text", "")).strip()
        if not text:
            return JSONResponse({"error": "say what the right fact is"}, status_code=400)
        w = state["world"]
        async with lock:
            if step == "candidates":
                result = correction.candidates(w, text)
            elif step == "plan":
                result = correction.plan(w, text, list(body.get("boxes") or []))
            elif step == "apply":
                result = await correction.apply(w, text, list(body.get("boxes") or []))
            else:
                return JSONResponse({"error": f"unknown step {step}"}, status_code=404)
        if "error" in result:
            return JSONResponse(result, status_code=400)
        return JSONResponse({"result": result, **snapshot()})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/state", get_state),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/story", story, methods=["POST"]),
            Route("/api/reset", reset, methods=["POST"]),
            Route("/api/box/{action}", box_action, methods=["POST"]),
            Route("/api/correction/{step}", correct, methods=["POST"]),
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
