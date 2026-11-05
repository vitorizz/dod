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

from .world import STORY, World

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
        return {**w.graph_view(), "log": w.log,
                "customers": ["ana", "ben", "chloe", "dev", "eli"],
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

    async def item_action(request: Request):
        body = await request.json()
        action, store, rid = request.path_params["action"], body["store"], body["record_id"]
        w = state["world"]
        async with lock:
            try:
                if action == "trace":
                    result = w.trace_view(store, rid)
                elif action == "blast":
                    result = w.blast_view(store, rid)
                elif action == "compare":
                    result = w.compare(store, rid)
                elif action == "excise":
                    result = w.excise(store, rid)
                elif action == "regenerate":
                    result = await w.regenerate(store, rid)
                else:
                    return JSONResponse({"error": f"unknown action {action}"}, status_code=404)
            except KeyError:
                return JSONResponse({"error": "no recorded write for that item"},
                                    status_code=404)
        return JSONResponse({"result": result, **snapshot()})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/state", get_state),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/story", story, methods=["POST"]),
            Route("/api/reset", reset, methods=["POST"]),
            Route("/api/item/{action}", item_action, methods=["POST"]),
        ],
        lifespan=lifespan,
    )


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
