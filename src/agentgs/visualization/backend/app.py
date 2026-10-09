"""Application lifecycle and static console delivery."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from agentgs.config.settings import RuntimeSettings
from agentgs.runtime.service import TaskService

from .routes import router


def create_app(root: Path, settings: RuntimeSettings | None = None) -> FastAPI:
    root = root.resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with TaskService(root, settings=settings) as tasks:
            app.state.tasks = tasks
            yield

    app = FastAPI(title="AgentGS Console", lifespan=lifespan)
    app.include_router(router, prefix="/api")
    dist = Path(__file__).parents[1] / "frontend" / "dist"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        if (dist / "index.html").is_file():
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})
        return HTMLResponse(
            "<h1>AgentGS Console</h1><p>Frontend build is missing. Run "
            "<code>npm ci &amp;&amp; npm run build</code> in "
            "<code>src/agentgs/visualization/frontend</code>, then restart the server.</p>"
            '<p>The API is available at <a href="/docs">/docs</a>.</p>',
            status_code=503,
        )

    return app
