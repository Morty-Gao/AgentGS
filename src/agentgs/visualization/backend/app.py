import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agentgs.config.paths import within
from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TERMINAL, TaskInput
from agentgs.messages import Message
from agentgs.runtime.service import TaskService
from agentgs.runtime.storage.store import QueueFull

TERMINAL_NAMES = {status.value for status in TERMINAL}


def create_app(root: Path, settings: RuntimeSettings | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.tasks = TaskService(root, settings=settings)
        yield
        await app.state.tasks.close()

    app = FastAPI(title="AgentGS", lifespan=lifespan)

    def service(request):
        return request.app.state.tasks

    async def find(request, task_id):
        try:
            return await service(request).get(task_id)
        except KeyError:
            raise HTTPException(404, "Unknown task") from None

    @app.get("/api/tasks")
    async def tasks(request: Request):
        return await service(request).list()

    @app.post("/api/tasks", status_code=202)
    async def submit(task: TaskInput, request: Request):
        try:
            return await service(request).submit(task)
        except QueueFull as exc:
            raise HTTPException(429, str(exc)) from None

    @app.get("/api/tasks/{task_id}")
    async def get(task_id: str, request: Request):
        return await find(request, task_id)

    @app.get("/api/tasks/{task_id}/messages", response_model=list[Message])
    async def messages(task_id: str, request: Request):
        record = await find(request, task_id)
        return record.result.messages if record.result else []

    @app.post("/api/tasks/{task_id}/cancel")
    async def cancel(task_id: str, request: Request):
        await find(request, task_id)
        try:
            return await service(request).cancel(task_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/tasks/{task_id}/events")
    async def events(task_id: str, request: Request, after: int = 0):
        await find(request, task_id)
        header = request.headers.get("last-event-id", "")
        cursor = max(after, int(header)) if header.isdigit() else after

        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                for event in await service(request).events(task_id, cursor):
                    cursor = event.sequence
                    yield f"id: {cursor}\ndata: {event.model_dump_json()}\n\n"
                if (await find(request, task_id)).status in TERMINAL:
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.4)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/api/tasks/{task_id}/artifacts/{artifact_id}")
    async def artifact(task_id: str, artifact_id: str, request: Request):
        record = await find(request, task_id)
        refs = (
            [a for result in record.result.tool_results for a in result.artifacts]
            if record.result
            else []
        )
        ref = next((a for a in refs if a.id == artifact_id), None)
        if ref is None:
            raise HTTPException(404, "Unknown artifact")
        path = within(root / "outputs" / "tasks" / task_id / "artifacts", ref.path)
        return FileResponse(path, media_type=ref.media_type)

    @app.get("/api/skills")
    def skills():
        from agentgs.skills.registry import builtin_skills

        return [s.spec for s in builtin_skills().values()]

    @app.get("/api/tools")
    def tools():
        from agentgs.runtime.tools.registry import builtin_tools

        return [t.spec for t in builtin_tools().values()]

    dist = Path(__file__).parents[1] / "frontend" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(dist / "index.html")

    return app
