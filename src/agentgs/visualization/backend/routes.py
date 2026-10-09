"""Typed task API, resumable runtime events and registered artifact downloads."""

import asyncio

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse

from agentgs.config.algorithm import load_algorithm
from agentgs.config.paths import within
from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import TERMINAL, TaskInput, TaskRecord, TaskStatus, ToolSpec
from agentgs.messages import Conversation, LogEntry
from agentgs.runtime.service import TaskService
from agentgs.runtime.storage.store import QueueFull
from agentgs.skills.base import SkillSpec

from .models import ConsoleConfig, TaskPage
from .queries import task_page, with_checkpoint

router = APIRouter()


def service(request: Request) -> TaskService:
    return request.app.state.tasks


async def find(request: Request, task_id: str) -> TaskRecord:
    try:
        record = await service(request).get(task_id)
    except KeyError:
        raise HTTPException(404, "Unknown task") from None
    return await with_checkpoint(service(request).root, record)


@router.get("/config", response_model=ConsoleConfig)
async def config(request: Request):
    tasks = service(request)
    return ConsoleConfig(algorithm=await invoke(load_algorithm, tasks.root), runtime=tasks.settings)


@router.get("/tasks", response_model=TaskPage)
async def tasks(
    request: Request,
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: TaskStatus | None = None,
    search: str = Query("", max_length=500),
):
    return await task_page(
        service(request).store, limit=limit, offset=offset, status=status, search=search.strip()
    )


@router.post("/tasks", status_code=202, response_model=TaskRecord)
async def submit(task: TaskInput, request: Request):
    if not task.question.strip() and not task.calls:
        raise HTTPException(422, "Provide a question, user message, or explicit tool calls")
    try:
        return await service(request).submit(task)
    except QueueFull as exc:
        raise HTTPException(429, str(exc)) from None


@router.get("/tasks/{task_id}", response_model=TaskRecord)
async def get(task_id: str, request: Request):
    return await find(request, task_id)


@router.get("/tasks/{task_id}/messages", response_model=list[LogEntry])
async def messages(task_id: str, request: Request):
    record = await find(request, task_id)
    return record.result.messages if record.result else record.spec.input.messages


@router.get("/tasks/{task_id}/conversation", response_model=Conversation)
async def conversation(task_id: str, request: Request):
    record = await find(request, task_id)
    return Conversation(
        messages=record.result.messages if record.result else record.spec.input.messages
    )


@router.get("/tasks/{task_id}/export")
async def export(task_id: str, request: Request):
    record = await find(request, task_id)
    return Response(
        record.model_dump_json(indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="task.json"'},
    )


@router.post("/tasks/{task_id}/cancel", response_model=TaskRecord)
async def cancel(task_id: str, request: Request):
    await find(request, task_id)
    try:
        return await service(request).cancel(task_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@router.get("/tasks/{task_id}/events")
async def events(task_id: str, request: Request, after: int = Query(0, ge=0, le=2**63 - 1)):
    await find(request, task_id)
    header = request.headers.get("last-event-id", "")
    cursor = after
    if header.isascii() and header.isdigit():
        if len(header) > 19 or int(header) > 2**63 - 1:
            raise HTTPException(400, "Event cursor is out of range")
        cursor = max(after, int(header))

    async def stream():
        nonlocal cursor
        while not await request.is_disconnected():
            # Read status before draining to retain events committed with terminal state.
            record = await service(request).get(task_id)
            if record.status in TERMINAL:
                job = service(request).jobs.get(task_id)
                if job is not None:
                    # Final status is saved before the supervisor appends its last event.
                    # Disconnecting a viewer must not cancel the supervisor's cleanup.
                    await asyncio.shield(asyncio.gather(job, return_exceptions=True))
            for event in await service(request).events(task_id, cursor):
                cursor = event.sequence
                yield f"id: {cursor}\ndata: {event.model_dump_json()}\n\n"
            if record.status in TERMINAL:
                # Explicit EOF also closes clients whose cursor is already exhausted.
                yield f'event: end\ndata: {{"status":"{record.status.value}"}}\n\n'
                break
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def find_artifact(task_id: str, artifact_id: str, request: Request):
    record = await find(request, task_id)
    refs = [a for r in record.result.tool_results for a in r.artifacts] if record.result else []
    ref = next((a for a in refs if a.id == artifact_id), None)
    if ref is None:
        raise HTTPException(404, "Unknown artifact")
    try:
        root = within(service(request).root / "outputs/tasks", task_id + "/artifacts")
        path = within(root, ref.path)
    except ValueError:
        raise HTTPException(404, "Artifact is outside its task") from None
    if not await invoke(path.is_file):
        raise HTTPException(404, "Artifact file is missing")
    return ref, path


@router.get("/tasks/{task_id}/artifacts/{artifact_id}")
async def artifact(task_id: str, artifact_id: str, request: Request):
    ref, path = await find_artifact(task_id, artifact_id, request)
    return FileResponse(
        path,
        filename=path.name,
        media_type=ref.media_type,
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.get("/tasks/{task_id}/artifacts/{artifact_id}/preview")
async def preview(task_id: str, artifact_id: str, request: Request):
    ref, path = await find_artifact(task_id, artifact_id, request)
    image_types = {"image/png", "image/jpeg", "image/webp", "image/gif"}
    is_image = ref.media_type in image_types
    if not is_image and ref.media_type not in {
        "application/json",
        "application/geo+json",
        "text/plain",
        "text/csv",
        "text/tab-separated-values",
        "text/markdown",
    }:
        raise HTTPException(415, "Preview is unavailable for this format; download the file")
    limit = 16 * 1024 * 1024 if is_image else 256 * 1024

    def read():
        with path.open("rb") as stream:
            return stream.read(limit + 1)

    try:
        content = await invoke(read)
    except OSError:
        raise HTTPException(404, "Artifact file is missing") from None
    if len(content) > limit:
        raise HTTPException(413, "Artifact exceeds the preview size limit; download the file")
    headers = {"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'"}
    if is_image:
        return Response(content, media_type=ref.media_type, headers=headers)
    return PlainTextResponse(content.decode("utf-8", errors="replace"), headers=headers)


@router.get("/skills", response_model=list[SkillSpec])
def skills():
    from agentgs.skills.registry import builtin_skills

    return [s.spec for s in builtin_skills().values()]


@router.get("/tools", response_model=list[ToolSpec])
def tools():
    from agentgs.runtime.tools.registry import builtin_tools

    return [t.spec for t in builtin_tools().values()]
