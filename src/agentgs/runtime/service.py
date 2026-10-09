"""Async task supervision with shared admission, deadlines and cancellation."""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.async_utils import drain, invoke
from agentgs.contracts.models import (
    TERMINAL,
    AgentResult,
    TaskRecord,
    TaskSpec,
    TaskStatus,
    ToolError,
)
from agentgs.runtime.concurrency import RuntimeCoordinator
from agentgs.runtime.resources import tree_rss
from agentgs.runtime.storage.store import TaskStore
from agentgs.runtime.subprocesses import spawn, stop


class TaskService:
    def __init__(self, root: Path, settings=None):
        self.root = root.resolve()
        self.settings = settings or RuntimeSettings()
        self.store = TaskStore(root / "outputs/runtime", initialize=False)
        self.coordinator = RuntimeCoordinator(root)
        self.jobs, self.processes = {}, {}
        self.closed = False
        self._submission_lock = asyncio.Lock()
        self._rss_lock = asyncio.Lock()
        self._rss_snapshot, self._rss_snapshot_at = {}, 0.0
        from agentgs.runtime.embeddings import EmbeddingRuntime

        self.embeddings = EmbeddingRuntime(self.root)

    async def __aenter__(self):
        await self.store.initialize()
        await self.coordinator.initialize()
        return self

    async def __aexit__(self, *args):
        await self.close()

    def _memory_usage(self, pid):
        now = time.monotonic()
        if now - self._rss_snapshot_at >= 0.05:
            self._rss_snapshot = {p.pid: tree_rss(p.pid) for p in list(self.processes.values())}
            self._rss_snapshot_at = time.monotonic()
        elif pid not in self._rss_snapshot:
            self._rss_snapshot[pid] = tree_rss(pid)
        active = {p.pid for p in list(self.processes.values())}
        return self._rss_snapshot.get(pid, 0), sum(
            rss for key, rss in self._rss_snapshot.items() if key in active
        )

    async def submit(self, task_input, **limits):
        async with self._submission_lock:
            if self.closed:
                raise RuntimeError("Task service is closed")
            spec = TaskSpec(
                input=task_input,
                timeout_seconds=limits.get(
                    "timeout_seconds", self.settings.default_timeout_seconds
                ),
                rss_limit_mb=limits.get("rss_limit_mb", self.settings.default_rss_limit_mb),
            )
            record = TaskRecord(spec=spec)
            try:
                await self.store.reserve(record, self.settings.max_pending_tasks)
            except asyncio.CancelledError:

                async def rollback():
                    try:
                        stored = await self.store.get(spec.id)
                    except KeyError:
                        return
                    stored.status = TaskStatus.CANCELLED
                    stored.error = ToolError(code="cancelled", message="Cancelled during admission")
                    await self.store.save(stored)
                    await self.store.emit(
                        spec.id, "task.cancelled", {"error": stored.error.model_dump()}
                    )

                await drain(asyncio.create_task(rollback()))
                raise
            self.jobs[spec.id] = asyncio.create_task(self._run(spec))
            return record

    async def get(self, task_id):
        return await self.store.get(task_id)

    async def list(self):
        return await self.store.list()

    async def events(self, task_id, after=0):
        return await self.store.events(task_id, after)

    async def wait(self, task_id):
        job = self.jobs.get(task_id)
        if job:
            await asyncio.shield(job)
        while True:
            record = await self.get(task_id)
            if record.status in TERMINAL:
                return record
            control = await self.store.control(task_id)
            if control and not await invoke(
                RuntimeCoordinator._alive, control["owner_pid"], control["owner_identity"]
            ):
                await self.store.recover()
                return await self.get(task_id)
            await asyncio.sleep(0.02)

    async def cancel(self, task_id):
        record = await self.get(task_id)
        if record.status in TERMINAL:
            job = self.jobs.get(task_id)
            if job:
                await asyncio.shield(job)
            return record
        await self.store.request_cancel(task_id)
        job = self.jobs.get(task_id)
        if job:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
            record = await self.get(task_id)
            # A coroutine cancelled before its first instruction cannot run finally.
            if record.status not in TERMINAL:
                record.status = TaskStatus.CANCELLED
                record.error = ToolError(code="cancelled", message="Cancelled before worker start")
                await self.store.save(record)
                await self.store.emit(
                    task_id, "task.cancelled", {"error": record.error.model_dump()}
                )
                self.jobs.pop(task_id, None)
            return record
        return await self.wait(task_id)

    async def close(self):
        async with self._submission_lock:
            self.closed = True
        await drain(asyncio.create_task(self._close_jobs()))

    async def _close_jobs(self):
        jobs = list(self.jobs.values())
        await asyncio.gather(*(self.cancel(key) for key in list(self.jobs)), return_exceptions=True)
        await asyncio.gather(*jobs, return_exceptions=True)
        await self.embeddings.close()

    async def _run(self, spec):
        record = await self.get(spec.id)
        process = reader = None
        admitted = False
        slot = self.coordinator.slot("tasks", self.settings.max_concurrent_tasks)
        target = asyncio.current_task()

        async def cancellation():
            while True:
                control = await self.store.control(spec.id)
                if control and control["cancelled"]:
                    target.cancel()
                    return
                await asyncio.sleep(0.02)

        watcher = asyncio.create_task(cancellation())
        try:
            # The deadline includes admission and preparation, not only worker time.
            elapsed = max(
                0,
                (
                    datetime.now(timezone.utc) - datetime.fromisoformat(record.created_at)
                ).total_seconds(),
            )
            async with asyncio.timeout(max(0, spec.timeout_seconds - elapsed)):
                await slot.__aenter__()
                admitted = True
                record.status = TaskStatus.PREPARING
                await self.store.save(record)
                work = self.root / "outputs/tasks" / spec.id
                request = work / "request.json"
                embedding_endpoint = None
                if not spec.input.calls:
                    from agentgs.config.algorithm import load_algorithm, resolve_algorithm

                    options = resolve_algorithm(await invoke(load_algorithm, self.root), spec.input)
                    if (
                        options.planning_mode != "data_only"
                        and options.candidate_pool == "retrieved"
                        and await invoke((self.root / "embedding/tools.npz").is_file)
                        and await invoke(
                            (self.root / options.index_path / "graph_store.json").is_file
                        )
                    ):
                        embedding_endpoint = await self.embeddings.endpoint(options)

                def prepare():
                    work.mkdir(parents=True, exist_ok=False)
                    request.write_text(
                        json.dumps(
                            {
                                "root": str(self.root),
                                "spec": spec.model_dump(mode="json"),
                                "runtime": self.settings.model_dump(),
                                "embedding_endpoint": embedding_endpoint,
                            }
                        ),
                        encoding="utf-8",
                    )

                await invoke(prepare)
                interpreter = self.settings.worker_python or sys.executable
                if self.settings.worker_python:
                    interpreter = str((self.root / self.settings.worker_python).absolute())
                    if not await invoke(Path(interpreter).is_file):
                        raise FileNotFoundError(
                            f"Configured worker interpreter missing: {interpreter}"
                        )
                elif await invoke((self.root / "tools/envs/reproduction/bin/python").exists):
                    interpreter = str(self.root / "tools/envs/reproduction/bin/python")
                env = {
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                }
                error_log = await invoke((work / "stderr.log").open, "wb")
                try:
                    process = await spawn(
                        interpreter,
                        "-m",
                        "agentgs.runtime.worker",
                        "--request",
                        str(request),
                        cwd=self.root,
                        env=env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=error_log,
                        start_new_session=os.name != "nt",
                        limit=2 * 1024 * 1024,
                    )
                finally:
                    await invoke(error_log.close)
                self.processes[spec.id] = process
                await self.store.worker(spec.id, process.pid)
                record.status = TaskStatus.RUNNING
                await self.store.save(record)
                await self.store.emit(
                    spec.id,
                    "task.running",
                    {
                        "pid": process.pid,
                        "memory_enforcement": "rss_watchdog",
                        "sandbox_scope": "code_steps",
                    },
                )

                async def consume():
                    while line := await process.stdout.readline():
                        event = await invoke(json.loads, line)
                        await self.store.emit(spec.id, event["type"], event["payload"])

                reader = asyncio.create_task(consume())
                while process.returncode is None:
                    async with self._rss_lock:
                        rss, _ = await invoke(self._memory_usage, process.pid)
                    total = await self.store.worker(spec.id, process.pid, rss)
                    record.peak_rss_bytes = max(record.peak_rss_bytes, rss)
                    if (
                        rss > spec.rss_limit_mb * 1024 * 1024
                        or total > self.settings.max_total_rss_mb * 1024 * 1024
                    ):
                        record.error = ToolError(
                            code="resource_limit", message="RSS watchdog limit exceeded"
                        )
                        await stop(process)
                        break
                    if reader.done() and reader.exception():
                        record.error = ToolError(
                            code="worker_protocol", message="Invalid worker event stream"
                        )
                        await stop(process)
                        break
                    await asyncio.sleep(0.05)
                await process.wait()
                if reader.done() and reader.exception():
                    record.error = record.error or ToolError(
                        code="worker_protocol", message="Invalid worker event stream"
                    )
                else:
                    await reader
                await self._read_result(record, work, process.returncode)
        except asyncio.CancelledError:
            record.status = TaskStatus.CANCELLED
            record.error = ToolError(code="cancelled", message="Task cancelled")
        except TimeoutError:
            record.status = TaskStatus.FAILED
            record.error = ToolError(
                code="timeout", message="Task admission/execution time budget exceeded"
            )
        except Exception as exc:
            record.status = TaskStatus.FAILED
            record.error = ToolError(
                code="runtime_error",
                message=f"{type(exc).__name__}: {exc}"
                if isinstance(exc, (ValueError, FileNotFoundError))
                else type(exc).__name__,
            )
        finally:
            await drain(
                asyncio.create_task(
                    self._finish_run(record, process, reader, watcher, slot, admitted)
                )
            )

    async def _finish_run(self, record, process, reader, watcher, slot, admitted):
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        try:
            await self._finalize(record, process, reader)
        finally:
            try:
                if admitted:
                    await slot.__aexit__(None, None, None)
            finally:
                self.jobs.pop(record.spec.id, None)

    async def _read_result(self, record, work, returncode):
        result_path = work / "result.json"
        if await invoke(result_path.exists):
            record.result = await invoke(
                lambda: AgentResult.model_validate_json(result_path.read_text())
            )
        if record.error:
            record.status = TaskStatus.FAILED
        elif returncode == 0 and record.result is not None:
            record.status = TaskStatus.SUCCEEDED
            if record.result.completion_status != "Completed":
                record.status = TaskStatus.FAILED
                record.error = ToolError(
                    code="task_incomplete",
                    message="Algorithm returned Not Completed; answer and messages are preserved",
                )
        else:
            path = work / "error.json"
            record.error = (
                await invoke(lambda: ToolError.model_validate_json(path.read_text()))
                if await invoke(path.exists)
                else ToolError(code="worker_exit", message=f"Worker exit {returncode}")
            )
            record.status = TaskStatus.FAILED

    async def _finalize(self, record, process, reader):
        if process:
            await stop(process, grace=2)
        if reader:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        self.processes.pop(record.spec.id, None)
        await self.store.worker(record.spec.id, None)
        partial = self.root / "outputs/tasks" / record.spec.id / "result.json"
        if record.result is None and await invoke(partial.is_file):
            try:
                record.result = await invoke(
                    lambda: AgentResult.model_validate_json(partial.read_text())
                )
            except (ValueError, OSError):
                pass
        await self.store.save(record)
        await self.store.emit(
            record.spec.id,
            "task." + record.status.value,
            {
                "error": record.error.model_dump() if record.error else None,
                "peak_rss_bytes": record.peak_rss_bytes,
            },
        )
