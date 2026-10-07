import asyncio
import json
import os
import sys
import time
from pathlib import Path

from agentgs.config.settings import load_runtime
from agentgs.contracts.models import (
    TERMINAL,
    AgentResult,
    TaskRecord,
    TaskSpec,
    TaskStatus,
    ToolError,
)
from agentgs.runtime.resources import kill_tree, tree_rss
from agentgs.runtime.storage.store import TaskStore


class TaskService:
    def __init__(self, root: Path, settings=None):
        self.root = root.resolve()
        self.settings = settings or load_runtime(root)
        self.store = TaskStore(root / "outputs" / "runtime")
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrent_tasks)
        self.jobs = {}
        self.processes = {}
        self.closed = False

    def submit(self, task_input, **limits):
        if self.closed:
            raise RuntimeError("Task service is closed")
        spec = TaskSpec(
            input=task_input,
            timeout_seconds=limits.get("timeout_seconds", self.settings.default_timeout_seconds),
            rss_limit_mb=limits.get("rss_limit_mb", self.settings.default_rss_limit_mb),
        )
        record = TaskRecord(spec=spec)
        self.store.save(record)
        self.store.emit(spec.id, "task.queued", {})
        self.jobs[spec.id] = asyncio.create_task(self._run(spec))
        return record

    def get(self, task_id):
        return self.store.get(task_id)

    def list(self):
        return self.store.list()

    def events(self, task_id, after=0):
        return self.store.events(task_id, after)

    async def wait(self, task_id):
        job = self.jobs.get(task_id)
        if job:
            await asyncio.shield(job)
        return self.get(task_id)

    async def cancel(self, task_id):
        job = self.jobs.get(task_id)
        if not job:
            record = self.get(task_id)
            if record.status not in TERMINAL:
                raise ValueError("Task belongs to another runtime instance")
            return record
        if not job.done():
            job.cancel()
        await asyncio.gather(job, return_exceptions=True)
        record = self.get(task_id)
        if record.status not in TERMINAL:
            record.status = TaskStatus.CANCELLED
            record.error = ToolError(code="cancelled", message="Cancelled before worker start")
            self.store.save(record)
            self.store.emit(task_id, "task.cancelled", {})
        return record

    async def close(self):
        self.closed = True
        await asyncio.gather(
            *(self.cancel(task_id) for task_id in list(self.jobs)), return_exceptions=True
        )

    async def _run(self, spec):
        record = self.get(spec.id)
        process = None
        reader = None
        try:
            async with self.semaphore:
                record.status = TaskStatus.PREPARING
                self.store.save(record)
                work = self.root / "outputs" / "tasks" / spec.id
                work.mkdir(parents=True, exist_ok=False)
                request = work / "request.json"
                request.write_text(
                    json.dumps(
                        {
                            "root": str(self.root),
                            "spec": spec.model_dump(mode="json"),
                            "runtime": self.settings.model_dump(),
                        }
                    ),
                    encoding="utf-8",
                )
                env = {
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                }
                with (work / "stderr.log").open("wb") as error_log:
                    process = await asyncio.create_subprocess_exec(
                        sys.executable,
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
                    self.processes[spec.id] = process
                    record.status = TaskStatus.RUNNING
                    self.store.save(record)
                    self.store.emit(
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
                            event = json.loads(line)
                            self.store.emit(spec.id, event["type"], event["payload"])

                    reader = asyncio.create_task(consume())
                    started = time.monotonic()
                    while process.returncode is None:
                        rss = tree_rss(process.pid)
                        record.peak_rss_bytes = max(record.peak_rss_bytes, rss)
                        total = sum(tree_rss(p.pid) for p in self.processes.values())
                        if (
                            rss > spec.rss_limit_mb * 1024 * 1024
                            or total > self.settings.max_total_rss_mb * 1024 * 1024
                        ):
                            record.error = ToolError(
                                code="resource_limit", message="RSS watchdog limit exceeded"
                            )
                            kill_tree(process.pid)
                            break
                        if time.monotonic() - started > spec.timeout_seconds:
                            record.error = ToolError(
                                code="timeout", message="Task wall-time budget exceeded"
                            )
                            kill_tree(process.pid)
                            break
                        if reader.done() and reader.exception():
                            record.error = ToolError(
                                code="worker_protocol", message="Invalid worker event stream"
                            )
                            kill_tree(process.pid)
                            break
                        await asyncio.sleep(0.05)
                    await process.wait()
                    await reader
                if record.error:
                    record.status = TaskStatus.FAILED
                elif process.returncode == 0 and (work / "result.json").exists():
                    record.result = AgentResult.model_validate_json(
                        (work / "result.json").read_text()
                    )
                    record.status = TaskStatus.SUCCEEDED
                else:
                    path = work / "error.json"
                    record.error = (
                        ToolError.model_validate_json(path.read_text())
                        if path.exists()
                        else ToolError(
                            code="worker_exit", message=f"Worker exit {process.returncode}"
                        )
                    )
                    record.status = TaskStatus.FAILED
        except asyncio.CancelledError:
            record.status = TaskStatus.CANCELLED
            record.error = ToolError(code="cancelled", message="Task cancelled")
        except Exception as exc:
            record.status = TaskStatus.FAILED
            record.error = ToolError(code="runtime_error", message=type(exc).__name__)
        finally:
            if process:
                # Also clean up orphaned descendants after the worker exits.
                kill_tree(process.pid)
                await process.wait()
            if reader and not reader.done():
                reader.cancel()
            if reader:
                await asyncio.gather(reader, return_exceptions=True)
            self.processes.pop(spec.id, None)
            self.store.save(record)
            self.store.emit(
                spec.id,
                "task." + record.status.value,
                {
                    "error": record.error.model_dump() if record.error else None,
                    "peak_rss_bytes": record.peak_rss_bytes,
                },
            )
