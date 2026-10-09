"""Bounded async execution of synchronous, asynchronous and process tools."""

import asyncio
import json
import logging
import time

from pydantic import ValidationError

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import ToolError, ToolResult
from agentgs.contracts.sandbox import SandboxResourceLimit
from agentgs.contracts.tool_feedback import assess_tool_execution
from agentgs.runtime.concurrency import RuntimeCoordinator
from agentgs.runtime.sandbox.base import SandboxUnavailable
from agentgs.runtime.tools.compute import compute_tool
from agentgs.runtime.tools.context import ToolPaths, tool_context
from agentgs.runtime.tools.process import ProcessToolRunner
from agentgs.runtime.tools.registry import builtin_tools, internal_operations

logger = logging.getLogger(__name__)


class LocalToolExecutor:
    def __init__(
        self,
        datasets,
        artifacts,
        events,
        code_executor=None,
        *,
        allow_internal_operations=False,
        include_scientific_tools=True,
        settings=None,
        coordinator=None,
    ):
        self.datasets, self.artifacts, self.events = datasets, artifacts, events
        self.registry = builtin_tools() if include_scientific_tools else {}
        self._builtin_handlers = {name: item.handler for name, item in self.registry.items()}
        self.internal = internal_operations() if allow_internal_operations else {}
        if code_executor is not None:
            self.internal["bash"] = internal_operations()["bash"]
        self.code_executor = code_executor
        self.settings = settings or RuntimeSettings()
        self.coordinator = coordinator or RuntimeCoordinator(datasets.root.parent)
        self._active = set()
        self._closed = False
        self._tasks = set()
        self._process_runner = ProcessToolRunner()

    def specs(self):
        return [item.spec for item in self.registry.values()]

    async def aclose(self):
        self._closed = True
        tasks = [task for task in self._tasks if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def execute(self, call, *, timeout=None):
        if self._closed:
            raise RuntimeError("Tool executor is closed")
        if call.id in self._active:
            raise ValueError("A tool call id cannot execute concurrently twice")
        self._active.add(call.id)
        task = asyncio.current_task()
        self._tasks.add(task)
        try:
            return await self._execute(call, timeout=timeout)
        finally:
            self._active.discard(call.id)
            self._tasks.discard(task)

    async def _execute(self, call, *, timeout=None):
        item = self.registry.get(call.tool) or self.internal.get(call.tool)
        queued_at = time.monotonic()
        started_at = finished_at = None

        async def started():
            nonlocal started_at
            started_at = time.monotonic()
            await self._emit("tool.started", {"call_id": call.id, "tool": call.tool})

        async def compute(params, arguments):
            nonlocal finished_at
            try:
                return await self._compute(
                    call, item, params, arguments, timeout=timeout, on_started=started
                )
            finally:
                finished_at = time.monotonic()

        try:
            async with asyncio.timeout(
                timeout if timeout is not None else self.settings.tool_timeout_seconds
            ):
                await self._emit("tool.queued", {"call_id": call.id, "tool": call.tool})
                if item is None:
                    raise KeyError(call.tool)
                arguments = {
                    key: value
                    for key, value in call.arguments.items()
                    if value is not None
                    or key not in item.input_type.model_fields
                    or item.input_type.model_fields[key].is_required()
                }
                params = item.input_type.model_validate(arguments)
                arguments = params.model_dump(exclude_unset=True, by_alias=True)
                arguments.update(call._runtime_arguments)
                if call.tool in {"bash", "ExecutePython"}:
                    data = await compute(params, arguments)
                else:
                    async with self.coordinator.tool_invocation(
                        self.settings,
                        timeout if timeout is not None else self.settings.tool_timeout_seconds,
                    ):
                        await started()
                        data = await compute(params, arguments)
                if call.tool == "bash":
                    # Shell streams are already preserved in data; avoid duplicating model context.
                    raw_output = None
                else:
                    data, raw_output = await invoke(self._decode_result, data)
                produced = await self._register_products(call, data)
                error = None
                if call.tool == "bash" and data["exit_code"] != 0:
                    error = ToolError(code="shell_exit", message=f"Bash exited {data['exit_code']}")
                result = ToolResult(
                    call_id=call.id,
                    tool=call.tool,
                    succeeded=error is None,
                    data=data,
                    artifacts=produced,
                    error=error,
                    source_type=item.source_type,
                    raw_output=raw_output,
                )
        except asyncio.CancelledError:
            await self._emit("tool.cancelled", {"call_id": call.id, "tool": call.tool})
            raise
        except TimeoutError:
            result = self._failure(
                call, "timeout", "Tool invocation exceeded its time budget", retryable=True
            )
        except SandboxUnavailable as exc:
            result = self._failure(call, "sandbox_unavailable", str(exc))
        except SandboxResourceLimit as exc:
            result = self._failure(call, "resource_limit", str(exc))
        except KeyError:
            result = self._failure(call, "unknown_tool", f"Unavailable tool: {call.tool}")
        except (FileNotFoundError, ImportError) as exc:
            result = self._failure(call, "dependency_missing", str(exc))
        except (ValidationError, ValueError, TypeError) as exc:
            result = self._failure(call, "validation", str(exc))
        except Exception as exc:
            result = self._failure(
                call, "execution_error", str(exc), retryable=True, raw_output=str(exc)
            )
        await self._emit(
            "tool.completed",
            {
                "call_id": call.id,
                "tool": call.tool,
                "succeeded": result.succeeded,
                "error": result.error.model_dump() if result.error else None,
                "arguments": call.arguments,
                "result": result.model_dump(mode="json"),
                "wait_seconds": (started_at or time.monotonic()) - queued_at,
                "execution_seconds": max(0, (finished_at or time.monotonic()) - started_at)
                if started_at is not None
                else 0,
            },
        )
        return result

    async def _compute(self, call, item, params, arguments, *, timeout=None, on_started=None):
        if call.tool == "bash":
            if self.code_executor is None:
                raise SandboxUnavailable("No code executor is configured")
            return await self.code_executor.run_shell(
                params.command,
                work=self.code_executor.work / "shell",
                data_root=self.datasets.root,
                timeout=timeout if timeout is not None else self.settings.tool_timeout_seconds,
                coordinator=self.coordinator,
                settings=self.settings,
                on_started=on_started,
            )
        if call.tool == "ExecutePython":
            if self.code_executor is None:
                raise SandboxUnavailable("No code executor is configured")
            return await self.code_executor.run_python(
                params.code,
                coordinator=self.coordinator,
                settings=self.settings,
                on_started=on_started,
            )
        unchanged_handler = self._builtin_handlers.get(call.tool) is item.handler
        unchanged_dispatch = getattr(self._domain, "__func__", None) is LocalToolExecutor._domain
        if (
            self.settings.tool_backend == "process"
            and item.process_safe
            and unchanged_handler
            and unchanged_dispatch
        ):
            from agentgs.contracts.models import ToolCall
            from agentgs.contracts.remote_sensing import enabled

            validated = ToolCall(
                id=call.id,
                tool=call.tool,
                arguments={
                    key: value
                    for key, value in arguments.items()
                    if key not in call._runtime_arguments
                },
            )
            validated._runtime_arguments = call._runtime_arguments
            return await self._process_runner.run(
                self.datasets.root.parent, self.artifacts.root, validated, rs_enabled=enabled()
            )
        output = self.artifacts.root / "calls" / call.id
        # Keep the context active while an async handler is actually awaited.
        paths = await invoke(
            ToolPaths.create, self.datasets.root.parent, output, self.artifacts.root
        )
        with tool_context(self.datasets.root.parent, output, self.artifacts.root, prepared=paths):
            return await invoke(self._domain, call.tool, arguments, call_id=call.id)

    def _domain(self, name, arguments, *, call_id=None):
        return compute_tool(name, arguments, self.datasets, self.registry)

    @staticmethod
    def _decode_result(data):
        raw = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        if assess_tool_execution(raw).score == 0:
            raise RuntimeError(raw)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                data = {"text": data}
        return data, raw

    @staticmethod
    def _failure(call, code, message, *, retryable=False, raw_output=None):
        return ToolResult(
            call_id=call.id,
            tool=call.tool,
            succeeded=False,
            error=ToolError(code=code, message=message, retryable=retryable),
            raw_output=raw_output,
        )

    async def _emit(self, kind, payload):
        try:
            await invoke(self.events.emit, kind, payload)
        except Exception:
            logger.exception("Auxiliary tool event could not be saved: %s", kind)

    async def _register_products(self, call, data):
        produced = []
        try:
            produced.append(
                await invoke(self.artifacts.write_json, call.id + ".json", data, call.tool)
            )
        except Exception as exc:
            logger.warning("Tool %s artifact recording failed: %s", call.tool, exc)
        # Each invocation owns its directory; never claim another call's files.
        output = self.artifacts.root / "calls" / call.id
        for path in await invoke(lambda: [p for p in output.rglob("*") if p.is_file()]):
            if path:
                try:
                    produced.append(await invoke(self.artifacts.register, path, call.tool))
                except Exception as exc:
                    logger.warning("Tool %s artifact registration failed: %s", call.tool, exc)
        return produced

    async def execute_many(self, calls, *, on_result=None):
        """Run independent calls concurrently and honor explicit dependencies."""
        calls = list(calls)
        by_id = {call.id: call for call in calls}
        if len(by_id) != len(calls):
            raise ValueError("Duplicate tool call ids")
        visiting, visited = set(), set()

        def validate(key):
            if key in visiting:
                raise ValueError("Cyclic tool dependencies")
            if key in visited:
                return
            visiting.add(key)
            for dependency in by_id[key].depends_on:
                if dependency not in by_id:
                    raise ValueError("Unknown tool dependency: " + dependency)
                validate(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in by_id:
            validate(key)
        jobs = {}

        async def one(call):
            result = None
            for dependency in call.depends_on:
                result = await jobs[dependency]
                if not result.succeeded:
                    result = self._failure(
                        call, "dependency_failed", "An upstream tool did not succeed"
                    )
                    await self._emit(
                        "tool.completed",
                        {
                            "call_id": call.id,
                            "tool": call.tool,
                            "succeeded": False,
                            "error": result.error.model_dump(),
                            "result": result.model_dump(mode="json"),
                        },
                    )
                    break
            else:
                result = await self.execute(call)
            if on_result:
                await invoke(on_result, result)
            return result

        try:
            # All tasks are installed before any coroutine can inspect jobs.
            jobs = {call.id: asyncio.create_task(one(call)) for call in calls}
            return await asyncio.gather(*(jobs[call.id] for call in calls))
        finally:
            for job in jobs.values():
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs.values(), return_exceptions=True)
