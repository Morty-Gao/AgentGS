import importlib.util
import json
import logging

from pydantic import ValidationError

from agentgs.agent.execution.feedback import assess_tool_execution
from agentgs.contracts.models import ToolError, ToolResult
from agentgs.runtime.sandbox.base import SandboxUnavailable
from agentgs.runtime.tools.context import tool_context
from agentgs.runtime.tools.domains.earthquake import (
    CatalogQuery,
    EventQuery,
    get_event,
    query_catalog,
)
from agentgs.runtime.tools.registry import builtin_tools, internal_operations

logger = logging.getLogger(__name__)


class LocalToolExecutor:
    def __init__(
        self, datasets, artifacts, events, code_executor=None, *, allow_internal_operations=False
    ):
        self.datasets, self.artifacts, self.events = datasets, artifacts, events
        self.registry = builtin_tools()
        self.internal = internal_operations() if allow_internal_operations else {}
        self.code_executor = code_executor
        self._known_files = set(self.artifacts.root.rglob("*"))

    def specs(self):
        return [item.spec for item in self.registry.values()]

    async def execute(self, call):
        self._emit("tool.started", {"call_id": call.id, "tool": call.tool})
        item = self.registry.get(call.tool) or self.internal.get(call.tool)
        try:
            if not item:
                raise KeyError(call.tool)
            params = item.input_type.model_validate(call.arguments)
            if call.tool == "ExecutePython":
                data = await self.code_executor.run_python(params.code)
            elif call.tool == "DeepEarthQueryUSGSCatalog":
                args = params.model_dump()
                modern = {"start_year", "end_year", "min_magnitude", "limit"}
                if args and set(args).issubset(modern):
                    data = query_catalog(CatalogQuery.model_validate(args), self.datasets)
                else:
                    data = self._domain(call.tool, args, call_id=call.id)
            elif call.tool == "DeepEarthGetUSGSEvent":
                if importlib.util.find_spec("pandas") is not None:
                    data = self._domain(call.tool, params.model_dump(), call_id=call.id)
                else:
                    data = get_event(EventQuery.model_validate(params.model_dump()), self.datasets)
            elif call.tool == "GeoBigDataExecuteGeoSPARQL":
                data = item.handler(params, self.datasets)
            else:
                data = self._domain(call.tool, params.model_dump(), call_id=call.id)
            raw_output = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            assessment = assess_tool_execution(raw_output)
            if assessment.score == 0:
                raise RuntimeError(raw_output)
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except ValueError:
                    data = {"text": data}
            produced = self._register_products(call, data)
            result = ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=True,
                data=data,
                artifacts=produced,
                source_type=item.source_type,
                raw_output=raw_output,
            )
        except SandboxUnavailable as exc:
            result = ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=False,
                error=ToolError(code="sandbox_unavailable", message=str(exc)),
            )
        except RuntimeError as exc:
            result = ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=False,
                error=ToolError(code="execution_error", message=str(exc), retryable=True),
                raw_output=str(exc),
            )
        except KeyError:
            result = ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=False,
                error=ToolError(code="unknown_tool", message=f"Unavailable tool: {call.tool}"),
            )
        except FileNotFoundError as exc:
            result = ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=False,
                error=ToolError(code="dependency_missing", message=str(exc)),
            )
        except ImportError as exc:
            result = ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=False,
                error=ToolError(code="dependency_missing", message=str(exc)),
                raw_output=f"Tool function execution error, {type(exc).__name__}: {exc}",
            )
        except (ValidationError, ValueError) as exc:
            result = ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=False,
                error=ToolError(code="validation", message=str(exc)),
            )
        self._emit(
            "tool.completed",
            {
                "call_id": call.id,
                "tool": call.tool,
                "succeeded": result.succeeded,
                "error": result.error.model_dump() if result.error else None,
                "arguments": call.arguments,
                "result": result.model_dump(mode="json"),
            },
        )
        return result

    def _emit(self, kind, payload):
        try:
            self.events.emit(kind, payload)
        except Exception:
            logger.exception("Auxiliary tool event could not be saved: %s", kind)

    def _register_products(self, call, data):
        produced = []
        try:
            produced.append(self.artifacts.write_json(call.id + ".json", data, call.tool))
        except Exception as exc:
            self._artifact_failure(call, exc)
        try:
            paths = list(self.artifacts.root.rglob("*"))
        except Exception as exc:
            self._artifact_failure(call, exc)
            return produced
        for path in paths:
            try:
                if path.is_file() and path not in self._known_files:
                    self._known_files.add(path)
                    if path.name != call.id + ".json":
                        produced.append(self.artifacts.register(path, call.tool))
            except Exception as exc:
                self._artifact_failure(call, exc)
        return produced

    def _artifact_failure(self, call, exc):
        logger.warning("Tool %s succeeded but artifact recording failed: %s", call.tool, exc)
        self._emit(
            "tool.artifact_failed",
            {"call_id": call.id, "tool": call.tool, "error_type": type(exc).__name__},
        )

    def _domain(self, name, arguments, *, call_id=None):
        output = self.artifacts.root / "calls" / call_id if call_id else self.artifacts.root
        with tool_context(self.datasets.root.parent, output, self.artifacts.root):
            return self.registry[name].handler(arguments)
