from pydantic import ValidationError

from agentgs.contracts.models import ToolError, ToolResult
from agentgs.runtime.sandbox.base import SandboxUnavailable
from agentgs.runtime.tools.domains.earthquake import get_event, query_catalog
from agentgs.runtime.tools.registry import builtin_tools


class LocalToolExecutor:
    def __init__(self, datasets, artifacts, events, code_executor=None):
        self.datasets, self.artifacts, self.events = datasets, artifacts, events
        self.registry = builtin_tools()
        self.code_executor = code_executor

    def specs(self):
        return [item.spec for item in self.registry.values()]

    async def execute(self, call):
        self.events.emit("tool.started", {"call_id": call.id, "tool": call.tool})
        item = self.registry.get(call.tool)
        try:
            if not item:
                raise KeyError(call.tool)
            params = item.input_type.model_validate(call.arguments)
            if call.tool == "ExecutePython":
                data = await self.code_executor.run_python(params.code)
            elif call.tool == "DeepEarthQueryUSGSCatalog":
                data = query_catalog(params, self.datasets)
            elif call.tool == "DeepEarthGetUSGSEvent":
                data = get_event(params, self.datasets)
            else:
                data = item.handler(params)
            artifact = self.artifacts.write_json(call.id + ".json", data, call.tool)
            result = ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=True,
                data=data,
                artifacts=[artifact],
                source_type=item.source_type,
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
                error=ToolError(code="execution_error", message=str(exc)),
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
        except (ValidationError, ValueError) as exc:
            result = ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=False,
                error=ToolError(code="validation", message=str(exc)),
            )
        self.events.emit(
            "tool.completed",
            {
                "call_id": call.id,
                "tool": call.tool,
                "succeeded": result.succeeded,
                "error": result.error.model_dump() if result.error else None,
            },
        )
        return result
