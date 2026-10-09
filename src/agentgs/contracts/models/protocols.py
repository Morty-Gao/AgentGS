"""Dependency interfaces implemented by the runtime and used by agents."""

from typing import Any, Awaitable, Callable, Protocol

from .tasks import AgentResult, TaskInput
from .tools import ArtifactRef, ToolCall, ToolResult, ToolSpec


class EventSink(Protocol):
    async def emit(self, kind: str, payload: dict[str, Any]) -> None: ...


class ToolExecutor(Protocol):
    def specs(self) -> list[ToolSpec]: ...
    async def execute(self, call: ToolCall, *, timeout: float | None = None) -> ToolResult: ...
    async def execute_many(
        self,
        calls: list[ToolCall],
        *,
        on_result: Callable[[ToolResult], Awaitable[None]] | None = None,
    ) -> list[ToolResult]: ...
    async def aclose(self) -> None: ...


class ArtifactStore(Protocol):
    async def write_json(self, name: str, data: Any, source: str) -> ArtifactRef: ...
    async def register(self, path: Any, source: str) -> ArtifactRef: ...


class AgentRunner(Protocol):
    async def run(self, task: TaskInput, context: Any) -> AgentResult: ...
