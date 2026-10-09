"""Dependency interfaces implemented by the runtime and used by agents."""

from typing import Any, Protocol

from .tasks import AgentResult, TaskInput
from .tools import ArtifactRef, ToolCall, ToolResult, ToolSpec


class EventSink(Protocol):
    def emit(self, kind: str, payload: dict[str, Any]) -> None: ...


class ToolExecutor(Protocol):
    def specs(self) -> list[ToolSpec]: ...
    async def execute(self, call: ToolCall) -> ToolResult: ...


class ArtifactStore(Protocol):
    def write_json(self, name: str, data: Any, source: str) -> ArtifactRef: ...


class AgentRunner(Protocol):
    async def run(self, task: TaskInput, context: Any) -> AgentResult: ...
