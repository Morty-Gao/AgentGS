from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class TaskStatus(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}


class ArtifactRef(Model):
    id: str
    path: str
    media_type: str
    sha256: str
    bytes: int
    source: str


class ToolCall(Model):
    id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolError(Model):
    code: str
    message: str
    retryable: bool = False


class ToolResult(Model):
    call_id: str
    tool: str
    succeeded: bool
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    error: ToolError | None = None
    source_type: Literal["raw_compute", "derived_lookup", "external_backend"] = "raw_compute"


class ToolSpec(Model):
    id: str
    description: str
    input_schema: dict[str, Any]
    effects: Literal["read", "write"] = "read"


class TaskInput(Model):
    question: str = ""
    calls: list[ToolCall] = Field(default_factory=list)
    model: str | None = None
    effort: str | None = None
    skill_ids: list[str] = Field(default_factory=list)
    max_model_calls: int = Field(default=8, ge=1, le=100)


class TaskSpec(Model):
    id: str = Field(default_factory=lambda: uuid4().hex)
    input: TaskInput
    timeout_seconds: float = Field(default=300, gt=0)
    rss_limit_mb: int = Field(default=1024, ge=64)


class AgentResult(Model):
    answer: str
    tool_results: list[ToolResult] = Field(default_factory=list)
    usage: list[dict[str, Any]] = Field(default_factory=list)


class TaskRecord(Model):
    spec: TaskSpec
    status: TaskStatus = TaskStatus.QUEUED
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    result: AgentResult | None = None
    error: ToolError | None = None
    peak_rss_bytes: int = 0


class Event(Model):
    task_id: str
    sequence: int
    type: str
    time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    payload: dict[str, Any] = Field(default_factory=dict)


class EventSink(Protocol):
    def emit(self, kind: str, payload: dict[str, Any]) -> None: ...


class ToolExecutor(Protocol):
    def specs(self) -> list[ToolSpec]: ...
    async def execute(self, call: ToolCall) -> ToolResult: ...


class ArtifactStore(Protocol):
    def write_json(self, name: str, data: Any, source: str) -> ArtifactRef: ...


class AgentRunner(Protocol):
    async def run(self, task: TaskInput, context: Any) -> AgentResult: ...
