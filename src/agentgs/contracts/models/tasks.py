"""Task lifecycle, conversation inputs, agent results and events."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from agentgs.messages import Conversation, Message

from .base import Model
from .tools import ToolCall, ToolError, ToolResult


class TaskStatus(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}


class TaskInput(Model):
    question: str = ""
    messages: list[Message] = Field(default_factory=list)
    calls: list[ToolCall] = Field(default_factory=list)
    model: str | None = None
    effort: str | None = None
    skill_ids: list[str] = Field(default_factory=list)
    max_model_calls: int | None = Field(default=None, ge=1, le=10000)
    input_resource: Any = None
    file_path_list: list[str] = Field(default_factory=list)
    output_contract: dict[str, Any] | None = None
    planning_mode: Literal["tool_only", "skill_guided"] | None = None
    require_llm: bool | None = None
    input_domain: str | None = None

    @model_validator(mode="after")
    def question_from_messages(self):
        Conversation(messages=self.messages)
        if not self.question.strip():
            for message in reversed(self.messages):
                if message.role == "user" and isinstance(message.content, str):
                    self.question = message.content
                    break
        return self


class TaskSpec(Model):
    id: str = Field(default_factory=lambda: uuid4().hex)
    input: TaskInput
    timeout_seconds: float = Field(default=300, gt=0)
    rss_limit_mb: int = Field(default=1024, ge=64)


class AgentResult(Model):
    answer: str
    tool_results: list[ToolResult] = Field(default_factory=list)
    usage: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    completion_status: Literal["Completed", "Not Completed"] = "Completed"
    tool_chain_completed: bool = True
    answer_generation_status: str = "success"
    algorithm_id: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_messages(self):
        Conversation(messages=self.messages)
        return self

    def execution_events(self):
        return Conversation(messages=self.messages).execution_events()


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
