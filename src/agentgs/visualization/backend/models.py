"""Small read models for the console; execution contracts stay in contracts/models."""

from pydantic import Field

from agentgs.config.algorithm import AlgorithmSettings
from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import Model, TaskStatus


class TaskSummary(Model):
    id: str
    question: str
    tools: list[str]
    status: TaskStatus
    created_at: str
    planning_mode: str | None
    model: str | None
    peak_rss_bytes: int
    completion_status: str | None
    error_code: str | None


class TaskPage(Model):
    items: list[TaskSummary]
    total: int
    limit: int
    offset: int
    counts: dict[TaskStatus, int] = Field(default_factory=dict)


class ConsoleConfig(Model):
    algorithm: AlgorithmSettings
    runtime: RuntimeSettings
