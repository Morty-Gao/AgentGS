from typing import Literal

from pydantic import Field

from agentgs.contracts.models import Model


class RuntimeSettings(Model):
    max_concurrent_tasks: int = Field(default=2, ge=1, le=128)
    max_pending_tasks: int = Field(default=1024, ge=1)
    max_concurrent_model_calls: int = Field(default=8, ge=1, le=128)
    max_concurrent_tool_calls: int = Field(default=4, ge=1, le=128)
    model_timeout_seconds: float = Field(default=180, gt=0)
    tool_timeout_seconds: float = Field(default=120, gt=0)
    tool_backend: Literal["process", "thread"] = "thread"
    default_timeout_seconds: float = Field(default=300, gt=0)
    default_rss_limit_mb: int = Field(default=1024, ge=64)
    max_total_rss_mb: int = Field(default=4096, ge=64)
    code_timeout_seconds: float = Field(default=20, gt=0, le=300)
    sandbox: str = "auto"
    allow_proot: bool = False
    worker_python: str | None = None
