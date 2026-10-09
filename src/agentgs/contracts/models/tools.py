"""Tool descriptions, invocations, results and artifact references."""

from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, PrivateAttr

from .base import Model


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
    depends_on: list[str] = Field(default_factory=list)
    _runtime_arguments: dict[str, Any] = PrivateAttr(default_factory=dict)


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
    raw_output: str | None = None


class ToolSpec(Model):
    id: str
    description: str
    input_schema: dict[str, Any]
    effects: Literal["read", "write"] = "read"
    category: str = ""
    input_description: str = ""
    output_description: str = ""
