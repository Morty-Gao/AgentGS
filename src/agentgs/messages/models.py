"""Provider-independent messages; reasoning is retained, never invented.

RuntimeEvent is an independent audit record excluded from provider requests.
Serialization retains all reasoning. Conversation.to_request_messages() applies
the user-turn policy to an independent OpenAI-formatted request copy.
"""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Function(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    # Keep the exact string, including malformed JSON returned by a model.
    arguments: str


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: Function
    index: int | None = Field(default=None, ge=0)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)
    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[FunctionCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    reasoning: Any = None
    provider_fields: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def import_provider_fields(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            if "function_call" in value or value.get("role") == "function":
                raise ValueError(
                    "Deprecated function_call/function messages are unsupported; use tool_calls/tool"
                )
            extras = {key: value.pop(key) for key in list(value) if key not in cls.model_fields}
            # A provider may return both reasoning and reasoning_content/details.
            # Keep the explicit reasoning field as well as the normalized value;
            # serialized internal messages already carry their original fields.
            if "reasoning" in value and "provider_fields" not in value:
                extras["reasoning"] = value["reasoning"]
            value["provider_fields"] = {**value.get("provider_fields", {}), **extras}
            if value.get("reasoning") is None:
                value["reasoning"] = next(
                    (
                        extras[key]
                        for key in ("reasoning_content", "reasoning_details", "thinking")
                        if key in extras
                    ),
                    None,
                )
            if value.get("tool_calls") is None:
                value["tool_calls"] = []
        return value

    @model_validator(mode="after")
    def validate_role(self):
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("A tool message requires tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages may contain function calls")
        return self

    @classmethod
    def from_openai(cls, value: dict[str, Any]) -> "Message":
        """Import an OpenAI-style message, preserving unknown provider fields."""
        return cls.model_validate(deepcopy(value))

    def to_openai(self, *, include_reasoning: bool = True) -> dict[str, Any]:
        """Convert content, function calls and provider fields to OpenAI format.

        Internal ids, timestamps, metadata, and event messages never go on the wire.
        include_reasoning is a conversion option, not a serialization option;
        Conversation.to_request_messages() selects it separately for each turn.
        """
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = [call.model_dump(exclude={"index"}) for call in self.tool_calls]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        for key in ("refusal", "audio"):
            if key in self.provider_fields:
                result[key] = self.provider_fields[key]
        if include_reasoning:
            for key in ("reasoning", "reasoning_content", "reasoning_details", "thinking"):
                if key in self.provider_fields:
                    result[key] = self.provider_fields[key]
            if self.reasoning is not None and not any(
                key in result
                for key in ("reasoning", "reasoning_content", "reasoning_details", "thinking")
            ):
                result["reasoning"] = self.reasoning
        return deepcopy(result)


class RuntimeEvent(BaseModel):
    """An audit record, never a model message."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)
    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    role: Literal["event"] = "event"
    metadata: dict[str, Any] = Field(default_factory=dict)


LogEntry = Message | RuntimeEvent


def parse_entry(value: dict[str, Any]) -> LogEntry:
    kind = RuntimeEvent if value.get("role") == "event" else Message
    return kind.model_validate(value)
