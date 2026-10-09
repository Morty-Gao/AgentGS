from typing import Any, Protocol

from pydantic import Field, SecretStr, field_validator

from agentgs.contracts.models import Model
from agentgs.messages import Conversation, Message


class Provider(Model):
    protocol: str = "openai_compatible"
    base_url: str
    api_key: SecretStr
    timeout_seconds: float = Field(default=120, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=5)
    output_token_parameter: str = "max_tokens"
    max_concurrent_requests: int | None = Field(default=None, ge=1)
    requests_per_minute: int | None = Field(default=None, ge=1)
    tokens_per_minute: int | None = Field(default=None, ge=1)


class Capabilities(Model):
    tool_calling: bool = True
    streaming: bool = False
    structured_output: bool = False


class LLMModel(Model):
    provider: str
    model: str
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    supported_efforts: list[str] = Field(default_factory=list)
    default_effort: str | None = None
    capabilities: Capabilities = Field(default_factory=Capabilities)


class LLMConfig(Model):
    schema_version: int = 1
    providers: dict[str, Provider]
    models: dict[str, LLMModel]
    roles: dict[str, str]


class ChatReply(Model):
    message: Message
    usage: dict[str, Any] | None = None
    attempts: int = 1
    response_model: str | None = None
    response_id: str | None = None
    raw_response: dict[str, Any] | None = None
    request_payload: dict[str, Any] | None = None

    @field_validator("message", mode="before")
    @classmethod
    def parse_message(cls, value):
        if isinstance(value, dict):
            return Message.from_openai({"role": "assistant", **value})
        return value


class ModelClient(Protocol):
    async def chat(
        self,
        messages: Conversation | list[Message | dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        effort: str | None = None,
        *,
        tool_choice: str | dict | None = None,
        parallel_tool_calls: bool = False,
        stage: str = "tool_calling",
        settings: Any = None,
    ) -> ChatReply: ...

    async def aclose(self) -> None: ...
