from typing import Any, Protocol

from pydantic import Field, SecretStr

from agentgs.contracts.models import Model


class Provider(Model):
    protocol: str = "openai_compatible"
    base_url: str
    api_key: SecretStr
    timeout_seconds: float = Field(default=120, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=5)
    output_token_parameter: str = "max_tokens"


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
    message: dict[str, Any]
    usage: dict[str, Any] | None = None
    attempts: int = 1


class ModelClient(Protocol):
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        effort: str | None = None,
    ) -> ChatReply: ...
