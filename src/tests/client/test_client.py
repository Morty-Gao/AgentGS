import json

import httpx
import pytest

from agentgs.client.models import LLMConfig
from agentgs.client.openai_compatible import OpenAICompatibleClient


@pytest.fixture
def config():
    return LLMConfig.model_validate(
        {
            "providers": {
                "p": {"base_url": "https://fixture/v1", "api_key": "secret", "max_attempts": 2}
            },
            "models": {
                "m": {
                    "provider": "p",
                    "model": "remote",
                    "context_window": 100,
                    "max_output_tokens": 10,
                    "supported_efforts": ["low"],
                }
            },
            "roles": {"planner": "m"},
        }
    )


async def test_retry_protocol_and_missing_usage(config):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "answer"}}]}
        )

    client = OpenAICompatibleClient(config, httpx.MockTransport(handler))
    try:
        reply = await client.chat([{"role": "user", "content": "q"}], effort="low")
        assert reply.usage is None
        assert len(requests) == 2
        assert json.loads(requests[-1].content)["reasoning_effort"] == "low"
        assert requests[-1].url.path == "/v1/chat/completions"
        with pytest.raises(ValueError, match="effort"):
            await client.chat([], effort="high")
    finally:
        await client.aclose()


async def test_error_body_is_not_exposed(config):
    client = OpenAICompatibleClient(
        config, httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "secret"}))
    )
    try:
        with pytest.raises(RuntimeError, match="HTTP 401") as exc:
            await client.chat([])
        assert "secret" not in str(exc.value)
    finally:
        await client.aclose()


async def test_provider_token_parameter(config):
    provider = config.providers["p"].model_copy(
        update={"output_token_parameter": "max_completion_tokens"}
    )
    config = config.model_copy(update={"providers": {"p": provider}})

    def handler(request):
        payload = json.loads(request.content)
        assert payload["max_completion_tokens"] == 10 and "max_tokens" not in payload
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    client = OpenAICompatibleClient(config, httpx.MockTransport(handler))
    try:
        assert (await client.chat([])).message["content"] == "ok"
    finally:
        await client.aclose()
