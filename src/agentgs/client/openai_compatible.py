import asyncio
import random

import httpx

from agentgs.client.errors import ProviderError
from agentgs.client.models import ChatReply, LLMConfig


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig, transport=None, events=None):
        self.config = config
        self.events = events
        self.http = httpx.AsyncClient(transport=transport)

    async def aclose(self):
        await self.http.aclose()

    async def chat(self, messages, tools=None, model=None, effort=None):
        name = model or self.config.roles.get("planner")
        if name not in self.config.models:
            raise ValueError(f"Unknown configured model: {name}")
        spec = self.config.models[name]
        provider = self.config.providers[spec.provider]
        if provider.protocol != "openai_compatible":
            raise ValueError(f"Unsupported provider protocol: {provider.protocol}")
        selected_effort = effort or spec.default_effort
        if selected_effort and selected_effort not in spec.supported_efforts:
            raise ValueError(f"Model {name} does not support effort {selected_effort}")
        if tools and not spec.capabilities.tool_calling:
            raise ValueError(f"Model {name} does not support tool calling")
        payload = {
            "model": spec.model,
            "messages": messages,
            provider.output_token_parameter: spec.max_output_tokens,
        }
        if tools:
            payload["tools"] = tools
        if selected_effort:
            payload["reasoning_effort"] = selected_effort
        # Context size is advisory until an exact model tokenizer is supplied.
        # The provider remains authoritative; never claim character counts are tokens.
        if provider.output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported output token parameter")
        for attempt in range(provider.max_attempts):
            if self.events:
                self.events.emit("model.attempt", {"model": name, "attempt": attempt + 1})
            try:
                response = await self.http.post(
                    provider.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + provider.api_key.get_secret_value()},
                    json=payload,
                    timeout=provider.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                return ChatReply(
                    message=body["choices"][0]["message"],
                    usage=body.get("usage"),
                    attempts=attempt + 1,
                    response_model=body.get("model"),
                    response_id=body.get("id"),
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {408, 429, 500, 502, 503, 504}:
                    raise ProviderError(exc.response.status_code) from None
                retry_after = exc.response.headers.get("retry-after", "")
                delay = min(float(retry_after), 30) if retry_after.isdigit() else 2**attempt
            except httpx.TransportError:
                delay = 2**attempt
            if attempt + 1 == provider.max_attempts:
                raise RuntimeError("LLM transport/retry budget exhausted") from None
            await asyncio.sleep(delay)
        raise RuntimeError("No LLM response")

    async def text(self, prompt, *, stage, settings, model=None, effort=None, system_prompt=""):
        role = (
            "argument"
            if stage == "tool_argument_generation"
            else "answer"
            if stage == "final_answer_generation"
            else "planner"
        )
        name = model or self.config.roles.get(role) or self.config.roles.get("planner")
        if name not in self.config.models:
            raise ValueError(f"Unknown configured model: {name}")
        spec = self.config.models[name]
        provider = self.config.providers[spec.provider]
        if provider.protocol != "openai_compatible":
            raise ValueError("Legacy transport requires an OpenAI-compatible provider")
        minimax = spec.model.lower().startswith(("minimax", "abab"))
        temperature = (
            (1.0 if minimax else 0.2) if settings.temperature is None else settings.temperature
        )
        if minimax:
            temperature = min(1.0, max(0.01, temperature))
        payload = {
            "model": spec.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                    or "You are ChatGPT, a large language model trained by OpenAI.\n",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if minimax:
            payload["reasoning_split"] = True
        if effort:
            if effort not in spec.supported_efforts:
                raise ValueError(f"Model {name} does not support effort {effort}")
            payload["reasoning_effort"] = effort
        attempts = settings.model_attempts
        if stage == "tool_argument_generation":
            attempts = min(attempts, settings.argument_attempts)
        for attempt in range(attempts):
            try:
                response = await self.http.post(
                    provider.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + provider.api_key.get_secret_value()},
                    json=payload,
                    timeout=provider.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                return ChatReply(
                    message=body["choices"][0]["message"],
                    usage=body.get("usage"),
                    attempts=attempt + 1,
                    response_model=body.get("model"),
                    response_id=body.get("id"),
                )
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if self.events:
                    self.events.emit(
                        "model.attempt_failed",
                        {"stage": stage, "attempt": attempt + 1, "status_code": code},
                    )
                if code not in {408, 409, 425, 429} and code < 500 or attempt + 1 == attempts:
                    raise ProviderError(code) from None
            except (httpx.TransportError, ValueError) as exc:
                if self.events:
                    self.events.emit(
                        "model.attempt_failed",
                        {"stage": stage, "attempt": attempt + 1, "error_type": type(exc).__name__},
                    )
                if attempt + 1 == attempts:
                    raise RuntimeError("LLM request/retry budget exhausted") from None
            delay = max(0.0, settings.retry_base_seconds) * 2**attempt
            delay += random.uniform(0, max(0.0, settings.retry_jitter_seconds))
            if delay:
                await asyncio.sleep(delay)
        raise RuntimeError("No LLM response")
