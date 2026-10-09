"""One native Chat Completions transport for every Agent stage."""

import asyncio
import json
import random

import httpx

from agentgs.client.errors import ProviderError
from agentgs.client.models import ChatReply, LLMConfig
from agentgs.messages import Conversation, Message


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig, transport=None, events=None):
        self.config = config
        self.events = events
        self.http = httpx.AsyncClient(transport=transport)

    async def aclose(self):
        await self.http.aclose()

    async def chat(
        self, messages, tools=None, model=None, effort=None, *, tool_choice=None,
        parallel_tool_calls=False, stage="tool_calling", settings=None,
    ):
        if isinstance(messages, Conversation):
            conversation = messages
        else:
            conversation = Conversation(messages=[
                m if isinstance(m, Message) else Message.from_openai(m)
                for m in messages
            ])
        messages = conversation.to_request_messages()
        role = "answer" if stage == "final_answer_generation" else "argument" if stage == "tool_argument_generation" else "planner"
        name = model or self.config.roles.get(role) or self.config.roles.get("planner")
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
        if provider.output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported output token parameter")
        payload = {
            "model": spec.model, "messages": messages,
            provider.output_token_parameter: spec.max_output_tokens,
        }
        if tools:
            payload.update(tools=tools, parallel_tool_calls=parallel_tool_calls)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if selected_effort:
            payload["reasoning_effort"] = selected_effort
        if settings is not None:
            minimax = spec.model.lower().startswith(("minimax", "abab"))
            temperature = settings.temperature
            if temperature is None:
                temperature = 1.0 if minimax else 0.2
            payload["temperature"] = min(1.0, max(0.01, temperature)) if minimax else temperature
            if minimax:
                payload["reasoning_split"] = True
        attempts = provider.max_attempts if settings is None else settings.model_attempts
        if settings is not None and stage == "tool_argument_generation":
            attempts = min(attempts, settings.argument_attempts)
        for attempt in range(attempts):
            if self.events:
                self.events.emit("model.attempt", {"model": name, "stage": stage, "attempt": attempt + 1})
            try:
                response = await self.http.post(
                    provider.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + provider.api_key.get_secret_value()},
                    json=payload, timeout=provider.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code not in {408, 409, 425, 429} and code < 500 or attempt + 1 == attempts:
                    raise ProviderError(code) from None
                retry_after = exc.response.headers.get("retry-after", "")
                delay = min(float(retry_after), 30) if retry_after.isdigit() else None
            except (httpx.TransportError, json.JSONDecodeError) as exc:
                if attempt + 1 == attempts:
                    raise RuntimeError("LLM transport/retry budget exhausted") from exc
                delay = None
            else:
                # Invalid successful replies are protocol errors, not network retries.
                return ChatReply(
                    message=body["choices"][0]["message"], usage=body.get("usage"),
                    attempts=attempt + 1, response_model=body.get("model"),
                    response_id=body.get("id"), raw_response=body, request_payload=payload,
                )
            if self.events:
                self.events.emit("model.attempt_failed", {"model": name, "stage": stage, "attempt": attempt + 1})
            if delay is None:
                base = 1.0 if settings is None else settings.retry_base_seconds
                jitter = 0.0 if settings is None else settings.retry_jitter_seconds
                delay = max(0.0, base) * 2**attempt + random.uniform(0, max(0.0, jitter))
            await asyncio.sleep(delay)
        raise RuntimeError("No LLM response")
