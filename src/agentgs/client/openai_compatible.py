"""One native Chat Completions transport for every Agent stage."""

import asyncio
import json
import random
from contextlib import asynccontextmanager

import httpx

from agentgs.client.errors import ProviderError
from agentgs.client.models import ChatReply, LLMConfig
from agentgs.contracts.async_utils import drain, invoke
from agentgs.messages import Conversation, Message


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig, transport=None, events=None):
        self.config = config
        self.events = events
        self.request_scope = None
        self.usage_recorder = None
        self._closed = False
        self._tasks = set()
        self.http = httpx.AsyncClient(transport=transport)

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError("Model client is closed")
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        self._closed = True
        caller = asyncio.current_task()

        async def close():
            tasks = [task for task in self._tasks if task is not caller]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.http.aclose()

        await drain(asyncio.create_task(close()))

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        effort=None,
        *,
        tool_choice=None,
        parallel_tool_calls=False,
        stage="tool_calling",
        settings=None,
    ):
        if self._closed:
            raise RuntimeError("Model client is closed")
        task = asyncio.current_task()
        self._tasks.add(task)
        try:
            return await self._chat(
                messages,
                tools,
                model,
                effort,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                stage=stage,
                settings=settings,
            )
        finally:
            self._tasks.discard(task)

    async def _chat(
        self,
        messages,
        tools=None,
        model=None,
        effort=None,
        *,
        tool_choice=None,
        parallel_tool_calls=False,
        stage="tool_calling",
        settings=None,
    ):
        if isinstance(messages, Conversation):
            conversation = messages
        else:
            conversation = Conversation(
                messages=[m if isinstance(m, Message) else Message.from_openai(m) for m in messages]
            )
        messages = conversation.to_request_messages()
        name, spec = self.config.configured_model(model)
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
            "model": spec.model,
            "messages": messages,
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
                await invoke(
                    self.events.emit,
                    "model.attempt",
                    {"model": name, "stage": stage, "attempt": attempt + 1},
                )
            try:

                @asynccontextmanager
                async def unlimited():
                    yield None

                # Reserve conservatively without requiring a provider-specific
                # tokenizer; successful replies settle to reported token usage.
                estimated_tokens = (
                    max(1, len(json.dumps(payload, ensure_ascii=False).encode("utf-8")))
                    + spec.max_output_tokens
                )
                scope = (
                    self.request_scope(spec.provider, estimated_tokens)
                    if self.request_scope
                    else unlimited()
                )
                async with scope as reservation:
                    response = await self.http.post(
                        provider.base_url.rstrip("/") + "/chat/completions",
                        headers={"Authorization": "Bearer " + provider.api_key.get_secret_value()},
                        json=payload,
                        timeout=provider.timeout_seconds,
                    )
                    response.raise_for_status()
                    body = await invoke(response.json)
                    if self.usage_recorder:
                        await self.usage_recorder(reservation, body.get("usage"))
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
                    message=body["choices"][0]["message"],
                    usage=body.get("usage"),
                    attempts=attempt + 1,
                    response_model=body.get("model"),
                    response_id=body.get("id"),
                    raw_response=body,
                    request_payload=payload,
                )
            if self.events:
                await invoke(
                    self.events.emit,
                    "model.attempt_failed",
                    {"model": name, "stage": stage, "attempt": attempt + 1},
                )
            if delay is None:
                base = 1.0 if settings is None else settings.retry_base_seconds
                jitter = 0.0 if settings is None else settings.retry_jitter_seconds
                delay = max(0.0, base) * 2**attempt + random.uniform(0, max(0.0, jitter))
            await asyncio.sleep(delay)
        raise RuntimeError("No LLM response")
