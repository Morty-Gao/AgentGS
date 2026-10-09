"""Runtime ownership of model capacity, provider rate limits and deadlines."""

import asyncio
from contextlib import asynccontextmanager

from agentgs.client.errors import TransportFailure
from agentgs.contracts.async_utils import drain


class ModelExecutor:
    def __init__(self, client, settings, coordinator):
        self.client, self.settings, self.coordinator = client, settings, coordinator
        self._closed = False
        self._tasks = set()
        self._managed_transport = hasattr(client, "request_scope")
        if self._managed_transport:
            client.request_scope = self.request_scope
            client.usage_recorder = self.record_usage

    @asynccontextmanager
    async def request_scope(self, provider_name, estimated_tokens):
        provider = self.client.config.providers[provider_name]
        capacities = {
            "models": self.settings.max_concurrent_model_calls,
            "provider:" + provider_name: provider.max_concurrent_requests
            or self.settings.max_concurrent_model_calls,
        }
        async with self.coordinator.invocation(
            capacities,
            rate_scope="provider:" + provider_name,
            tokens=estimated_tokens,
            rpm=provider.requests_per_minute,
            tpm=provider.tokens_per_minute,
        ) as reservation:
            yield reservation

    async def record_usage(self, reservation, usage):
        await self.coordinator.settle(
            reservation, usage.get("total_tokens") if isinstance(usage, dict) else None
        )

    async def chat(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Model executor is closed")
        task = asyncio.current_task()
        self._tasks.add(task)
        try:
            async with asyncio.timeout(self.settings.model_timeout_seconds):
                if self._managed_transport:
                    return await self.client.chat(*args, **kwargs)
                async with self.coordinator.slot(
                    "models", self.settings.max_concurrent_model_calls
                ):
                    return await self.client.chat(*args, **kwargs)
        except TimeoutError as exc:
            raise TransportFailure("LLM request deadline exceeded") from exc
        finally:
            self._tasks.discard(task)

    async def aclose(self):
        self._closed = True
        caller = asyncio.current_task()

        async def close():
            tasks = [task for task in self._tasks if task is not caller]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.client.aclose()

        await drain(asyncio.create_task(close()))
