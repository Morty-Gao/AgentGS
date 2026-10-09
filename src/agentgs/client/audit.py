"""Task-local transport auditing, independent of the provider and its credentials."""

from contextvars import ContextVar

recorder = ContextVar("model_transport_recorder", default=None)


async def record(kind, payload):
    callback = recorder.get()
    if callback is not None:
        await callback(kind, payload)
