"""Await injected async implementations and keep blocking callbacks off the loop."""

import asyncio
import inspect
from contextlib import suppress


async def invoke(function, /, *args, **kwargs):
    if inspect.iscoroutinefunction(function):
        return await function(*args, **kwargs)
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        # Finish the callback, but keep its failure from replacing cancellation.
        with suppress(Exception, asyncio.CancelledError):
            result = await drain(task)
            if inspect.iscoroutine(result):
                # A wrapper may have created a coroutine without starting it.
                result.close()
            elif inspect.isawaitable(result):
                pending = asyncio.ensure_future(result)
                pending.cancel()
                await drain(pending)
        raise
    if inspect.isawaitable(result):
        return await result
    return result


async def drain(task):
    """Finish cleanup even if the owning task receives repeated cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()
