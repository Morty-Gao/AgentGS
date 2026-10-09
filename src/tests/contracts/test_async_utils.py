import asyncio
import inspect
import threading

import pytest

from agentgs.contracts.async_utils import drain, invoke


@pytest.mark.parametrize("outcome", ["value", "error", "coroutine", "future", "task", "task_error"])
async def test_cancelled_invoke_finishes_callback_and_preserves_cancellation(outcome):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    returned = []
    operation_started = asyncio.Event()
    operation_finished = asyncio.Event()

    async def operation():
        operation_started.set()
        try:
            await asyncio.Future()
        finally:
            operation_finished.set()
            if outcome == "task_error":
                raise RuntimeError("Returned task failed during cleanup")

    pending = None
    if outcome == "future":
        pending = loop.create_future()
    elif outcome in {"task", "task_error"}:
        pending = asyncio.create_task(operation())
        await operation_started.wait()

    def callback():
        loop.call_soon_threadsafe(started.set)
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("Test did not release the callback")
            if outcome == "error":
                raise RuntimeError("Callback failed during cancellation cleanup")
            if outcome == "coroutine":
                result = operation()
                returned.append(result)
                return result
            return pending if pending is not None else 42
        finally:
            finished.set()

    task = asyncio.create_task(invoke(callback))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel("request cancelled")
        # The callback must finish even if cancellation is requested again.
        loop.call_soon(task.cancel, "cancel again")
        release.set()
        with pytest.raises(asyncio.CancelledError, match="request cancelled"):
            await task
        assert task.cancelled()
        assert finished.is_set()
        if returned:
            assert inspect.getcoroutinestate(returned[0]) == inspect.CORO_CLOSED
            assert not operation_started.is_set()
        if pending is not None:
            assert pending.done()
            if outcome == "task_error":
                assert isinstance(pending.exception(), RuntimeError)
            else:
                assert pending.cancelled()
        if outcome in {"task", "task_error"}:
            assert operation_finished.is_set()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for coroutine in returned:
            coroutine.close()
        if pending is not None:
            pending.cancel()
            if isinstance(pending, asyncio.Task):
                await asyncio.gather(pending, return_exceptions=True)


async def test_invoke_keeps_normal_sync_async_and_wrapped_async_results():
    async def add(a, *, b):
        return a + b

    assert await invoke(lambda a, *, b: a + b, 2, b=3) == 5
    assert await invoke(add, 2, b=3) == 5
    assert await invoke(lambda a, *, b: add(a, b=b), 2, b=3) == 5


async def test_uncancelled_callback_failure_still_propagates():
    def callback():
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        await invoke(callback)


async def test_drain_still_propagates_cleanup_failures():
    async def cleanup():
        raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await drain(asyncio.create_task(cleanup()))
