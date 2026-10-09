"""Cancellation-safe child process lifecycle and bounded output collection."""

import asyncio
import signal

from agentgs.contracts.async_utils import drain, invoke
from agentgs.runtime.resources import kill_tree


async def stop(process, *, grace=0):
    if grace and process.returncode is None:
        try:
            process.send_signal(signal.SIGTERM)
            async with asyncio.timeout(grace):
                await process.wait()
        except (TimeoutError, ProcessLookupError):
            pass
    await invoke(kill_tree, process.pid)
    await process.wait()


async def spawn(*args, **kwargs):
    creation = asyncio.create_task(asyncio.create_subprocess_exec(*args, **kwargs))
    try:
        return await asyncio.shield(creation)
    except asyncio.CancelledError:
        process = await drain(creation)
        await drain(asyncio.create_task(stop(process)))
        raise


async def read_stream(stream, limit=1024 * 1024):
    if stream is None:
        return ""
    output = bytearray()
    while block := await stream.read(8192):
        output.extend(block)
        if len(output) > limit:
            raise ValueError("Process output exceeded its byte limit")
    return output.decode(errors="replace")


async def collect(process):
    readers = [
        asyncio.create_task(read_stream(process.stdout)),
        asyncio.create_task(read_stream(process.stderr)),
    ]
    try:
        stdout, stderr = await asyncio.gather(*readers)
        await process.wait()
        return stdout, stderr
    finally:
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
