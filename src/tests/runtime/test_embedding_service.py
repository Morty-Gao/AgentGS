import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from agentgs.client.embedding import LocalEmbedding, save_embeddings
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.runtime.embeddings import EmbeddingRuntime, RemoteEmbedding


async def test_runtime_warms_once_and_shares_vectors_across_processes(tmp_path, monkeypatch):
    calls = []

    def encode(texts, **kwargs):
        calls.extend(texts)
        return iter([np.array([1.0, 2.0]) for _ in texts])

    monkeypatch.setattr(LocalEmbedding, "_load_model", lambda self: SimpleNamespace(embed=encode))
    save_embeddings(tmp_path / "embedding/questions.npz", ["prepared"], [[3.0, 4.0]])
    runtime = EmbeddingRuntime(tmp_path)
    options = AlgorithmSettings()
    try:
        endpoints = await asyncio.gather(*(runtime.endpoint(options) for _ in range(4)))
        assert all(e == endpoints[0] for e in endpoints)
        script = (
            "import json,sys; from agentgs.runtime.embeddings import RemoteEmbedding; "
            "c=RemoteEmbedding(json.loads(sys.stdin.read())); "
            "print(json.dumps([c.get_text_embedding('prepared'),c.get_text_embedding('new')]))"
        )
        for _ in range(2):
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            out, _ = await child.communicate(json.dumps(endpoints[0]).encode())
            assert child.returncode == 0
            assert json.loads(out) == [[3.0, 4.0], [1.0, 2.0]]
        assert calls == ["geoscience", "new"]
        with pytest.raises(ValueError, match="configuration changed"):
            await runtime.endpoint(options.model_copy(update={"embedding_threads": 9}))
    finally:
        await runtime.close()
    assert runtime.starting.result().embedder is None
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.endpoint(options)
    with pytest.raises(OSError):
        await asyncio.to_thread(RemoteEmbedding(endpoints[0]).get_text_embedding, "prepared")


async def test_cancelled_startup_is_owned_and_closed_by_runtime(tmp_path, monkeypatch):
    import threading

    started, release = threading.Event(), threading.Event()

    def warmup(self):
        started.set()
        release.wait(5)

    monkeypatch.setattr(LocalEmbedding, "warmup", warmup)
    runtime = EmbeddingRuntime(tmp_path)
    task = asyncio.create_task(runtime.endpoint(AlgorithmSettings()))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await runtime.close()
    assert not runtime.starting.result().thread.is_alive()
