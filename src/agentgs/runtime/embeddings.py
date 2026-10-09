"""Service-owned CPU encoder shared by isolated task processes over loopback JSON."""

import asyncio
import hmac
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import ProxyHandler, Request, build_opener

from agentgs.client.embedding import LocalEmbedding
from agentgs.contracts.async_utils import invoke


class EmbeddingServer:
    def __init__(self, embedder):
        self.embedder = embedder
        self.token = secrets.token_hex(32)
        self.embedder.warmup()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if not hmac.compare_digest(self.headers.get("Authorization", ""), owner.token):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4 * 1024 * 1024:
                        raise ValueError("Invalid embedding request length")
                    request = json.loads(self.rfile.read(length))
                    mode = request.get("mode")
                    if mode == "stats":
                        result = {"pid": os.getpid(), **owner.embedder.stats}
                    else:
                        text = request.get("text")
                        if not isinstance(text, str) or mode not in {"text", "query"}:
                            raise ValueError("Invalid embedding request")
                        encode = (
                            owner.embedder.get_text_embedding
                            if mode == "text"
                            else owner.embedder.get_query_embedding
                        )
                        result = {"vector": encode(text)}
                    status = 200
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    status = 500
                payload = json.dumps(result, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def endpoint(self):
        return {"url": f"http://127.0.0.1:{self.server.server_port}", "token": self.token}

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.embedder = None


class RemoteEmbedding:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def request(self, mode, text=None):
        request = Request(
            self.endpoint["url"],
            data=json.dumps({"mode": mode, "text": text}).encode(),
            headers={"Authorization": self.endpoint["token"], "Content-Type": "application/json"},
        )
        # The local service must not be routed through an external model proxy.
        with build_opener(ProxyHandler({})).open(request, timeout=120) as response:
            return json.load(response)

    def get_text_embedding(self, text):
        return self.request("text", text)["vector"]

    def get_query_embedding(self, text):
        return self.request("query", text)["vector"]


class EmbeddingRuntime:
    def __init__(self, root):
        self.root = root
        self.starting = None
        self.key = None
        self.closed = False

    async def endpoint(self, options):
        if self.closed:
            raise RuntimeError("Embedding runtime is closed")
        key = (
            options.model_cache,
            options.embedding_model,
            options.embedding_threads,
            options.embedding_batch_size,
        )
        if self.starting is None:
            self.key = key

            def start():
                return EmbeddingServer(
                    LocalEmbedding(
                        self.root / options.model_cache,
                        options.embedding_model,
                        options.embedding_threads,
                        options.embedding_batch_size,
                        store=self.root / "embedding",
                    )
                )

            self.starting = asyncio.create_task(invoke(start))
        elif key != self.key:
            raise ValueError("Embedding configuration changed during the runtime session")
        server = await asyncio.shield(self.starting)
        return server.endpoint()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.starting is not None:
            try:
                server = await self.starting
            except Exception:
                return
            await invoke(server.close)
