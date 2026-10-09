"""Cross-process invocation capacity and rate limits, scoped to a project root.

SQLite transactions provide atomic admission across API instances and workers.
All database and process inspection work runs outside the event loop. Leases
belong to a PID plus its creation time, so crashed processes cannot hold slots.
"""

import asyncio
import hashlib
import os
import socket
import sqlite3
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4

import psutil

from agentgs.contracts.async_utils import drain, invoke

_tool_admission = ContextVar("tool_admission", default=None)


def worker_identity():
    """PID liveness is meaningful only within the same machine and namespaces."""
    parts = [socket.gethostname(), psutil.Process().username()]
    for name in ("/proc/self/ns/pid", "/proc/self/ns/cgroup", "/proc/sys/kernel/random/boot_id"):
        path = Path(name)
        try:
            parts.append(str(path.readlink()) if path.is_symlink() else path.read_text())
        except OSError:
            pass
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:20]


class RuntimeCoordinator:
    def __init__(self, root: Path):
        self.path = (
            Path(root).resolve()
            / "outputs/runtime/coordination"
            / worker_identity()
            / "coordination.sqlite3"
        )
        self.pid = os.getpid()
        self.identity = psutil.Process(self.pid).create_time()
        self._ready = False
        self._init_lock = asyncio.Lock()

    def _initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY, scope TEXT, pid INTEGER, identity REAL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS lease_scope ON leases(scope)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS policies (scope TEXT PRIMARY KEY, capacity INTEGER)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS rates (id TEXT PRIMARY KEY, scope TEXT, started REAL, tokens INTEGER)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS rate_scope ON rates(scope,started)")

    async def initialize(self):
        async with self._init_lock:
            if not self._ready:
                await invoke(self._initialize)
                self._ready = True

    @staticmethod
    def _alive(pid, identity):
        try:
            proc = psutil.Process(pid)
            return proc.create_time() == identity and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return True

    def _admit(self, lease, capacities, rate_scope, tokens, rpm, tpm):
        now = time.time()
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            availability = True
            living = {}
            for scope, capacity in capacities.items():
                rows = db.execute(
                    "SELECT id,pid,identity FROM leases WHERE scope=?", (scope,)
                ).fetchall()
                active = 0
                for key, pid, identity in rows:
                    owner = (pid, identity)
                    if owner not in living:
                        living[owner] = self._alive(pid, identity)
                    if living[owner]:
                        active += 1
                    else:
                        db.execute("DELETE FROM leases WHERE id=?", (key,))
                previous = db.execute(
                    "SELECT capacity FROM policies WHERE scope=?", (scope,)
                ).fetchone()
                if previous and previous[0] != capacity and active:
                    raise ValueError(
                        f"Conflicting active runtime capacity for {scope}: {previous[0]} vs {capacity}"
                    )
                db.execute(
                    "INSERT INTO policies VALUES (?,?) ON CONFLICT(scope) DO UPDATE SET capacity=excluded.capacity",
                    (scope, capacity),
                )
                availability &= active < capacity
            if not availability:
                return False
            if rate_scope and (rpm or tpm):
                db.execute("DELETE FROM rates WHERE started<=?", (now - 60,))
                count, used = db.execute(
                    "SELECT COUNT(*),COALESCE(SUM(tokens),0) FROM rates WHERE scope=?",
                    (rate_scope,),
                ).fetchone()
                if rpm and count >= rpm or tpm and used + tokens > tpm:
                    return False
                db.execute("INSERT INTO rates VALUES (?,?,?,?)", (lease, rate_scope, now, tokens))
            for index, scope in enumerate(capacities):
                db.execute(
                    "INSERT INTO leases VALUES (?,?,?,?)",
                    (lease + "_" + str(index), scope, self.pid, self.identity),
                )
            return True

    def _release(self, lease):
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("DELETE FROM leases WHERE id LIKE ?", (lease + "_%",))

    @asynccontextmanager
    async def invocation(self, capacities, *, rate_scope=None, tokens=0, rpm=None, tpm=None):
        if tpm and tokens > tpm:
            raise ValueError("A model request's reserved tokens exceed the configured TPM budget")
        await self.initialize()
        lease = uuid4().hex
        admission = None
        delay = 0.01
        try:
            while True:
                admission = asyncio.create_task(
                    asyncio.to_thread(self._admit, lease, capacities, rate_scope, tokens, rpm, tpm)
                )
                if await asyncio.shield(admission):
                    break
                await asyncio.sleep(delay)
                delay = min(0.2, delay * 1.5)
            yield lease if rate_scope and (rpm or tpm) else None
        finally:
            if admission is not None:
                await drain(admission)
            await drain(asyncio.create_task(asyncio.to_thread(self._release, lease)))

    @asynccontextmanager
    async def slot(self, scope, capacity):
        async with self.invocation({scope: capacity}):
            yield

    @asynccontextmanager
    async def tool_invocation(self, settings, timeout):
        """Share tool capacity with nested sandbox execution in the same task."""
        owner = (self.path.absolute(), asyncio.current_task(), settings.max_concurrent_tool_calls)
        async with asyncio.timeout(timeout):
            if _tool_admission.get() == owner:
                yield
                return
            async with self.slot("tools", settings.max_concurrent_tool_calls):
                token = _tool_admission.set(owner)
                try:
                    yield
                finally:
                    _tool_admission.reset(token)

    def _rate(self, key, scope, tokens, rpm, tpm):
        now = time.time()
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM rates WHERE started<=?", (now - 60,))
            count, used = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(tokens),0) FROM rates WHERE scope=?", (scope,)
            ).fetchone()
            if rpm and count >= rpm or tpm and used + tokens > tpm:
                return False
            db.execute("INSERT INTO rates VALUES (?,?,?,?)", (key, scope, now, tokens))
            return True

    async def rate(self, scope, tokens, *, rpm=None, tpm=None):
        if not rpm and not tpm:
            return None
        if tpm and tokens > tpm:
            raise ValueError("A model request's reserved tokens exceed the configured TPM budget")
        await self.initialize()
        key = uuid4().hex
        while not await invoke(self._rate, key, scope, tokens, rpm, tpm):
            await asyncio.sleep(0.05)
        return key

    async def settle(self, key, tokens):
        if key is not None and tokens is not None:

            def update():
                with sqlite3.connect(self.path, timeout=10) as db:
                    db.execute("UPDATE rates SET tokens=? WHERE id=?", (max(0, int(tokens)), key))

            await invoke(update)
