"""Async task persistence and cross-instance cancellation control."""

import asyncio
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import psutil

from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import TERMINAL, Event, TaskRecord, TaskStatus, ToolError
from agentgs.runtime.concurrency import RuntimeCoordinator, sqlite_busy, worker_identity
from agentgs.runtime.resources import kill_tree


class QueueFull(RuntimeError):
    pass


class _ClosingConnection:
    """One sqlite connection that commits and closes, and shares a process lock.

    ``with sqlite3.connect() as db`` commits but leaves the connection open.
    Concurrent ``BEGIN IMMEDIATE`` from the supervisor's thread pool then
    raises ``database is locked`` even though only one process is writing.
    """

    def __init__(self, path: Path, lock: threading.Lock, timeout: float = 30):
        self._path = path
        self._lock = lock
        self._timeout = timeout
        self._db = None
        self._locked = False

    def execute(self, *args, **kwargs):
        return self._db.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._db.executemany(*args, **kwargs)

    def __enter__(self):
        self._lock.acquire()
        self._locked = True
        try:
            self._db = sqlite3.connect(self._path, timeout=self._timeout)
            self._db.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)}")
            self._db.__enter__()
            return self
        except Exception:
            self._release()
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._db is not None:
                try:
                    return self._db.__exit__(exc_type, exc, traceback)
                finally:
                    self._db.close()
                    self._db = None
        finally:
            self._release()

    def _release(self):
        if self._locked:
            self._lock.release()
            self._locked = False


class TaskStore:
    def __init__(self, root: Path, *, initialize=True):
        self.path = root / "tasks.sqlite3"
        self.pool = worker_identity()
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._db_lock = threading.Lock()
        if initialize:
            self._initialize()

    def _initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, record TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS events (task_id TEXT, sequence INTEGER, record TEXT NOT NULL, PRIMARY KEY(task_id,sequence))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS controls (task_id TEXT PRIMARY KEY, owner_pid INTEGER, owner_identity REAL, cancelled INTEGER DEFAULT 0, worker_pid INTEGER, rss INTEGER DEFAULT 0)"
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(controls)")}
            if "worker_identity" not in columns:
                db.execute("ALTER TABLE controls ADD COLUMN worker_identity REAL")
            if "owner_pool" not in columns:
                db.execute("ALTER TABLE controls ADD COLUMN owner_pool TEXT")
            if "state" not in columns:
                db.execute("ALTER TABLE controls ADD COLUMN state TEXT NOT NULL DEFAULT 'queued'")
                rows = db.execute(
                    "SELECT c.task_id,t.record FROM controls c JOIN tasks t ON t.id=c.task_id"
                ).fetchall()
                db.executemany(
                    "UPDATE controls SET state=? WHERE task_id=?",
                    ((json.loads(encoded).get("status", "queued"), key) for key, encoded in rows),
                )
            db.execute("CREATE INDEX IF NOT EXISTS control_state ON controls(state)")
            db.execute(
                "CREATE INDEX IF NOT EXISTS control_pool_state ON controls(owner_pool,state)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS control_active_workers ON controls(owner_pool) WHERE worker_pid IS NOT NULL"
            )
        self._initialized = True

    async def initialize(self):
        async with self._init_lock:
            if not self._initialized:
                await invoke(self._initialize)

    def connection(self, timeout: float = 30):
        return _ClosingConnection(self.path, self._db_lock, timeout)

    def _retry_locked(self, operation):
        delay = 0.02
        for attempt in range(8):
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                if not sqlite_busy(exc) or attempt == 7:
                    raise
                time.sleep(delay)
                delay = min(0.5, delay * 2)

    def save_sync(self, record):
        def once():
            with self.connection() as db:
                db.execute(
                    "INSERT INTO tasks VALUES (?,?) ON CONFLICT(id) DO UPDATE SET record=excluded.record",
                    (record.spec.id, record.model_dump_json()),
                )
                db.execute(
                    "UPDATE controls SET state=? WHERE task_id=?",
                    (record.status.value, record.spec.id),
                )

        self._retry_locked(once)

    async def save(self, record):
        await self.initialize()
        await invoke(self.save_sync, record.model_copy(deep=True))

    def get_sync(self, task_id):
        def once():
            with self.connection() as db:
                row = db.execute("SELECT record FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                raise KeyError(task_id)
            return self._decode_record(row[0])

        return self._retry_locked(once)

    @staticmethod
    def _decode_record(encoded):
        """Read archived requests without reintroducing obsolete public options.

        Old records used require_llm to couple rule policy and plan recovery.
        Translate only at the storage boundary; keep the original database row.
        """
        value = json.loads(encoded)
        request = value["spec"]["input"]
        # The former staged runtime selector is also present in saved requests.
        # Reading a result does not select or execute that retired algorithm.
        request.pop("algorithm", None)
        if "require_llm" in request:
            historical = request.pop("require_llm")
            request.setdefault(
                "allow_rule_planning", None if historical is None else not historical
            )
            if historical is not None:
                request.setdefault("plan_attempts", 1 if historical else 3)
        result = value.get("result")
        if result is not None and "trace" in result:
            # Retain old evidence as metadata instead of inventing model calls.
            result.setdefault("provenance", {})["archived_trace"] = result.pop("trace")
        return TaskRecord.model_validate(value)

    async def get(self, task_id):
        await self.initialize()
        return await invoke(self.get_sync, task_id)

    def list_sync(self):
        with self.connection() as db:
            rows = db.execute("SELECT record FROM tasks ORDER BY rowid DESC").fetchall()
        return [self._decode_record(row[0]) for row in rows]

    async def list(self):
        await self.initialize()
        return await invoke(self.list_sync)

    @staticmethod
    def _emit(db, task_id, kind, payload):
        seq = db.execute(
            "SELECT COALESCE(MAX(sequence),0)+1 FROM events WHERE task_id=?", (task_id,)
        ).fetchone()[0]
        event = Event(task_id=task_id, sequence=seq, type=kind, payload=payload)
        db.execute("INSERT INTO events VALUES (?,?,?)", (task_id, seq, event.model_dump_json()))
        return event

    def emit_sync(self, task_id, kind, payload):
        def once():
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                return self._emit(db, task_id, kind, payload)

        return self._retry_locked(once)

    async def emit(self, task_id, kind, payload):
        await self.initialize()
        return await invoke(self.emit_sync, task_id, kind, payload)

    def events_sync(self, task_id, after=0):
        def once():
            with self.connection() as db:
                rows = db.execute(
                    "SELECT record FROM events WHERE task_id=? AND sequence>? ORDER BY sequence",
                    (task_id, after),
                ).fetchall()
            return [Event.model_validate_json(row[0]) for row in rows]

        return self._retry_locked(once)

    async def events(self, task_id, after=0):
        await self.initialize()
        return await invoke(self.events_sync, task_id, after)

    async def reserve(self, record, capacity):
        await self.initialize()
        await self.recover()
        pid, identity = os.getpid(), psutil.Process().create_time()

        def insert():
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                count = db.execute(
                    "SELECT COUNT(*) FROM controls WHERE owner_pool=? AND state IN ('queued','preparing','running')",
                    (self.pool,),
                ).fetchone()[0]
                if count >= capacity:
                    raise QueueFull("Runtime task queue is full")
                db.execute(
                    "INSERT INTO tasks VALUES (?,?)", (record.spec.id, record.model_dump_json())
                )
                db.execute(
                    "INSERT INTO controls(task_id,owner_pid,owner_identity,owner_pool) VALUES (?,?,?,?)",
                    (record.spec.id, pid, identity, self.pool),
                )
                self._emit(db, record.spec.id, "task.queued", {})

        await invoke(self._retry_locked, insert)

    async def control(self, task_id):
        await self.initialize()

        def read():
            with self.connection() as db:
                row = db.execute(
                    "SELECT owner_pid,owner_identity,cancelled,worker_pid,rss FROM controls WHERE task_id=?",
                    (task_id,),
                ).fetchone()
            return (
                dict(zip(("owner_pid", "owner_identity", "cancelled", "worker_pid", "rss"), row))
                if row
                else None
            )

        return await invoke(self._retry_locked, read)

    async def request_cancel(self, task_id):
        await self.initialize()

        def update():
            with self.connection() as db:
                db.execute("UPDATE controls SET cancelled=1 WHERE task_id=?", (task_id,))

        await invoke(self._retry_locked, update)

    async def worker(self, task_id, pid, rss=0, *, best_effort=False):
        """Record the worker pid and RSS. Polling skips a locked database.

        The RSS sample runs for every live task. On the shared GPFS database a
        locked write is not a task failure; dropping one sample keeps the worker
        alive. Registration and clearing the pid still wait and retry.
        """
        await self.initialize()

        def update():
            with self.connection(0.2 if best_effort else 30) as db:
                identity = None
                if pid is not None:
                    try:
                        identity = psutil.Process(pid).create_time()
                    except psutil.NoSuchProcess:
                        pid_value = None
                    else:
                        pid_value = pid
                else:
                    pid_value = None
                db.execute(
                    "UPDATE controls SET worker_pid=?,rss=?,worker_identity=? WHERE task_id=?",
                    (pid_value, rss if pid_value else 0, identity, task_id),
                )
                return db.execute(
                    "SELECT COALESCE(SUM(rss),0) FROM controls WHERE owner_pool=? AND worker_pid IS NOT NULL",
                    (self.pool,),
                ).fetchone()[0]

        if best_effort:
            try:
                return await invoke(update)
            except sqlite3.OperationalError as exc:
                if sqlite_busy(exc):
                    return None
                raise
        return await invoke(self._retry_locked, update)

    async def recover(self):
        """Reclaim queue entries and orphan workers left by dead supervisors."""
        await self.initialize()

        def reap():
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                rows = db.execute(
                    "SELECT c.task_id,c.owner_pid,c.owner_identity,c.worker_pid,c.worker_identity,t.record FROM controls c JOIN tasks t ON t.id=c.task_id WHERE c.owner_pool=? AND (c.state IN ('queued','preparing','running') OR c.worker_pid IS NOT NULL)",
                    (self.pool,),
                ).fetchall()
                for key, owner, identity, worker, worker_identity, encoded in rows:
                    record = self._decode_record(encoded)
                    owner_alive = RuntimeCoordinator._alive(owner, identity)
                    worker_alive = bool(
                        worker
                        and worker_identity
                        and RuntimeCoordinator._alive(worker, worker_identity)
                    )
                    if record.status in TERMINAL:
                        if worker_alive and not owner_alive:
                            kill_tree(worker)
                        if not worker_alive or not owner_alive:
                            db.execute(
                                "UPDATE controls SET worker_pid=NULL,worker_identity=NULL,rss=0 WHERE task_id=?",
                                (key,),
                            )
                        continue
                    if owner_alive:
                        continue
                    if worker_alive:
                        kill_tree(worker)
                    record.status = TaskStatus.FAILED
                    record.error = ToolError(
                        code="owner_lost", message="Task supervisor exited before completion"
                    )
                    db.execute(
                        "UPDATE tasks SET record=? WHERE id=?", (record.model_dump_json(), key)
                    )
                    db.execute(
                        "UPDATE controls SET worker_pid=NULL,worker_identity=NULL,rss=0,state='failed' WHERE task_id=?",
                        (key,),
                    )
                    self._emit(db, key, "task.failed", {"error": record.error.model_dump()})

        await invoke(self._retry_locked, reap)
