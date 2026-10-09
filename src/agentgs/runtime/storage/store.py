"""Async task persistence and cross-instance cancellation control."""

import asyncio
import json
import os
import sqlite3
from pathlib import Path

import psutil

from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import TERMINAL, Event, TaskRecord, TaskStatus, ToolError
from agentgs.runtime.concurrency import RuntimeCoordinator
from agentgs.runtime.resources import kill_tree


class QueueFull(RuntimeError):
    pass


class TaskStore:
    def __init__(self, root: Path, *, initialize=True):
        self.path = root / "tasks.sqlite3"
        self._initialized = False
        self._init_lock = asyncio.Lock()
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
        self._initialized = True

    async def initialize(self):
        async with self._init_lock:
            if not self._initialized:
                await invoke(self._initialize)

    def connection(self):
        return sqlite3.connect(self.path, timeout=10)

    def save_sync(self, record):
        with self.connection() as db:
            db.execute(
                "INSERT INTO tasks VALUES (?,?) ON CONFLICT(id) DO UPDATE SET record=excluded.record",
                (record.spec.id, record.model_dump_json()),
            )

    async def save(self, record):
        await self.initialize()
        await invoke(self.save_sync, record.model_copy(deep=True))

    def get_sync(self, task_id):
        with self.connection() as db:
            row = db.execute("SELECT record FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise KeyError(task_id)
        return self._decode_record(row[0])

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
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._emit(db, task_id, kind, payload)

    async def emit(self, task_id, kind, payload):
        await self.initialize()
        return await invoke(self.emit_sync, task_id, kind, payload)

    def events_sync(self, task_id, after=0):
        with self.connection() as db:
            rows = db.execute(
                "SELECT record FROM events WHERE task_id=? AND sequence>? ORDER BY sequence",
                (task_id, after),
            ).fetchall()
        return [Event.model_validate_json(row[0]) for row in rows]

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
                rows = db.execute("SELECT record FROM tasks").fetchall()
                count = sum(self._decode_record(row[0]).status not in TERMINAL for row in rows)
                if count >= capacity:
                    raise QueueFull("Runtime task queue is full")
                db.execute(
                    "INSERT INTO tasks VALUES (?,?)", (record.spec.id, record.model_dump_json())
                )
                db.execute(
                    "INSERT INTO controls(task_id,owner_pid,owner_identity) VALUES (?,?,?)",
                    (record.spec.id, pid, identity),
                )
                self._emit(db, record.spec.id, "task.queued", {})

        await invoke(insert)

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

        return await invoke(read)

    async def request_cancel(self, task_id):
        await self.initialize()

        def update():
            with self.connection() as db:
                db.execute("UPDATE controls SET cancelled=1 WHERE task_id=?", (task_id,))

        await invoke(update)

    async def worker(self, task_id, pid, rss=0):
        await self.initialize()

        def update():
            with self.connection() as db:
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
                return db.execute("SELECT COALESCE(SUM(rss),0) FROM controls").fetchone()[0]

        return await invoke(update)

    async def recover(self):
        """Reclaim queue entries and orphan workers left by dead supervisors."""
        await self.initialize()

        def reap():
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                rows = db.execute(
                    "SELECT c.task_id,c.owner_pid,c.owner_identity,c.worker_pid,c.worker_identity,t.record FROM controls c JOIN tasks t ON t.id=c.task_id"
                ).fetchall()
                for key, owner, identity, worker, worker_identity, encoded in rows:
                    record = self._decode_record(encoded)
                    if record.status in TERMINAL or RuntimeCoordinator._alive(owner, identity):
                        continue
                    if (
                        worker
                        and worker_identity
                        and RuntimeCoordinator._alive(worker, worker_identity)
                    ):
                        kill_tree(worker)
                    record.status = TaskStatus.FAILED
                    record.error = ToolError(
                        code="owner_lost", message="Task supervisor exited before completion"
                    )
                    db.execute(
                        "UPDATE tasks SET record=? WHERE id=?", (record.model_dump_json(), key)
                    )
                    db.execute(
                        "UPDATE controls SET worker_pid=NULL,worker_identity=NULL,rss=0 WHERE task_id=?",
                        (key,),
                    )
                    self._emit(db, key, "task.failed", {"error": record.error.model_dump()})

        await invoke(reap)
