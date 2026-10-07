import sqlite3
from pathlib import Path

from agentgs.contracts.models import Event, TaskRecord


class TaskStore:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "tasks.sqlite3"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, record TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS events (task_id TEXT, sequence INTEGER, record TEXT NOT NULL, PRIMARY KEY(task_id,sequence))"
            )

    def connection(self):
        return sqlite3.connect(self.path, timeout=10)

    def save(self, record):
        with self.connection() as db:
            db.execute(
                "INSERT INTO tasks VALUES (?,?) ON CONFLICT(id) DO UPDATE SET record=excluded.record",
                (record.spec.id, record.model_dump_json()),
            )

    def get(self, task_id):
        with self.connection() as db:
            row = db.execute("SELECT record FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise KeyError(task_id)
        return TaskRecord.model_validate_json(row[0])

    def list(self):
        with self.connection() as db:
            rows = db.execute("SELECT record FROM tasks ORDER BY rowid DESC").fetchall()
        return [TaskRecord.model_validate_json(row[0]) for row in rows]

    def emit(self, task_id, kind, payload):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            seq = db.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 FROM events WHERE task_id=?", (task_id,)
            ).fetchone()[0]
            event = Event(task_id=task_id, sequence=seq, type=kind, payload=payload)
            db.execute("INSERT INTO events VALUES (?,?,?)", (task_id, seq, event.model_dump_json()))
        return event

    def events(self, task_id, after=0):
        with self.connection() as db:
            rows = db.execute(
                "SELECT record FROM events WHERE task_id=? AND sequence>? ORDER BY sequence",
                (task_id, after),
            ).fetchall()
        return [Event.model_validate_json(row[0]) for row in rows]
