"""Busy and locked sqlite errors stay inside admission and task writes."""

import os
import sqlite3

import pytest

from agentgs.contracts.models import TaskInput, TaskRecord, TaskSpec, TaskStatus
from agentgs.runtime.concurrency import RuntimeCoordinator, sqlite_busy
from agentgs.runtime.storage.store import TaskStore


def _record():
    return TaskRecord(spec=TaskSpec(input=TaskInput(question="q")))


def test_sqlite_busy_matches_text_and_primary_code():
    locked = sqlite3.OperationalError("database is locked")
    busy = sqlite3.OperationalError("database is busy")
    coded = sqlite3.OperationalError("disk I/O error")
    coded.sqlite_errorcode = 261
    other = sqlite3.OperationalError("no such table: tasks")
    assert sqlite_busy(locked) and sqlite_busy(busy) and sqlite_busy(coded)
    assert not sqlite_busy(other)


def test_admit_treats_locked_without_errorcode_as_not_admitted(tmp_path, monkeypatch):
    import agentgs.runtime.concurrency as concurrency

    coordinator = RuntimeCoordinator(tmp_path)

    def connect(*args, **kwargs):
        raise sqlite3.OperationalError(connect.message)

    for message in ("database is locked", "database is busy"):
        connect.message = message
        monkeypatch.setattr(concurrency.sqlite3, "connect", connect)
        assert coordinator._admit("lease", {"tasks": 1}, None, 0, None, None) is False

    def missing_code(*args, **kwargs):
        exc = sqlite3.OperationalError("disk I/O error")
        exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
        raise exc

    monkeypatch.setattr(concurrency.sqlite3, "connect", missing_code)
    assert coordinator._admit("lease", {"tasks": 1}, None, 0, None, None) is False

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("no such table: leases")

    monkeypatch.setattr(concurrency.sqlite3, "connect", broken)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        coordinator._admit("lease", {"tasks": 1}, None, 0, None, None)


class _Connection:
    def __init__(self, db, fail):
        self._db = db
        self._fail = fail

    def execute(self, statement, parameters=()):
        if self._fail["left"] and self._fail["needle"] in statement:
            self._fail["left"] -= 1
            self._fail["hits"] += 1
            raise self._fail["error"]
        return self._db.execute(statement, parameters)

    def __enter__(self):
        self._db.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._db.__exit__(exc_type, exc, traceback)


def _patch_connection(monkeypatch, store, needle, error, left):
    real = store.connection
    state = {"left": left, "hits": 0, "needle": needle, "error": error}

    def connection(*args, **kwargs):
        return _Connection(real(*args, **kwargs), state)

    monkeypatch.setattr(store, "connection", connection)
    return state


async def test_reserve_retries_a_transient_lock_without_an_errorcode(project, monkeypatch):
    store = TaskStore(project / "storage")
    state = _patch_connection(
        monkeypatch,
        store,
        "COUNT(*)",
        sqlite3.OperationalError("database is locked"),
        1,
    )
    record = _record()
    await store.reserve(record, 4)
    assert state["hits"] == 1
    assert state["left"] == 0
    assert (await store.get(record.spec.id)).status == TaskStatus.QUEUED


async def test_reserve_does_not_swallow_unrelated_sqlite_errors(project, monkeypatch):
    store = TaskStore(project / "storage")
    state = _patch_connection(
        monkeypatch,
        store,
        "COUNT(*)",
        sqlite3.OperationalError("no such table: tasks"),
        3,
    )
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        await store.reserve(_record(), 4)
    assert state["hits"] == 1


async def test_save_and_emit_retry_transient_locks(project, monkeypatch):
    store = TaskStore(project / "storage")
    record = _record()
    await store.reserve(record, 4)
    state = _patch_connection(
        monkeypatch,
        store,
        "INSERT INTO tasks",
        sqlite3.OperationalError("database is busy"),
        1,
    )
    record.status = TaskStatus.RUNNING
    await store.save(record)
    assert state["hits"] == 1
    assert (await store.get(record.spec.id)).status == TaskStatus.RUNNING
    emit_state = _patch_connection(
        monkeypatch,
        store,
        "BEGIN IMMEDIATE",
        sqlite3.OperationalError("database is locked"),
        1,
    )
    event = await store.emit(record.spec.id, "task.note", {"ok": True})
    assert emit_state["hits"] == 1
    assert event.sequence == 2


async def test_worker_retries_a_transient_lock(project, monkeypatch):
    store = TaskStore(project / "storage")
    record = _record()
    await store.reserve(record, 4)
    state = _patch_connection(
        monkeypatch,
        store,
        "UPDATE controls",
        sqlite3.OperationalError("database is locked"),
        1,
    )
    total = await store.worker(record.spec.id, os.getpid(), 10)
    assert state["hits"] == 1
    assert total == 10


async def test_worker_sample_skips_a_locked_database(project, monkeypatch):
    store = TaskStore(project / "storage")
    record = _record()
    await store.reserve(record, 4)
    state = _patch_connection(
        monkeypatch,
        store,
        "UPDATE controls",
        sqlite3.OperationalError("database is locked"),
        5,
    )
    assert await store.worker(record.spec.id, os.getpid(), 10, best_effort=True) is None
    assert state["hits"] == 1
