"""Read-only projections of persisted tasks and atomic worker checkpoints."""

import json
from pathlib import Path

from agentgs.config.paths import within
from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import TERMINAL, AgentResult, TaskRecord, TaskStatus
from agentgs.runtime.storage.store import TaskStore

from .models import TaskPage, TaskSummary


async def task_page(
    store: TaskStore, *, limit: int, offset: int, status: TaskStatus | None, search: str
) -> TaskPage:
    await store.initialize()

    def read():
        # Project in SQLite so polling never deserializes conversation payloads.
        columns = """
            id, json_extract(record,'$.spec.input.question'),
            (SELECT json_group_array(json_extract(value,'$.tool'))
             FROM json_each(record,'$.spec.input.calls')),
            json_extract(record,'$.status'), json_extract(record,'$.created_at'),
            json_extract(record,'$.spec.input.planning_mode'),
            json_extract(record,'$.spec.input.model'),
            coalesce(json_extract(record,'$.peak_rss_bytes'),0),
            json_extract(record,'$.result.completion_status'),
            json_extract(record,'$.error.code')
        """
        clauses, params = [], []
        if status is not None:
            clauses.append("json_extract(record,'$.status')=?")
            params.append(status.value)
        if search:
            clauses.append("""(instr(lower(id),lower(?))>0
                OR instr(lower(coalesce(json_extract(record,'$.spec.input.question'),'')),lower(?))>0
                OR EXISTS (SELECT 1 FROM json_each(record,'$.spec.input.calls')
                  WHERE instr(lower(json_extract(value,'$.tool')),lower(?))>0))""")
            params.extend([search] * 3)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with store.connection() as db:
            db.execute("BEGIN")
            total = db.execute("SELECT count(*) FROM tasks" + where, params).fetchone()[0]
            counts = dict(
                db.execute("SELECT json_extract(record,'$.status'),count(*) FROM tasks GROUP BY 1")
            )
            rows = db.execute(
                "SELECT "
                + columns
                + " FROM tasks"
                + where
                + " ORDER BY rowid DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        keys = tuple(TaskSummary.model_fields)
        items = []
        for row in rows:
            value = dict(zip(keys, row))
            value["question"] = value["question"] or ""
            value["tools"] = json.loads(value["tools"])
            items.append(TaskSummary.model_validate(value))
        return TaskPage(items=items, total=total, limit=limit, offset=offset, counts=counts)

    return await invoke(read)


async def with_checkpoint(root: Path, record: TaskRecord) -> TaskRecord:
    if record.status in TERMINAL or record.result is not None:
        return record

    def read():
        try:
            path = within(root / "outputs/tasks", record.spec.id + "/result.json")
            result = AgentResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return record
        return record.model_copy(update={"result": result})

    return await invoke(read)
