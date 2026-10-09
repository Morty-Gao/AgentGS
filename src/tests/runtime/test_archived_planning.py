"""Archived results remain readable after the public planning-policy rename."""

import json

import pytest

from agentgs.contracts.models import AgentResult, TaskInput, TaskRecord, TaskSpec, TaskStatus
from agentgs.runtime.storage.store import TaskStore


@pytest.mark.parametrize("historical", [None, False, True])
async def test_archived_rule_policy_and_budgets_are_read_without_rewriting(project, historical):
    store = TaskStore(project / "archive")
    record = TaskRecord(
        spec=TaskSpec(input=TaskInput(question="Archived task")),
        status=TaskStatus.SUCCEEDED,
        result=AgentResult(answer="Original result"),
    )
    value = record.model_dump(mode="json")
    request = value["spec"]["input"]
    for field in ("allow_rule_planning", "plan_attempts", "tool_attempts"):
        request.pop(field)
    request.update(require_llm=historical, algorithm="legacy")
    trace = [{"type": "plan.created", "tools": ["HistoricalTool"]}]
    value["result"]["trace"] = trace
    encoded = json.dumps(value)
    with store.connection() as db:
        db.execute("INSERT INTO tasks VALUES (?,?)", (record.spec.id, encoded))
    restored = await store.get(record.spec.id)
    assert (await store.list())[0] == restored
    assert restored.spec.input.allow_rule_planning is (
        None if historical is None else not historical
    )
    assert restored.spec.input.plan_attempts == (
        None if historical is None else 1 if historical else 3
    )
    assert restored.spec.input.tool_attempts is None
    assert restored.result.answer == "Original result"
    assert restored.result.provenance["archived_trace"] == trace
    assert restored.result.messages == []
    with store.connection() as db:
        assert db.execute("SELECT record FROM tasks").fetchone()[0] == encoded
    # Admission must also tolerate old completed records while accepting only new inputs.
    fresh = TaskRecord(spec=TaskSpec(input=TaskInput(question="New task")))
    await store.reserve(fresh, 1)
    assert (await store.get(fresh.spec.id)).spec.input.allow_rule_planning is None
