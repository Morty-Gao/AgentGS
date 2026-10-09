"""Exact-match runs for caller-supplied cases.

These cases already contain a ``TaskInput``. They are not an evaluation dataset:
gold expectations are compared here and never sent to the runtime.
"""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import Model, TaskInput, TaskStatus
from agentgs.evaluation.analysis.conversation import (
    as_conversation,
    conversation_usage,
    tool_executions,
)


class EvaluationCase(Model):
    id: str
    input: TaskInput
    expected_answer: str | None = None
    expected_tools: list[str] | None = None


def load_cases(path):
    cases = [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate evaluation case ids")
    return cases


async def evaluate(service, cases, output: Path):
    cases = list(cases)
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate evaluation case ids")
    run = output / uuid4().hex
    await invoke(run.mkdir, parents=True, exist_ok=False)

    async def one(case):
        # Only TaskInput crosses the runtime boundary; gold stays here.
        task = await service.submit(case.input)
        record = await service.wait(task.spec.id)
        conversation = as_conversation(record.result.messages if record.result else [])
        actual = [
            event["tool_name"]
            for event in tool_executions(conversation)
            if event["status"] == "success"
        ]
        row = {
            "case_id": case.id,
            "task_id": task.spec.id,
            "status": record.status.value,
            "tool_chain_match": actual == case.expected_tools
            if case.expected_tools is not None
            else None,
            "actual_tools": actual,
            "final_answer": record.result.answer if record.result else "",
            "answer_exact_match": record.result.answer.strip() == case.expected_answer.strip()
            if record.result and case.expected_answer is not None
            else None,
            "messages": conversation.model_dump(mode="json")["messages"],
            "token_usage": conversation_usage(
                conversation, record.result.usage if record.result else None
            ),
            "usage": record.result.usage if record.result else None,
            "error": record.error.model_dump() if record.error else None,
        }
        name = case.id.encode().hex()
        await invoke(conversation.save, run / f"{name}.messages.json")
        await invoke(
            (run / (name + ".json")).write_text,
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return row

    semaphore = asyncio.Semaphore(service.settings.max_concurrent_tasks)

    async def bounded(case):
        async with semaphore:
            return await one(case)

    rows = await asyncio.gather(*(bounded(case) for case in cases))
    await invoke(
        (run / "summary.json").write_text,
        json.dumps(
            {
                "cases": len(rows),
                "succeeded": sum(row["status"] == TaskStatus.SUCCEEDED for row in rows),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return run
