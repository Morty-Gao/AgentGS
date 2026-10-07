import json
from pathlib import Path
from uuid import uuid4

from agentgs.contracts.models import TaskStatus
from agentgs.evaluation.models import EvaluationCase


async def evaluate(service, cases, output: Path):
    run = output / uuid4().hex
    run.mkdir(parents=True, exist_ok=False)

    async def one(case):
        # Only TaskInput crosses the runtime boundary; gold stays here.
        task = service.submit(case.input)
        record = await service.wait(task.spec.id)
        actual = [result.tool for result in record.result.tool_results] if record.result else []
        row = {
            "case_id": case.id,
            "task_id": task.spec.id,
            "status": record.status.value,
            "tool_chain_match": actual == case.expected_tools if case.expected_tools else None,
            "answer_exact_match": record.result.answer.strip() == case.expected_answer.strip()
            if record.result and case.expected_answer is not None
            else None,
            "usage": record.result.usage if record.result else None,
            "error": record.error.model_dump() if record.error else None,
        }
        (run / (case.id.encode().hex() + ".json")).write_text(
            json.dumps(row, ensure_ascii=False, indent=2)
        )
        return row

    import asyncio

    rows = await asyncio.gather(*(one(case) for case in cases))
    (run / "summary.json").write_text(
        json.dumps(
            {
                "cases": len(rows),
                "succeeded": sum(row["status"] == TaskStatus.SUCCEEDED for row in rows),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return run


def load_cases(path):
    cases = [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("Duplicate evaluation case ids")
    return cases
