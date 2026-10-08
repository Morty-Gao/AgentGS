"""Bounded real-provider acceptance checks; never a full paid benchmark run."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from agentgs.client.config import load_llm
from agentgs.evaluation.benchmark import public_input, score_prediction, usage_summary, write_json
from agentgs.evaluation.datasets.geotooleval import load_tasks
from agentgs.evaluation.datasets.version import benchmark_workbook
from agentgs.runtime.answer_contracts import AnswerContracts
from agentgs.runtime.service import TaskService


async def validate(root, ids, output, max_calls, timeout):
    config = load_llm(root)
    workbook = benchmark_workbook(root)
    tasks = {t.question_id: t for t in load_tasks(workbook)}
    contracts = AnswerContracts(root)
    report = {
        "protocol": "bounded_live_agent_validation",
        "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "question_ids": ids,
        "max_model_stages_per_task": max_calls,
        "timeout_per_task_seconds": timeout,
        "models": {name: model.model_dump(mode="json") for name, model in config.models.items()},
        "roles": config.roles,
        "provider_urls_sha256": {
            name: hashlib.sha256(p.base_url.encode()).hexdigest()
            for name, p in config.providers.items()
        },
        "results": [],
    }
    service = TaskService(root)
    try:
        for qid in ids:
            task = tasks[qid]
            request = public_input(task, output_contract=contracts.find(task.question))
            request.max_model_calls = max_calls
            record = service.submit(request, timeout_seconds=timeout)
            finished = await service.wait(record.spec.id)
            result = finished.result
            row = {
                "question_id": qid,
                "task": finished.model_dump(mode="json"),
                "scoring": score_prediction(
                    task,
                    result.answer if result else "",
                    result.trace if result else [],
                    usage_summary(result.usage if result else []),
                ),
            }
            report["results"].append(row)
            write_json(output, report)
            print(
                json.dumps(
                    {
                        "question_id": qid,
                        "status": finished.status.value,
                        "task_id": finished.spec.id,
                        "model_stages": len(result.usage) if result else 0,
                        "fas": row["scoring"]["scores"]["fas"],
                    }
                ),
                flush=True,
            )
            if finished.error and any(
                code in finished.error.message for code in ("HTTP 401", "HTTP 403")
            ):
                break
    finally:
        await service.close()


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default="61,86")
    parser.add_argument("--max-model-calls", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--output", type=Path, default=root / "outputs/verification/live.json")
    args = parser.parse_args()
    ids = [int(i) for i in args.ids.split(",")]
    if not 1 <= len(set(ids)) <= 5 or not 1 <= args.max_model_calls <= 12:
        parser.error("Validation accepts 1..5 distinct tasks and 1..12 model stages per task")
    asyncio.run(validate(root, ids, args.output, args.max_model_calls, args.timeout))


if __name__ == "__main__":
    main()
