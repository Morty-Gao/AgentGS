"""Frozen GeoToolEval metrics, public inputs and append-only experiment attempts."""

import hashlib
import json
import os
from pathlib import Path

from agentgs.contracts.models import TaskInput
from agentgs.evaluation.datasets.geotooleval import load_tasks
from agentgs.evaluation.metrics.answers import score_answer
from agentgs.evaluation.metrics.tools import aggregate_scores, score_task

PRIVATE_KEYS = frozenset(
    {
        "gold_answer",
        "gold_sparql",
        "gold_chain",
        "gold_steps",
        "expected_answer",
        "reference_answer",
        "reference_tools",
    }
)


def public_resource(value):
    if isinstance(value, dict):
        return {
            key: public_resource(item) for key, item in value.items() if key not in PRIVATE_KEYS
        }
    if isinstance(value, list):
        return [public_resource(item) for item in value]
    return value


def public_input(task, *, planning_mode="skill_guided", require_llm=True, output_contract=None):
    try:
        resource = json.loads(task.input_resource) if task.input_resource else None
    except ValueError:
        resource = task.input_resource
    return TaskInput(
        question=task.question,
        input_resource=public_resource(resource),
        planning_mode=planning_mode,
        require_llm=require_llm,
        output_contract=output_contract,
        input_domain="remote_sensing" if task.category == "RemoteSensingTools" else None,
    )


def usage_summary(calls):
    usages = [call.get("usage") for call in calls or []]
    complete = (
        bool(usages)
        and all(
            isinstance(usage, dict) and usage.get("total_tokens") is not None for usage in usages
        )
        and all(call.get("attempts", 1) == 1 for call in calls or [])
    )
    return {
        "usage_complete": complete,
        "total_tokens": sum(
            int(usage.get("total_tokens") or 0) for usage in usages if isinstance(usage, dict)
        )
        if complete
        else None,
        "reported_tokens": sum(
            int(usage.get("total_tokens") or 0) for usage in usages if isinstance(usage, dict)
        ),
    }


def score_prediction(task, answer, trace=(), usage=None):
    final = score_answer(task, answer)
    scores = score_task(
        gold_steps=list(task.gold_steps),
        trace=list(trace),
        answer_score=final["score"],
        token_usage=usage,
    )
    return {
        "question_id": task.question_id,
        "category_id": task.category_id,
        "category": task.category,
        "answer_score": final,
        "scores": scores,
        "gold_steps": list(task.gold_steps),
    }


def summarize(rows, total=603):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["category_id"], []).append(row["scores"]["fas"])
    micro = sum(row["scores"]["fas"] for row in rows) / len(rows) if rows else None
    macro = (
        sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)
        if grouped
        else None
    )
    return {
        "submitted": len(rows),
        "expected": total,
        "complete": len(rows) == total and len(grouped) == 10,
        "micro_fas": micro,
        "macro_fas": macro,
        "full_score_rate": sum(row["scores"]["fas"] == 1 for row in rows) / len(rows)
        if rows
        else None,
        "domains": {str(k): {"n": len(v), "fas": sum(v) / len(v)} for k, v in grouped.items()},
        "composite_metrics": aggregate_scores(rows),
        "results": rows,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, default=str) + "\n"
    )
    tmp.replace(path)


def score_predictions(workbook, predictions, output, *, allow_partial=False):
    tasks = {task.question_id: task for task in load_tasks(workbook)}
    rows = []
    seen = set()
    for line in predictions.read_text("utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        qid = int(record.get("question_id", record.get("case_id")))
        if qid in seen or qid not in tasks:
            raise ValueError("Duplicate or unknown prediction ID")
        seen.add(qid)
        rows.append(
            score_prediction(
                tasks[qid],
                record.get("final_answer", record.get("answer", "")),
                record.get("trace", record.get("tool_trace", [])),
                record.get("token_usage", record.get("usage")),
            )
        )
    if not allow_partial and seen != set(tasks):
        raise ValueError("Complete benchmark scoring requires every frozen task ID")
    report = summarize(rows, len(tasks))
    manifest = workbook.parent / "version_manifest.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        root = workbook.parent.parent.parent
        path = root / metadata["compatibility_contracts"]
        formats = {
            row["question_id"]: row["answer_format"]
            for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        }
        from agentgs.evaluation.metrics.compatibility import VERSION, score_compatible

        records = [
            json.loads(line)
            for line in predictions.read_text("utf-8-sig").splitlines()
            if line.strip()
        ]
        compatibility = [
            {
                "question_id": int(record.get("question_id", record.get("case_id"))),
                "result": score_compatible(
                    tasks[int(record.get("question_id", record.get("case_id")))],
                    record.get("final_answer", record.get("answer", "")),
                    formats[int(record.get("question_id", record.get("case_id")))],
                ),
            }
            for record in records
        ]
        report["compatibility"] = {
            "protocol": VERSION,
            "format_version": "1.1",
            "results": compatibility,
            "note": "Separate historical compatibility scores; frozen FAS and composite metrics above are unchanged.",
        }
    write_json(output, report)
    return report


async def run_benchmark(
    service,
    workbook: Path,
    output: Path,
    *,
    ids=None,
    retry_failed=False,
    planning_mode="skill_guided",
    require_llm=True,
    contract_provider=None,
):
    tasks = load_tasks(workbook)
    selected = [task for task in tasks if ids is None or task.question_id in ids]
    input_hash = hashlib.sha256(workbook.read_bytes()).hexdigest()
    identity = {
        "schema_version": 1,
        "workbook_sha256": input_hash,
        "planning_mode": planning_mode,
        "require_llm": require_llm,
        "algorithm": "geotoolagent-geotooleval-101-snapshot-20261008",
        "protocol": "actual_agent_execution",
    }
    # Code and non-secret configuration determine an experiment's identity.
    code_files = sorted((service.root / "src/agentgs").rglob("*.py"))
    identity["code_sha256"] = hashlib.sha256(
        b"".join(path.read_bytes() for path in code_files)
    ).hexdigest()
    for name in ["agent.toml", "llm.toml", "runtime.toml"]:
        path = service.root / name
        identity[name + "_sha256"] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        )
    from agentgs.client.config import load_llm

    configuration = load_llm(service.root)
    identity["effective_models"] = {
        name: model.model_dump(mode="json") for name, model in configuration.models.items()
    }
    identity["roles"] = configuration.roles
    identity["providers"] = {
        name: {
            "protocol": provider.protocol,
            "base_url_sha256": hashlib.sha256(provider.base_url.encode()).hexdigest(),
            "timeout_seconds": provider.timeout_seconds,
        }
        for name, provider in configuration.providers.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError(
            "Code, model configuration, inputs or protocol differ; use a new run directory"
        )
    write_json(manifest, identity)
    lock = output / ".run.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        for task in selected:
            folder = output / "tasks" / f"{task.question_id:04d}"
            current = folder / "result.json"
            if current.exists():
                previous = json.loads(current.read_text())
                if not retry_failed or previous["status"] == "succeeded":
                    continue
            folder.mkdir(parents=True, exist_ok=True)
            number = len(list(folder.glob("attempt_*"))) + 1
            attempt = folder / f"attempt_{number:03d}"
            attempt.mkdir(exist_ok=False)
            request = public_input(
                task,
                planning_mode=planning_mode,
                require_llm=require_llm,
                output_contract=contract_provider.find(task.question)
                if contract_provider
                else None,
            )
            write_json(attempt / "public_input.json", request.model_dump(mode="json"))
            record = service.submit(request)
            result = await service.wait(record.spec.id)
            write_json(attempt / "task_record.json", result.model_dump(mode="json"))
            answer = result.result.answer if result.result else ""
            trace = result.result.trace if result.result else []
            tokens = usage_summary(result.result.usage if result.result else [])
            scored = score_prediction(task, answer, trace, tokens)
            row = {
                **scored,
                "status": result.status.value,
                "task_id": record.spec.id,
                "final_answer": answer,
                "attempt": number,
                "error": result.error.model_dump() if result.error else None,
                "trace": trace,
                "token_usage": tokens,
            }
            write_json(attempt / "score.json", row)
            write_json(current, row)
            # Authentication failures cannot be repaired by rerunning more tasks.
            if result.error and (
                "HTTP 401" in result.error.message or "HTTP 403" in result.error.message
            ):
                break
        rows = [
            json.loads(path.read_text())
            for path in sorted((output / "tasks").glob("*/result.json"))
        ]
        report = summarize(rows, len(tasks))
        write_json(output / "summary.json", report)
        with (output / "predictions.jsonl").open("w") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        {
                            "question_id": row["question_id"],
                            "final_answer": row["final_answer"],
                            "status": row["status"],
                            "trace": row.get("trace", []),
                            "token_usage": row.get("token_usage"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return report
    finally:
        lock.unlink(missing_ok=True)
