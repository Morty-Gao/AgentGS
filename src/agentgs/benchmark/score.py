"""Score saved answers. This reads a dataset and a prediction file; it does not run the agent."""

import json
from pathlib import Path

from agentgs.benchmark.io import write_json
from agentgs.benchmark.selection import resolve_dataset
from agentgs.evaluation.datasets.version import compatibility_contracts
from agentgs.evaluation.metrics.report import score_prediction, summarize


def score_predictions(
    dataset, predictions, output, *, allow_partial=False, project=None, benchmark_scope="full"
):
    source = resolve_dataset(dataset)
    from agentgs.benchmark.scope import select_scope

    population, access_policy = select_scope(
        source.load(), benchmark_scope, planning_mode="tool_only"
    )
    tasks = {task.question_id: task for task in population}
    rows = []
    seen = set()
    records = []
    for line in Path(predictions).read_text("utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Each prediction must be an object")
        if record.get("trace") and "messages" not in record and "conversation" not in record:
            raise ValueError(
                "Trace-only predictions are unsupported; provide Conversation messages"
            )
        qid = _prediction_id(record, tasks)
        if qid in seen or qid not in tasks:
            raise ValueError("Duplicate or unknown prediction ID")
        seen.add(qid)
        records.append(record)
        conversation = record.get("conversation", record.get("messages", []))
        rows.append(
            score_prediction(
                tasks[qid],
                record.get("final_answer", record.get("answer", "")),
                conversation,
                record.get("token_usage", record.get("usage")),
            )
        )
    if not allow_partial and seen != set(tasks):
        raise ValueError("Complete benchmark scoring requires every frozen task ID")
    domains = (
        len({task.category_id for task in population})
        if benchmark_scope == "common_data"
        else source.domain_count
    )
    report = summarize(rows, len(tasks), domain_count=domains)
    report["data_access_policy"] = access_policy
    path = compatibility_contracts(source.source, project=project)
    if path is not None:
        formats = {
            row["question_id"]: row["answer_format"]
            for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        }
        from agentgs.evaluation.metrics.compatibility import VERSION, score_compatible

        compatibility = []
        for record in records:
            question_id = _prediction_id(record, tasks)
            compatibility.append(
                {
                    "question_id": question_id,
                    "result": score_compatible(
                        tasks[question_id],
                        record.get("final_answer", record.get("answer", "")),
                        formats[question_id],
                    ),
                }
            )
        report["compatibility"] = {
            "protocol": VERSION,
            "results": compatibility,
            "note": (
                "Separate historical compatibility scores; answer FAS and "
                "conversation metrics above are unchanged."
            ),
        }
    write_json(output, report)
    return report


def _prediction_id(record, tasks):
    raw = record.get("question_id", record.get("case_id"))
    if raw in tasks:
        return raw
    if isinstance(raw, str) and raw.strip().isdigit() and int(raw) in tasks:
        return int(raw)
    if isinstance(raw, int) and not isinstance(raw, bool) and str(raw) in tasks:
        return str(raw)
    return raw
