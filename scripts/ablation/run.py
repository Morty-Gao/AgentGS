"""Preview or execute frozen experiment cells using the existing AgentGS adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from _study import (
    DEFAULT_STUDY,
    ROOT,
    data_inventory,
    digest,
    file_hash,
    load_study,
    request_for,
    source_snapshot,
    worker_only,
    write_json,
)

sys.path.insert(0, str(ROOT / "scripts/benchmark"))
from _agentgs import classify_result, run_batch  # noqa: E402

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.async_utils import invoke
from agentgs.evaluation.datasets.geotooleval import GeoToolEvalDataset
from agentgs.runtime.answer_contracts import AnswerContracts
from agentgs.runtime.service import TaskService


def scorer():
    spec = importlib.util.spec_from_file_location(
        "module_study_scorer", ROOT / "scripts/evaluation/agentgs/score.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def entries(study, suite, cells, repeats):
    if suite == "pilot":
        rows = [
            {"cell": row["cell"], "repeat": 0, "suite": "pilot", "run_id": row["cell"] + "/pilot"}
            for row in study["schedule"]
            if row["suite"] == "primary" and row["repeat"] == 1
        ]
    else:
        rows = [row for row in study["schedule"] if suite == "all" or row["suite"] == suite]
    if cells:
        unknown = set(cells) - {row["cell"] for row in rows}
        if unknown:
            raise ValueError(
                f"Cells are unavailable in this suite (A4 is deferred): {sorted(unknown)}"
            )
        rows = [row for row in rows if row["cell"] in cells]
    if repeats:
        rows = [row for row in rows if row["repeat"] in repeats]
    if not rows:
        raise ValueError("The requested schedule is empty")
    return rows


def execute_cell(study, directory, entry):
    root = Path(study["root"])
    cell = next(cell for cell in study["cells"] if cell["id"] == entry["cell"])
    dataset = directory / (
        "inputs/pilot10.jsonl" if entry["suite"] == "pilot" else "inputs/study60.jsonl"
    )
    rows = GeoToolEvalDataset(dataset).load()
    random.Random(study["design"]["schedule_seed"] + entry["repeat"]).shuffle(rows)
    destination = directory / "runs" / entry["run_id"]
    predictions = destination / "predictions.jsonl"
    settings = RuntimeSettings.model_validate(study["design"]["runtime"])
    contracts = AnswerContracts(root)
    identity = {
        "agent": "agentgs",
        "study_id": study["study_id"],
        "cell": entry["cell"],
        "repeat": entry["repeat"],
        "planning_mode": "skill_guided",
        "dataset_sha256": file_hash(dataset),
        "question_ids": sorted(row.question_id for row in rows),
        "effective_algorithm": cell["effective_algorithm"],
        "runtime": settings.model_dump(mode="json"),
        "outer_retries": 0,
    }

    @asynccontextmanager
    async def open_runner():
        async with TaskService(root, settings=settings) as service:

            async def execute(row, attempt):
                task = request_for(row, study, cell, contracts)
                await invoke(
                    write_json, attempt / "public_input.json", task.model_dump(mode="json")
                )
                submitted = await service.submit(task)
                await invoke(
                    write_json,
                    attempt / "submission.json",
                    {"task_id": submitted.spec.id, "question_id": row.question_id},
                )
                result = await service.wait(submitted.spec.id)
                await invoke(
                    write_json, attempt / "task_record.json", result.model_dump(mode="json")
                )
                events = await service.events(submitted.spec.id)
                await invoke(
                    (attempt / "events.jsonl").write_text,
                    "".join(event.model_dump_json() + "\n" for event in events),
                    encoding="utf-8",
                )
                outcome = classify_result(result)
                outcome["task_id"] = submitted.spec.id
                if result.result:
                    outcome["provenance"] = result.result.provenance
                return outcome

            yield execute

    code = run_batch(
        rows=rows,
        output=predictions,
        identity=identity,
        retries=0,
        retry_failed=False,
        concurrency=settings.max_concurrent_tasks,
        opener=open_runner,
    )
    load_study(directory)
    if data_inventory(root) != study["data_inventory"]:
        raise ValueError(
            "Prepared data metadata changed during this cell; do not mix this run with the frozen study"
        )
    records = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    if len(records) != len(rows) or {row["question_id"] for row in records} != {
        row.question_id for row in rows
    }:
        raise ValueError(
            "Cell is incomplete. Resume the same cell; missing questions cannot enter paired estimates"
        )
    scores = destination / "scores.json"
    scorer()(predictions, scores, dataset, project=root)
    write_json(
        destination / "receipt.json",
        {
            "study_id": study["study_id"],
            **entry,
            "predictions_sha256": file_hash(predictions),
            "scores_sha256": file_hash(scores),
            "identity_sha256": digest(identity),
            "failed_questions": sum(row["status"] != "succeeded" for row in records),
            "runner_exit_code": code,
        },
    )
    if all(row["status"] != "succeeded" for row in records):
        raise ValueError(
            "Every question failed before a usable answer; inspect the protocol before continuing"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument(
        "--suite", choices=["pilot", "primary", "interaction", "all"], default="pilot"
    )
    parser.add_argument("--cell", action="append")
    parser.add_argument("--repeat", type=int, choices=[0, 1, 2, 3], action="append")
    parser.add_argument(
        "--execute", action="store_true", help="Without this flag, only print the frozen schedule"
    )
    args = parser.parse_args()
    study = load_study(args.study, check_source=args.execute)
    chosen = entries(study, args.suite, args.cell, args.repeat)
    print(
        json.dumps(
            {
                "execute": args.execute,
                "study_id": study["study_id"],
                "source_matches_frozen_study": source_snapshot(Path(study["root"]))
                == study["source_snapshot"],
                "episodes": len(chosen) * (10 if args.suite == "pilot" else 60),
                "schedule": chosen,
            },
            indent=2,
        )
    )
    if not args.execute:
        return
    worker_only()
    if data_inventory(Path(study["root"])) != study["data_inventory"]:
        raise ValueError("Prepared data changed since selection; prepare a fresh study")
    for entry in chosen:
        load_study(args.study)
        execute_cell(study, args.study.resolve(), entry)


if __name__ == "__main__":
    main()
