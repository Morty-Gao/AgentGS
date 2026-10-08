"""Re-score saved experiment answers against their independent frozen reports."""

import argparse
import json
from pathlib import Path

from agentgs.evaluation.benchmark import write_json
from agentgs.evaluation.datasets.geotooleval import load_tasks
from agentgs.evaluation.datasets.version import benchmark_workbook
from agentgs.evaluation.metrics.compatibility import score_compatible


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=root / "outputs/verification/reference_scores.json"
    )
    args = parser.parse_args()
    tasks = {t.question_id: t for t in load_tasks(benchmark_workbook(root))}
    formats = {
        row["question_id"]: row["answer_format"]
        for row in (
            json.loads(line)
            for line in (root / "data/answer_formats/archive/v1_1.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    saved = json.loads((root / "data/geotooleval/release_v1_1/rescore_and_replay.json").read_text())
    checks = []
    for record in saved:
        for key, expected in record.items():
            if not key.endswith("_scoring"):
                continue
            answer_key = key.removesuffix("_scoring") + "_answer"
            actual = score_compatible(
                tasks[record["question_id"]], record[answer_key], formats[record["question_id"]]
            )
            checks.append(
                {
                    "question_id": record["question_id"],
                    "answer_version": answer_key,
                    "passed": json.loads(json.dumps(actual)) == expected,
                }
            )
    report = {
        "protocol": "saved_experiment_original_and_compatibility_replay",
        "model_calls": 0,
        "checks": checks,
        "passed": all(row["passed"] for row in checks),
    }
    write_json(args.output, report)
    print(
        json.dumps({"cases": len(checks), "passed": report["passed"], "report": str(args.output)})
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
