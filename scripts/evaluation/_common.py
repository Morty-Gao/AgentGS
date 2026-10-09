"""Score saved predictions. This does not call a model or run a tool."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from agentgs.benchmark.score import score_predictions
from agentgs.messages import Conversation


def load_predictions(path: Path) -> list[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Each prediction must be an object")
            records.append(record)
    return records


def planning_mode(predictions: Path) -> str | None:
    """Read the run manifest beside ``predictions.jsonl``, when one was written."""
    manifest = predictions.parent / (predictions.stem + ".run") / "manifest.json"
    if not manifest.is_file():
        return None
    value = json.loads(manifest.read_text(encoding="utf-8")).get("planning_mode")
    return value if isinstance(value, str) else None


def _dump_conversation(value):
    if isinstance(value, Conversation):
        return value.model_dump(mode="json")
    return value


def score_records(
    dataset,
    records: list[dict],
    output: Path,
    *,
    prepare,
    finish,
    allow_partial: bool = False,
    project: Path | None = None,
) -> dict:
    """Apply one method's view, then the shared answer and conversation metrics."""
    normalized = []
    notes = []
    for record in records:
        answer, conversation, note = prepare(record)
        normalized.append(
            {
                "question_id": record.get("question_id", record.get("case_id")),
                "final_answer": answer,
                "conversation": _dump_conversation(conversation),
            }
        )
        notes.append(note)
    descriptor, name = tempfile.mkstemp(prefix="agentgs-eval-", suffix=".jsonl")
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in normalized),
            encoding="utf-8",
        )
        report = score_predictions(
            dataset,
            temporary,
            output,
            allow_partial=allow_partial,
            project=project,
        )
    finally:
        temporary.unlink(missing_ok=True)
    for row, note in zip(report["results"], notes, strict=True):
        row["evaluation"] = note
    finish(report)
    output = Path(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
