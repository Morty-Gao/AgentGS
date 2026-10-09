from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATEGORY_NAMES = {
    1: "DeepEarthTools",
    2: "GeoBigDataTools",
    3: "MineralResourcesTools",
    4: "ClimateChangeTools",
    5: "EnergyResourcesTools",
    6: "OceanCarbonTools",
    7: "DeepTimeBiosphereTools",
    8: "CCSTools",
    9: "RemoteSensingTools",
    10: "NaturalHazardTools",
}


@dataclass(frozen=True)
class EvalTask:
    question_id: int
    category_id: int
    category: str
    question: str
    expected_answer: str
    answer_type: str
    input_resource: Any
    chain_mode: str
    gold_chain_text: str
    gold_steps: tuple[dict[str, Any], ...]


def load_tasks(dataset_path: Path) -> list[EvalTask]:
    """Read one workbook row per JSONL record, preserving original column names."""
    required = {
        "题目序号",
        "所属类别",
        "题目正文",
        "题目答案",
        "标准工具链调用",
        "单/多工具链调用",
        "答案类型",
        "输入资源",
        "标准工具调用明细",
    }
    tasks: list[EvalTask] = []
    seen = set()

    def text(value):
        return "" if value is None else str(value)

    with dataset_path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"GeoToolEval line {line_number} has invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"GeoToolEval line {line_number} must be an object")
            missing = required - row.keys()
            if missing:
                raise ValueError(
                    f"GeoToolEval line {line_number} is missing columns: {sorted(missing)}"
                )
            question_id = int(row["题目序号"])
            category_id = int(row["所属类别"])
            if question_id in seen:
                raise ValueError(f"Duplicate GeoToolEval question id: {question_id}")
            seen.add(question_id)
            raw_steps = row["标准工具调用明细"]
            try:
                gold_steps = json.loads(raw_steps) if isinstance(raw_steps, str) else raw_steps
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Question {question_id} has invalid standard tool-call JSON: {exc}"
                ) from exc
            if not isinstance(gold_steps, list) or not gold_steps:
                raise ValueError(f"Question {question_id} has no standard tool calls")
            if not all(isinstance(step, dict) for step in gold_steps):
                raise ValueError(f"Question {question_id} has invalid standard tool-call objects")
            tasks.append(
                EvalTask(
                    question_id=question_id,
                    category_id=category_id,
                    category=CATEGORY_NAMES.get(category_id, f"Category{category_id}"),
                    question=text(row["题目正文"]),
                    expected_answer=text(row["题目答案"]),
                    answer_type=text(row["答案类型"]),
                    input_resource=row["输入资源"],
                    chain_mode=text(row["单/多工具链调用"]),
                    gold_chain_text=text(row["标准工具链调用"]),
                    gold_steps=tuple(gold_steps),
                )
            )
    if not tasks:
        raise ValueError("GeoToolEval dataset is empty")
    return tasks


def build_agent_question(task: EvalTask) -> str:
    if not task.input_resource:
        return task.question
    resource = task.input_resource
    if not isinstance(resource, str):
        resource = json.dumps(resource, ensure_ascii=False)
    return f"{task.question}\n\nEvaluation input resource (authoritative benchmark locator; use it exactly when relevant, but retrieve and select the registered tools yourself): {resource}"
