"""Builtin GeoToolEval dataset. Other benchmarks subclass ``Dataset`` the same way."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agentgs.evaluation.datasets.base import Dataset, Example

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

_REQUIRED_COLUMNS = {
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


@dataclass(frozen=True)
class EvalTask(Example):
    """One GeoToolEval row, including the workbook fields the scorer does not read."""

    chain_mode: str = ""
    gold_chain_text: str = ""


class GeoToolEvalDataset(Dataset):
    """JSONL exported from the GeoToolEval workbook. One object per line."""

    name = "geotooleval"
    domain_count = 10

    def load(self) -> list[EvalTask]:
        tasks: list[EvalTask] = []
        seen: set[int] = set()
        with self.source.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                tasks.append(_parse_row(line_number, line, seen))
        if not tasks:
            raise ValueError("GeoToolEval dataset is empty")
        return tasks

    def require(self, question_ids: Iterable[int]) -> dict[int, EvalTask]:
        selected = {int(value) for value in question_ids}
        found = {task.question_id: task for task in self.load()}
        missing = selected - found.keys()
        if missing:
            raise KeyError(f"Question IDs not found in GeoToolEval: {sorted(missing)}")
        return {question_id: found[question_id] for question_id in selected}


def load_tasks(dataset_path: Path) -> list[EvalTask]:
    return GeoToolEvalDataset(dataset_path).load()


def build_agent_question(task: Example) -> str:
    if not task.input_resource:
        return task.question
    resource = task.input_resource
    if not isinstance(resource, str):
        resource = json.dumps(resource, ensure_ascii=False)
    return (
        f"{task.question}\n\nEvaluation input resource "
        "(authoritative benchmark locator; use it exactly when relevant, "
        f"but retrieve and select the registered tools yourself): {resource}"
    )


def _text(value):
    return "" if value is None else str(value)


def _parse_row(line_number: int, line: str, seen: set[int]) -> EvalTask:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GeoToolEval line {line_number} has invalid JSON: {exc}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"GeoToolEval line {line_number} must be an object")
    missing = _REQUIRED_COLUMNS - row.keys()
    if missing:
        raise ValueError(f"GeoToolEval line {line_number} is missing columns: {sorted(missing)}")
    question_id = int(row["题目序号"])
    category_id = int(row["所属类别"])
    if question_id in seen:
        raise ValueError(f"Duplicate GeoToolEval question id: {question_id}")
    seen.add(question_id)
    gold_steps = _gold_steps(question_id, row["标准工具调用明细"])
    return EvalTask(
        question_id=question_id,
        category_id=category_id,
        category=CATEGORY_NAMES.get(category_id, f"Category{category_id}"),
        question=_text(row["题目正文"]),
        expected_answer=_text(row["题目答案"]),
        answer_type=_text(row["答案类型"]),
        input_resource=row["输入资源"],
        chain_mode=_text(row["单/多工具链调用"]),
        gold_chain_text=_text(row["标准工具链调用"]),
        gold_steps=tuple(gold_steps),
    )


def _gold_steps(question_id: int, raw_steps: Any) -> list[dict[str, Any]]:
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
    return gold_steps
