import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]


def test_frozen_fas_all_answer_types_match_original_source():
    if (
        importlib.util.find_spec("openpyxl") is None
        or not (ROOT / "data/geotooleval/GeoToolEval.xlsx").exists()
    ):
        pytest.skip("Frozen benchmark workbook and reproduction profile are required")
    from agentgs.evaluation.datasets.geotooleval import load_tasks
    from agentgs.evaluation.metrics.answers import score_answer

    cases = json.loads((ROOT / "src/tests/fixtures/evaluation/fas.json").read_text("utf-8"))
    tasks = {
        task.question_id: task for task in load_tasks(ROOT / "data/geotooleval/GeoToolEval.xlsx")
    }
    assert len(tasks) == 603
    for case in cases:
        # The oracle is JSON; numeric detail pairs are tuples in the Python scorer.
        actual = score_answer(tasks[case["question_id"]], case["answer"])
        assert json.loads(json.dumps(actual)) == case["expected"]


def test_public_task_never_contains_private_reference_data():
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("Benchmark adapter optional dependency")
    from agentgs.evaluation.benchmark import public_input
    from agentgs.evaluation.datasets.geotooleval import EvalTask

    task = EvalTask(
        1,
        1,
        "DeepEarthTools",
        "A public question",
        "PRIVATE_ANSWER",
        "单项文本结果",
        '{"source_id":1,"gold_answer":"PRIVATE_VALUE"}',
        "single",
        "PRIVATE_CHAIN",
        ({"tool_name": "PRIVATE_TOOL"},),
    )
    request = public_input(task).model_dump_json()
    assert "PRIVATE_" not in request and "gold_answer" not in request


def test_missing_usage_remains_unknown_for_original_cost_metrics():
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("Benchmark adapter optional dependency")
    from agentgs.evaluation.metrics.tools import score_task

    result = score_task(
        gold_steps=[{"tool_name": "T", "arguments": {"value": 1}}],
        trace=[
            {
                "type": "tool_call",
                "tool_name": "T",
                "status": "success",
                "tool_input": '{"value":1}',
            }
        ],
        answer_score=1,
        token_usage={"usage_complete": False, "total_tokens": 0},
    )
    assert result["cost_score"] is None and result["final_score"] is None


def test_inflight_or_retried_requests_cannot_be_reported_as_complete_token_usage():
    from agentgs.evaluation.benchmark import usage_summary

    partial = usage_summary(
        [{"usage": {"total_tokens": 10}, "attempts": 1}, {"usage": None, "attempts": None}]
    )
    assert partial["usage_complete"] is False and partial["total_tokens"] is None
    assert partial["reported_tokens"] == 10
    retried = usage_summary([{"usage": {"total_tokens": 10}, "attempts": 2}])
    assert retried["usage_complete"] is False and retried["total_tokens"] is None
