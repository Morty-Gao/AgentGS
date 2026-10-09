"""Method-specific evaluation views over saved predictions."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
    search_path = list(sys.path)
    common = sys.modules.pop("_common", None)
    try:
        sys.path[:0] = [str(ROOT / "scripts/evaluation"), str(ROOT / "scripts/benchmark")]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = search_path
        sys.modules.pop("_common", None)
        if common is not None:
            sys.modules["_common"] = common


earth = _load("earth_eval", ROOT / "scripts/evaluation/earthagent/score.py")
geo = _load("geo_eval", ROOT / "scripts/evaluation/geoagent/score.py")
agent = _load("agentgs_eval", ROOT / "scripts/evaluation/agentgs/score.py")


def dataset(tmp_path):
    row = {
        "question_id": 1,
        "category_id": 1,
        "question": "Compute the value",
        "expected_answer": "2",
        "gold_chain_text": "DeepEarthQueryLITHO1",
        "chain_mode": "single_tool",
        "answer_type": "single_number",
        "input_resource": "",
        "gold_steps": [{"tool_name": "DeepEarthQueryLITHO1", "arguments": {"site": "andes"}}],
    }
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


def write_predictions(tmp_path, record):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(record) + "\n")
    return path


def receipt(name, call_id, site="andes"):
    return {
        "call_id": call_id,
        "name": name,
        "arguments": {"site": site},
        "bound_arguments": {"site": site},
        "result": {"value": 2},
        "status": "success",
    }


def test_earth_uses_the_final_plain_turn_and_its_observations(tmp_path):
    record = {
        "question_id": 1,
        "answer": "999",
        "conversation": {
            "schema_version": 2,
            "messages": [
                {"role": "user", "content": "Compute the value"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "gold",
                            "type": "function",
                            "function": {
                                "name": "DeepEarthQueryLITHO1",
                                "arguments": '{"site": "andes"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "gold",
                    "content": '{"value": 2}',
                    "metadata": {"status": "success", "tool_input": {"site": "andes"}},
                },
            ],
        },
        "trace": {
            "schema": "earth-agent-react-v2",
            "prompt": "Compute the value",
            "turns": [
                {
                    "assistant": {"content": "FINAL: 0"},
                    "observations": [
                        {
                            "tool_call_id": "other",
                            "name": "OtherTool",
                            "arguments": {"site": "andes"},
                            "bound_arguments": {"site": "andes"},
                            "result": {"value": 0},
                            "status": "success",
                        }
                    ],
                },
                {"assistant": {"content": "2"}, "observations": []},
            ],
        },
    }
    report = earth.evaluate(
        write_predictions(tmp_path, record),
        tmp_path / "scores.json",
        dataset(tmp_path),
        allow_partial=True,
        project=tmp_path,
    )
    scored = report["results"][0]
    assert scored["scores"]["fas"] == 1
    assert scored["scores"]["tsf1"] == 0
    assert scored["evaluation"]["answer_extraction"]["method"] == "plain_final_response"
    assert report["tool_match"] == "react_final_attempt"


def test_geo_scores_only_the_selected_execution(tmp_path):
    record = {
        "question_id": 1,
        "answer": "9",
        "answer_extraction": {"answer": "9", "method": "terminal_stdout"},
        "conversation": {"schema_version": 2, "messages": [{"role": "user", "content": "Compute"}]},
        "trace": {
            "schema": "geoagent-mcts-v2",
            "prompt": "Compute the value",
            "selected_execution_id": "chosen",
            "events": [
                {
                    "phase": "execute",
                    "execution_id": "pruned",
                    "stdout": "FINAL: 9",
                    "calls": [receipt("OtherTool", "a")],
                },
                {
                    "phase": "execute",
                    "execution_id": "chosen",
                    "stdout": "prefix\nFINAL: 2",
                    "calls": [receipt("DeepEarthQueryLITHO1", "b")],
                },
            ],
        },
    }
    report = geo.evaluate(
        write_predictions(tmp_path, record),
        tmp_path / "scores.json",
        dataset(tmp_path),
        allow_partial=True,
        project=tmp_path,
    )
    scored = report["results"][0]["scores"]
    assert scored["fas"] == 1
    assert scored["tsf1"] == 1
    assert scored["pas"] == 1
    assert report["tool_match"] == "selected_execution"


def test_geo_rejects_an_answer_that_is_not_on_the_selected_execution(tmp_path):
    record = {
        "question_id": 1,
        "answer": "9",
        "answer_extraction": {"answer": "9", "method": "terminal_stdout"},
        "trace": {
            "schema": "geoagent-mcts-v2",
            "selected_execution_id": "chosen",
            "events": [
                {"phase": "execute", "execution_id": "pruned", "stdout": "9", "calls": []},
                {
                    "phase": "execute",
                    "execution_id": "chosen",
                    "stdout": "no answer here",
                    "calls": [],
                },
            ],
        },
    }
    prepared_answer, _conversation, note = geo.prepare(record)
    assert prepared_answer == ""
    assert note["answer_extraction"]["method"] == "rejected_unselected_output"


def test_agentgs_keeps_the_stage_answer_and_data_only_is_fas(tmp_path):
    record = {
        "question_id": 1,
        "final_answer": "2",
        "conversation": {
            "schema_version": 2,
            "messages": [
                {"role": "user", "content": "Compute the value\nFINAL: 9"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call1",
                            "type": "function",
                            "function": {
                                "name": "DeepEarthQueryLITHO1",
                                "arguments": '{"site": "andes"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call1",
                    "content": '{"value": 2}',
                    "metadata": {"status": "success", "tool_input": {"site": "andes"}},
                },
            ],
        },
    }
    predictions = write_predictions(tmp_path, record)
    guided = agent.evaluate(
        predictions,
        tmp_path / "guided.json",
        dataset(tmp_path),
        allow_partial=True,
        project=tmp_path,
    )
    assert guided["results"][0]["scores"]["fas"] == 1
    assert guided["results"][0]["scores"]["tsf1"] == 1
    assert guided["primary_metric"] == "final_score"
    assert guided["comparison"]["tsf1"] == 1
    manifest = predictions.parent / "predictions.run"
    manifest.mkdir()
    (manifest / "manifest.json").write_text(json.dumps({"planning_mode": "data_only"}))
    only = agent.evaluate(
        predictions,
        tmp_path / "data-only.json",
        dataset(tmp_path),
        allow_partial=True,
        project=tmp_path,
    )
    assert only["primary_metric"] == "fas"
    assert only["tool_match"] == "not_applicable"
    assert "tsf1" not in only["comparison"]
    assert only["comparison"]["micro_fas"] == 1
