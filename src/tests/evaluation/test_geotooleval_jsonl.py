import json

import pytest

from agentgs.evaluation.datasets.geotooleval import (
    GeoToolEvalDataset,
    build_agent_question,
    load_tasks,
)
from agentgs.evaluation.datasets.version import benchmark_dataset, compatibility_contracts


def record():
    return {
        "question_id": 1,
        "category_id": 1,
        "question": "Original question\nsecond line, keep the spaces  ",
        "expected_answer": 0,
        "gold_chain_text": "ToolA",
        "chain_mode": "single_tool",
        "answer_type": "single_number",
        "input_resource": '{"source_id":1}',
        "gold_steps": '[{"tool_name":"ToolA","arguments":{"value":0}}]',
        "extra_field": "extra column remains in the JSONL",
    }


def write_records(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def test_jsonl_preserves_text_zero_answer_and_structured_gold(tmp_path):
    path = tmp_path / "GeoToolEval.jsonl"
    source = record()
    write_records(path, [source])
    task = load_tasks(path)[0]
    assert task.question == source["question"]
    assert task.expected_answer == "0"
    assert task.gold_steps == ({"tool_name": "ToolA", "arguments": {"value": 0}},)
    assert source["input_resource"] in build_agent_question(task)
    assert json.loads(path.read_text())["extra_field"] == source["extra_field"]
    assert GeoToolEvalDataset(path).require([1])[1].expected_answer == "0"
    with pytest.raises(KeyError, match="not found"):
        GeoToolEvalDataset(path).require([2])


def test_jsonl_accepts_native_call_arrays_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "GeoToolEval.jsonl"
    source = record()
    source["gold_steps"] = json.loads(source["gold_steps"])
    write_records(path, [source])
    assert len(load_tasks(path)[0].gold_steps) == 1
    write_records(path, [source, source])
    with pytest.raises(ValueError, match="Duplicate"):
        load_tasks(path)


@pytest.mark.parametrize(
    "content, message",
    [("", "empty"), ("{\n", "invalid JSON"), ("[]\n", "object"), ("{}\n", "missing columns")],
)
def test_invalid_jsonl_is_rejected(tmp_path, content, message):
    path = tmp_path / "GeoToolEval.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_tasks(path)


def test_benchmark_selects_embedded_jsonl_instead_of_old_workbook(tmp_path):
    assert benchmark_dataset(tmp_path) == tmp_path / "data/geotooleval.jsonl"
    folder = tmp_path / "data/metadata"
    folder.mkdir(parents=True)
    (folder / "geotooleval.json").write_text(
        json.dumps({"version_manifest": {"selected_workbook": "old.xlsx"}})
    )
    assert benchmark_dataset(tmp_path) == tmp_path / "data/geotooleval.jsonl"


def test_compatibility_contracts_use_prepared_metadata_instead_of_dataset_depth(project):
    metadata = project / "data/metadata"
    metadata.mkdir()
    path = metadata / "compatibility.jsonl"
    path.write_text("{}")
    (metadata / "geotooleval.json").write_text(
        json.dumps(
            {
                "version_manifest": {
                    "compatibility_contracts": "data/metadata/compatibility.jsonl"
                },
            }
        )
    )
    assert compatibility_contracts(project / "data/geotooleval.jsonl") == path
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Selected compatibility contracts"):
        compatibility_contracts(project / "data/geotooleval.jsonl")
