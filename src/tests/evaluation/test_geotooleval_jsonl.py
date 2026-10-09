import json

import pytest

from agentgs.evaluation.datasets.geotooleval import build_agent_question, load_tasks
from agentgs.evaluation.datasets.version import benchmark_dataset
from agentgs.evaluation.metrics.tools import load_gold_tasks


def record():
    return {
        "题目序号": 1,
        "所属类别": 1,
        "题目正文": "原题\n第二行，保留空格  ",
        "题目答案": 0,
        "标准工具链调用": "ToolA",
        "单/多工具链调用": "单工具链调用",
        "答案类型": "单项数值结果",
        "输入资源": '{"source_id":1}',
        "标准工具调用明细": '[{"tool_name":"ToolA","arguments":{"value":0}}]',
        "额外字段": "扩展列仍保存在 JSONL 中",
    }


def write_records(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def test_jsonl_preserves_text_zero_answer_and_structured_gold(tmp_path):
    path = tmp_path / "GeoToolEval.jsonl"
    source = record()
    write_records(path, [source])
    task = load_tasks(path)[0]
    assert task.question == source["题目正文"]
    assert task.expected_answer == "0"
    assert task.gold_steps == ({"tool_name": "ToolA", "arguments": {"value": 0}},)
    assert source["输入资源"] in build_agent_question(task)
    assert json.loads(path.read_text())["额外字段"] == source["额外字段"]
    assert load_gold_tasks(path, [1])[1]["expected_answer"] == "0"
    with pytest.raises(KeyError, match="not found"):
        load_gold_tasks(path, [2])


def test_jsonl_accepts_native_call_arrays_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "GeoToolEval.jsonl"
    source = record()
    source["标准工具调用明细"] = json.loads(source["标准工具调用明细"])
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
