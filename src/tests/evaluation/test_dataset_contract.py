"""A dataset is a Dataset subclass. Metrics score its examples without running the agent."""

import pytest

from agentgs.benchmark.inputs import public_input
from agentgs.evaluation.datasets.base import Dataset, Example, get_dataset
from agentgs.evaluation.metrics.report import score_prediction
from agentgs.messages import Conversation


class SampleDataset(Dataset):
    name = "sample-benchmark"
    domain_count = None

    def load(self):
        return [
            Example(
                question_id="case-a",
                category_id="demo",
                category="Demo",
                question="What is 2?",
                expected_answer="2",
                answer_type="单数值结果",
                input_resource={"source_id": 1, "gold_answer": "PRIVATE"},
                gold_steps=({"tool_name": "T", "arguments": {"x": 1}},),
            )
        ]


def test_custom_dataset_registers_and_scores_without_leaking_gold(tmp_path):
    source = tmp_path / "sample.jsonl"
    source.write_text("{}\n")
    dataset = get_dataset("sample-benchmark", source)
    assert isinstance(dataset, SampleDataset)
    example = dataset.load()[0]
    conversation = Conversation()
    call_id = conversation.begin_tool("T", '{"x":1}')
    conversation.finish_tool(call_id, "2", status="success")
    scored = score_prediction(example, "2", conversation)
    assert scored["question_id"] == "case-a"
    assert scored["scores"]["fas"] == scored["scores"]["pas"] == 1
    request = public_input(example).model_dump_json()
    assert "PRIVATE" not in request and "gold_answer" not in request
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset("missing-benchmark", source)


def test_dataset_names_are_unique():
    class First(Dataset):
        name = "duplicate-benchmark-name"

        def load(self):
            return []

    with pytest.raises(TypeError, match="already used"):

        class Second(Dataset):
            name = "duplicate-benchmark-name"

            def load(self):
                return []

    assert Dataset.registry["duplicate-benchmark-name"] is First
