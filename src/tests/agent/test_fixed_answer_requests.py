import json
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentgs.agent.answering.formats import RESOURCE_MARKER, check_fixed_answer
from agentgs.agent.planning.model_stages import ModelStages
from agentgs.agent.state import RunState
from agentgs.benchmark.inputs import public_input
from agentgs.client.models import LLMConfig
from agentgs.client.openai_compatible import OpenAICompatibleClient
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.datasets.geotooleval import ANSWER_SYSTEM_PROMPT
from agentgs.evaluation.datasets.geotooleval import load_tasks
from agentgs.runtime.answer_contracts import AnswerContracts

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data/geotooleval.jsonl"


def test_all_workbook_formats_are_loaded_without_private_fields():
    contracts = AnswerContracts(ROOT)
    rows = [json.loads(line) for line in DATASET.read_text().splitlines()]
    assert len(rows) == 603
    for row in rows:
        contract = contracts.find(row["question"] + RESOURCE_MARKER + "{}")
        assert contract is not None
        assert set(contract) == {"question_id", "question", "question_sha256", "answer_format"}
        assert contract["answer_format"]["template"] == row["answer_template"]
        assert contract["answer_format"]["notes"] == row["format_notes"]
        assert row["answer_template"] in contract["answer_format"]["llm_instruction"]
        assert row["format_notes"] in contract["answer_format"]["llm_instruction"]
        assert contract["answer_format"] == row["answer_format"]
        assert contract["question_sha256"] == row["question_sha256"]
    assert contracts.system_prompt == ANSWER_SYSTEM_PROMPT


def test_embedded_format_settings_preserve_precision_options_and_repeated_records():
    by_id = {
        row["question_id"]: row["answer_format"] for row in AnswerContracts(ROOT).contracts.values()
    }
    assert by_id[1]["decimals"] == 2
    assert by_id[287]["decimals"] == 3
    assert by_id[137]["repeat_lines"] == [0]
    assert by_id[137]["max_records"] == 5
    assert "repeat_lines" not in by_id[465]
    assert by_id[495]["allowed_options"] == ["A", "B", "C", "D"]


@pytest.mark.parametrize("incomplete", [False, True])
async def test_every_question_format_reaches_the_provider_request(incomplete):
    contracts = AnswerContracts(ROOT)
    tasks = load_tasks(DATASET)
    payloads = []

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "unknown"}}]})

    config = LLMConfig.model_validate(
        {
            "providers": {"fixture": {"base_url": "https://fixture/v1", "api_key": "fixture"}},
            "models": {
                "fixture": {
                    "provider": "fixture",
                    "model": "fixture",
                    "context_window": 128000,
                    "max_output_tokens": 8192,
                }
            },
        }
    )
    client = OpenAICompatibleClient(config, transport=httpx.MockTransport(handler))
    context = SimpleNamespace(
        tools=SimpleNamespace(specs=lambda: []),
        client=client,
        options=AlgorithmSettings(),
        checkpoint=None,
        events=SimpleNamespace(emit=lambda *args: None),
    )
    try:
        for task in tasks:
            request = public_input(task, output_contract=contracts.find(task.question))
            assert request.output_contract is not None
            state = RunState(context, request)
            stages = ModelStages(state, lambda _: request.output_contract)
            await stages.generate_output(
                task.question, "fixture tool evidence", execution_incomplete=incomplete
            )
            messages = payloads[-1]["messages"]
            assert messages[0]["role"] == "system"
            assert contracts.system_prompt in unescape(messages[0]["content"])
            prompt = unescape(messages[-1]["content"])
            spec = request.output_contract["answer_format"]
            context_text = prompt.split("<task-context>\n", 1)[1].split("\n</task-context>", 1)[0].split("\n\n", 1)[1]
            assert json.loads(context_text)["output_contract"]["answer_format"] == spec
            assert "fixture tool evidence" in prompt
            assert state.usage[0]["stage"] == "final_answer_generation"
        assert len(payloads) == 603
    finally:
        await client.aclose()


def test_formats_load_from_only_the_unified_dataset(tmp_path):
    dataset = tmp_path / "data/geotooleval.jsonl"
    dataset.parent.mkdir(parents=True)
    row = json.loads(DATASET.read_text().splitlines()[0])
    dataset.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    contracts = AnswerContracts(tmp_path)
    public = contracts.find(row["question"])
    assert public["answer_format"] == row["answer_format"]
    assert not (tmp_path / "data/answer_formats").exists()
    request = public_input(load_tasks(dataset)[0], output_contract=public)
    assert set(request.output_contract) == {
        "question_id",
        "question",
        "question_sha256",
        "answer_format",
    }
    assert "expected_answer" not in request.model_dump_json()
    assert "gold_steps" not in request.model_dump_json()


def test_missing_embedded_formats_are_explicit(tmp_path):
    dataset = tmp_path / "data/geotooleval.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n")
    with pytest.raises(ValueError, match="missing answer-format columns"):
        AnswerContracts(tmp_path)


@pytest.mark.parametrize("field", ["question_sha256", "template", "notes"])
def test_inconsistent_embedded_formats_are_rejected(tmp_path, field):
    dataset = tmp_path / "data/geotooleval.jsonl"
    dataset.parent.mkdir(parents=True)
    row = json.loads(DATASET.read_text().splitlines()[0])
    if field == "question_sha256":
        row[field] = "0" * 64
    else:
        row["answer_format"][field] = "changed"
    dataset.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="hash mismatch|inconsistent answer formats"):
        AnswerContracts(tmp_path)


def test_selected_unified_dataset_is_used_without_separate_format_files(tmp_path):
    data = tmp_path / "data"
    metadata = data / "metadata"
    metadata.mkdir(parents=True)
    dataset = data / "selected.jsonl"
    row = json.loads(DATASET.read_text().splitlines()[0])
    dataset.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    (metadata / "geotooleval.json").write_text(
        json.dumps({"version_manifest": {"selected_dataset": "data/selected.jsonl"}})
    )
    assert AnswerContracts(tmp_path).find(row["question"])["answer_format"] == row["answer_format"]
    dataset.unlink()
    with pytest.raises(FileNotFoundError, match="Selected GeoToolEval dataset"):
        AnswerContracts(tmp_path)


@pytest.mark.parametrize(
    ("qid", "answer", "valid"),
    [
        (1, "46.54 km", True),
        (1, "46.5 km", False),
        (151, "country: United States; count: 5\ncountry: Canada; count: 4", True),
        (151, "United States: 5\nCanada: 4", False),
        (199, "temperature_anomaly: 3.41 °C", True),
        (199, "3.41 °C", False),
        (451, "Great-circle distance: 16504.900 km", True),
        (451, "16504.900 km", False),
    ],
)
def test_labels_repeated_records_and_precision_are_checked(qid, answer, valid):
    task = next(t for t in load_tasks(DATASET) if t.question_id == qid)
    contract = AnswerContracts(ROOT).find(task.question)
    assert check_fixed_answer(answer, contract)["valid"] is valid
