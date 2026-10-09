import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentgs.agent.answering.formats import RESOURCE_MARKER, check_fixed_answer
from agentgs.agent.planning.model_stages import ModelStages
from agentgs.agent.state import RunState
from agentgs.client.models import LLMConfig
from agentgs.client.openai_compatible import OpenAICompatibleClient
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.evaluation.benchmark import public_input
from agentgs.evaluation.datasets.geotooleval import load_tasks
from agentgs.runtime.answer_contracts import AnswerContracts

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data/geotooleval.jsonl"


def test_all_workbook_formats_are_loaded_without_private_fields():
    contracts = AnswerContracts(ROOT)
    rows = [json.loads(line) for line in DATASET.read_text().splitlines()]
    assert len(rows) == 603
    for row in rows:
        contract = contracts.find(row["题目正文"] + RESOURCE_MARKER + "{}")
        assert contract is not None
        assert set(contract) == {"question_id", "question", "question_sha256", "answer_format"}
        assert contract["answer_format"]["template"] == row["固定回答格式"]
        assert contract["answer_format"]["notes"] == row["格式说明"]
        assert row["固定回答格式"] in contract["answer_format"]["llm_instruction"]
        assert row["格式说明"] in contract["answer_format"]["llm_instruction"]


@pytest.mark.parametrize("incomplete", [False, True])
async def test_every_question_format_reaches_the_provider_request(incomplete):
    contracts = AnswerContracts(ROOT)
    tasks = load_tasks(DATASET)
    payloads = []

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "unknown"}}]})

    config = LLMConfig.model_validate({
        "providers": {"fixture": {"base_url": "https://fixture/v1", "api_key": "fixture"}},
        "models": {"fixture": {"provider": "fixture", "model": "fixture",
                               "context_window": 128000, "max_output_tokens": 8192}},
        "roles": {"answer": "fixture", "planner": "fixture"},
    })
    client = OpenAICompatibleClient(config, transport=httpx.MockTransport(handler))
    context = SimpleNamespace(
        tools=SimpleNamespace(specs=lambda: []), client=client, options=AlgorithmSettings(), checkpoint=None,
        events=SimpleNamespace(emit=lambda *args: None),
    )
    try:
        for task in tasks:
            request = public_input(task, output_contract=contracts.find(task.question))
            assert request.output_contract is not None
            state = RunState(context, request)
            stages = ModelStages(state, lambda _: request.output_contract,
                                 contracts.system_prompt)
            await stages.generate_output(task.question, "fixture tool evidence", model=None,
                                         execution_incomplete=incomplete)
            messages = payloads[-1]["messages"]
            assert messages[-1]["role"] == "developer" and messages[-1]["content"].startswith(contracts.system_prompt)
            prompt = messages[-1]["content"]
            spec = request.output_contract["answer_format"]
            assert spec["template"] in prompt
            assert spec["notes"] in prompt
            assert "fixture tool evidence" in prompt
            assert state.usage[0]["stage"] == "final_answer_generation"
        assert len(payloads) == 603
    finally:
        await client.aclose()


def test_missing_format_snapshot_is_explicit(tmp_path):
    dataset = tmp_path / "data/geotooleval.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n")
    with pytest.raises(FileNotFoundError, match="answer contracts"):
        AnswerContracts(tmp_path)


@pytest.mark.parametrize(("qid", "answer", "valid"), [
    (1, "46.54 km", True), (1, "46.5 km", False),
    (151, "country: United States; count: 5\ncountry: Canada; count: 4", True),
    (151, "United States: 5\nCanada: 4", False),
    (199, "temperature_anomaly: 3.41 °C", True), (199, "3.41 °C", False),
    (451, "Great-circle distance: 16504.900 km", True), (451, "16504.900 km", False),
])
def test_labels_repeated_records_and_precision_are_checked(qid, answer, valid):
    task = next(t for t in load_tasks(DATASET) if t.question_id == qid)
    contract = AnswerContracts(ROOT).find(task.question)
    assert check_fixed_answer(answer, contract)["valid"] is valid
