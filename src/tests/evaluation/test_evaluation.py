import json

from agentgs.contracts.models import TaskInput, ToolCall
from agentgs.evaluation.models import EvaluationCase
from agentgs.evaluation.runner import evaluate
from agentgs.runtime.service import TaskService


async def test_gold_is_kept_outside_runtime(project):
    case = EvaluationCase(
        id="case",
        input=TaskInput(calls=[ToolCall(tool="GeoMagnitudeToEnergy", arguments={"magnitude": 7})]),
        expected_answer="PRIVATE_GOLD",
        expected_tools=["GeoMagnitudeToEnergy"],
    )
    service = TaskService(project)
    try:
        run = await evaluate(service, [case], project / "outputs" / "evaluations")
        row = json.loads((run / "summary.json").read_text())["results"][0]
        request = (project / "outputs" / "tasks" / row["task_id"] / "request.json").read_text()
        assert "PRIVATE_GOLD" not in request
        assert row["tool_chain_match"] is True and row["answer_exact_match"] is False
    finally:
        await service.close()
