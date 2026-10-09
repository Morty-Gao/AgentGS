import asyncio
import json
from types import SimpleNamespace

import pytest

from agentgs.benchmark.cases import EvaluationCase, evaluate
from agentgs.contracts.models import (
    AgentResult,
    TaskInput,
    TaskRecord,
    TaskSpec,
    TaskStatus,
    ToolCall,
)
from agentgs.messages import Conversation, Message
from agentgs.runtime.service import TaskService


async def test_gold_is_kept_outside_runtime(project):
    case = EvaluationCase(
        id="case",
        input=TaskInput(
            calls=[
                ToolCall(
                    tool="DeepEarthQueryUSGSCatalog",
                    arguments={"year": 2011, "operation": "largest"},
                )
            ]
        ),
        expected_answer="PRIVATE_GOLD",
        expected_tools=["DeepEarthQueryUSGSCatalog"],
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


async def test_runner_bounds_async_submissions_and_preserves_history(tmp_path):
    class Service:
        settings = SimpleNamespace(max_concurrent_tasks=2)
        running = peak = 0

        async def submit(self, request):
            self.running += 1
            self.peak = max(self.peak, self.running)
            assert self.running <= 2
            return TaskRecord(spec=TaskSpec(input=request))

        async def wait(self, task_id):
            await asyncio.sleep(0.01)
            self.running -= 1
            return TaskRecord(
                spec=TaskSpec(id=task_id, input=TaskInput()),
                status=TaskStatus.SUCCEEDED,
                result=AgentResult(
                    answer="new",
                    messages=history.messages
                    + [
                        Message(role="assistant", content="new", reasoning="retained"),
                    ],
                ),
            )

    history = Conversation.from_openai([{"role": "user", "content": "old"}])
    call_id = history.begin_tool("OldTool", "{}")
    history.finish_tool(call_id, "old")
    history.append(Message(role="user", content="new"))
    cases = [
        EvaluationCase(
            id=str(i),
            input=TaskInput(messages=history.messages),
            expected_tools=[],
        )
        for i in range(5)
    ]
    service = Service()
    run = await evaluate(service, cases, tmp_path)
    rows = json.loads((run / "summary.json").read_text())["results"]
    assert service.peak == 2
    assert all(row["actual_tools"] == [] and row["tool_chain_match"] for row in rows)
    restored = Conversation.load(run / f"{cases[0].id.encode().hex()}.messages.json")
    assert restored.messages[-1].reasoning == "retained"
    assert restored.messages[: len(history.messages)] == history.messages


async def test_duplicate_cases_are_rejected_before_submission(tmp_path):
    case = EvaluationCase(id="duplicate", input=TaskInput())
    with pytest.raises(ValueError, match="Duplicate evaluation case ids"):
        await evaluate(None, [case, case], tmp_path)
