import asyncio
import json
from types import SimpleNamespace

import pytest

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import AgentResult, TaskRecord, TaskSpec, TaskStatus, ToolError
from agentgs.evaluation.benchmark import run_benchmark
from agentgs.evaluation.datasets.geotooleval import EvalTask
from agentgs.messages import Conversation, Message


class Service:
    def __init__(self, root, *, auth_failure=False):
        self.root = root
        self.settings = RuntimeSettings(max_concurrent_tasks=2)
        self.processes = {}
        self.records = {}
        self.peak = 0
        self.auth_failure = auth_failure

    def submit(self, request):
        record = TaskRecord(spec=TaskSpec(input=request))
        self.records[record.spec.id] = record
        return record

    async def wait(self, task_id):
        self.processes[task_id] = object()
        self.peak = max(self.peak, len(self.processes))
        await asyncio.sleep(0.02)
        self.processes.pop(task_id)
        record = self.records[task_id]
        if self.auth_failure:
            record.status = TaskStatus.FAILED
            record.error = ToolError(code="execution_failed", message="HTTP 401")
        else:
            record.status = TaskStatus.SUCCEEDED
            record.result = AgentResult(
                answer="123",
                messages=[Message(role="assistant", content="123", reasoning="fixture reasoning")],
            )
        return record

    def events(self, task_id):
        return []


@pytest.fixture
def benchmark_project(project, monkeypatch):
    from agentgs.evaluation import benchmark

    tasks = [
        EvalTask(
            question_id=i,
            category_id=i,
            category=f"Category{i}",
            question=f"Question {i}",
            expected_answer="123",
            answer_type="单数值结果",
            input_resource="",
            chain_mode="单工具调用",
            gold_chain_text="Tool",
            gold_steps=({"tool": "Tool", "params": {}},),
        )
        for i in range(1, 6)
    ]
    monkeypatch.setattr(benchmark, "load_tasks", lambda _: tasks)
    monkeypatch.setattr(
        "agentgs.client.config.load_llm",
        lambda _: SimpleNamespace(models={}, providers={}, roles={}),
    )
    dataset = project / "dataset.jsonl"
    dataset.write_text("frozen dataset")
    return project, dataset


async def test_parallel_width_durable_logs_and_resume(benchmark_project):
    root, dataset = benchmark_project
    output = root / "experiment"
    service = Service(root)
    report = await run_benchmark(service, dataset, output)
    assert service.peak == 2
    assert len(service.records) == report["submitted"] == 5
    assert not (output / ".run.lock").exists()
    events = [
        json.loads(line) for line in (output / "experiment.events.jsonl").read_text().splitlines()
    ]
    assert sum(e["type"] == "benchmark.task_finished" for e in events) == 5
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 5
    for folder in (output / "tasks").iterdir():
        attempt = folder / "attempt_001"
        assert (attempt / "events.jsonl").exists()
        assert (attempt / "task_record.json").exists()
        assert (attempt / "submission.json").exists()
        saved = json.loads((folder / "result.json").read_text())
        assert "trace" not in saved
        assert saved["messages"][0]["reasoning"] == "fixture reasoning"
        conversation = Conversation.from_jsonl((attempt / "messages.jsonl").read_text())
        assert conversation.messages[0].reasoning == "fixture reasoning"
        assert "expected_answer" not in (attempt / "public_input.json").read_text()
    resumed = Service(root)
    await run_benchmark(resumed, dataset, output)
    assert not resumed.records
    changed = Service(root)
    changed.settings.max_concurrent_tasks = 3
    with pytest.raises(ValueError, match="protocol differ"):
        await run_benchmark(changed, dataset, output)


async def test_auth_failure_stops_new_tasks_and_preserves_retry(benchmark_project):
    root, dataset = benchmark_project
    output = root / "auth_experiment"
    failing = Service(root, auth_failure=True)
    first = await run_benchmark(failing, dataset, output)
    assert len(failing.records) == 2
    assert first["execution"]["stopped_for_auth"]
    successful = Service(root)
    second = await run_benchmark(successful, dataset, output, retry_failed=True)
    assert len(successful.records) == second["submitted"] == 5
    assert (output / "tasks/0001/attempt_001/score.json").exists()
    assert (output / "tasks/0001/attempt_002/score.json").exists()
