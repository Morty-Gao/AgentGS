import asyncio
import json
from types import SimpleNamespace

import pytest

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import AgentResult, TaskRecord, TaskSpec, TaskStatus, ToolError
from agentgs.evaluation.benchmark import run_benchmark, score_predictions
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

    async def submit(self, request):
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

    async def events(self, task_id):
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
            gold_steps=({"tool_name": "Tool", "arguments": {}},),
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
        assert Conversation.load(attempt / "conversation.json") == conversation
        assert conversation.messages[0].reasoning == "fixture reasoning"
        assert "expected_answer" not in (attempt / "public_input.json").read_text()
    resumed = Service(root)
    await run_benchmark(resumed, dataset, output)
    assert not resumed.records
    changed = Service(root)
    changed.settings.max_concurrent_tasks = 3
    with pytest.raises(ValueError, match="protocol differ"):
        await run_benchmark(changed, dataset, output)


async def test_tool_ablation_protocol_records_effective_overrides(benchmark_project):
    root, dataset = benchmark_project
    (root / "config.toml").write_text(
        '[agent]\nplanning_mode="skill_guided"\nallow_rule_planning=true\n'
        "plan_attempts=7\ntool_attempts=5\nshared_bridge=true\n"
    )
    service = Service(root)
    output = root / "tool_ablation"
    await run_benchmark(
        service,
        dataset,
        output,
        planning_mode="tool_only",
        allow_rule_planning=False,
        plan_attempts=2,
        tool_attempts=1,
    )
    protocol = json.loads((output / "protocol.json").read_text())
    effective = protocol["effective_algorithm"]
    assert effective["planning_mode"] == "tool_only"
    assert effective["allow_rule_planning"] is False and effective["shared_bridge"] is False
    assert effective["plan_attempts"] == 2 and effective["tool_attempts"] == 1
    assert set(protocol["planning_resources_sha256"]) == {"tool_catalog", "tool_scope"}
    assert "require_llm" not in protocol and "require_llm" not in effective
    for record in service.records.values():
        request = record.spec.input
        assert request.planning_mode == "tool_only" and request.allow_rule_planning is False
        assert request.plan_attempts == 2 and request.tool_attempts == 1
    with pytest.raises(ValueError, match="protocol differ"):
        await run_benchmark(
            Service(root),
            dataset,
            output,
            planning_mode="tool_only",
            plan_attempts=3,
            tool_attempts=1,
        )


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


async def test_saved_native_experiment_has_identical_offline_scores(benchmark_project, monkeypatch):
    root, dataset = benchmark_project
    original_wait = Service.wait

    async def wait(service, task_id):
        record = await original_wait(service, task_id)
        conversation = Conversation()
        conversation.record_event("model.started", {"request_id": task_id})
        call_id = conversation.begin_tool("Tool", "{}")
        conversation.finish_tool(call_id, "123", status="success")
        conversation.append(Message(role="assistant", content="123", reasoning="native reasoning"))
        conversation.record_event(
            "model.completed",
            {
                "request_id": task_id,
                "usage": {"total_tokens": 12},
                "attempts": 1,
            },
        )
        record.result = AgentResult(
            answer="123",
            messages=conversation.messages,
            usage=[{"usage": {"total_tokens": 999}, "attempts": 1}],
        )
        return record

    monkeypatch.setattr(Service, "wait", wait)
    output = root / "native_experiment"
    live = await run_benchmark(Service(root), dataset, output)
    offline = score_predictions(dataset, output / "predictions.jsonl", root / "offline.json")
    assert live["composite_metrics"] == offline["composite_metrics"]
    assert live["composite_metrics"]["total_model_tokens"] == 60
    assert live["composite_metrics"]["final_score"] == 100
    for actual, restored in zip(live["results"], offline["results"], strict=True):
        assert actual["scores"] == restored["scores"]
        assert actual["completion_status"] == "Completed"
        assert actual["tool_chain_completed"] is True


async def test_resume_rejects_obsolete_scores_without_overwriting_report(benchmark_project):
    root, dataset = benchmark_project
    output = root / "obsolete_experiment"
    await run_benchmark(Service(root), dataset, output)
    summary = (output / "summary.json").read_bytes()
    saved = output / "tasks/0001/result.json"
    row = json.loads(saved.read_text())
    row["scores"].pop("protocol")
    saved.write_text(json.dumps(row))
    resumed = Service(root)
    with pytest.raises(ValueError, match="different scoring protocol"):
        await run_benchmark(resumed, dataset, output)
    assert not resumed.records
    assert (output / "summary.json").read_bytes() == summary
    assert not (output / ".run.lock").exists()


async def test_unknown_ids_are_rejected_before_submission(benchmark_project):
    root, dataset = benchmark_project
    service = Service(root)
    with pytest.raises(ValueError, match="Unknown benchmark question IDs"):
        await run_benchmark(service, dataset, root / "unknown", ids={100})
    assert not service.records
