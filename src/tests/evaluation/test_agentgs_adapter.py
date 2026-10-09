"""AgentGS benchmark adapter: public requests, retries, and scoring records."""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/benchmark"))
import _agentgs as agentgs_batch  # noqa: E402

from agentgs.contracts.models import (  # noqa: E402
    AgentResult,
    TaskInput,
    TaskRecord,
    TaskSpec,
    TaskStatus,
    ToolError,
)
from agentgs.evaluation.analysis.conversation import (  # noqa: E402
    conversation_usage,
    tool_executions,
)
from agentgs.evaluation.datasets.geotooleval import EvalTask  # noqa: E402
from agentgs.messages import Conversation  # noqa: E402


def row(question_id):
    return SimpleNamespace(question_id=question_id, category_id=1)


def example():
    return EvalTask(
        question_id=1,
        category_id=1,
        category="DeepEarthTools",
        question="Compute the value",
        expected_answer="SECRET_GOLD",
        answer_type="single_number",
        input_resource={"expected_answer": "SECRET_GOLD", "location_key": "andes"},
        chain_mode="single",
        gold_chain_text="SECRET_CHAIN",
        gold_steps=({"tool_name": "DeepEarthQueryLITHO1", "arguments": {"site": "andes"}},),
    )


def conversation_with(token_total, request_id, tool=None):
    conversation = Conversation()
    conversation.append({"role": "user", "content": "Compute the value"})
    if tool:
        call = conversation.begin_tool(tool, "{}", call_id=f"call-{request_id}")
        conversation.finish_tool(call, '{"value": 2}', status="success", tool_input="{}")
    conversation.record_event(
        "model.completed",
        {
            "request_id": request_id,
            "usage": {"total_tokens": token_total},
            "attempts": 1,
            "status": "completed",
        },
    )
    return conversation


@asynccontextmanager
async def yielding(function):
    yield function


def run(tmp_path, rows, execute, **kwargs):
    return agentgs_batch.run_batch(
        rows=rows,
        output=tmp_path / "predictions.jsonl",
        identity={"protocol": "agentgs-geotooleval-v1"},
        retries=kwargs.pop("retries", 3),
        retry_failed=kwargs.pop("retry_failed", False),
        concurrency=kwargs.pop("concurrency", 2),
        opener=lambda: yielding(execute),
        **kwargs,
    )


def test_request_omits_gold_and_data_only_disables_rule_planning():
    request = agentgs_batch.task_request(
        example(),
        planning_mode="data_only",
        allow_rule_planning=True,
        plan_attempts=None,
        tool_attempts=None,
        command_attempts=None,
        modules=None,
        ablations=[],
        contract_provider=None,
    )
    encoded = request.model_dump_json()
    assert "SECRET_GOLD" not in encoded
    assert "SECRET_CHAIN" not in encoded
    assert request.allow_rule_planning is False
    assert request.input_resource == {"location_key": "andes"}


def test_not_completed_with_answer_is_produced():
    result = TaskRecord(
        spec=TaskSpec(input=TaskInput(question="Compute the value")),
        status=TaskStatus.FAILED,
        result=AgentResult(
            answer="2",
            completion_status="Not Completed",
            messages=[{"role": "user", "content": "Compute the value"}],
        ),
        error=ToolError(code="task_incomplete", message="Not Completed"),
    )
    outcome = agentgs_batch.classify_result(result)
    assert outcome["status"] == "succeeded"
    assert outcome["final_answer"] == "2"
    assert outcome["error"] is None


def test_timeout_is_retried_even_with_partial_answer():
    result = TaskRecord(
        spec=TaskSpec(input=TaskInput(question="Compute the value")),
        status=TaskStatus.FAILED,
        result=AgentResult(
            answer="partial",
            completion_status="Not Completed",
            messages=[{"role": "user", "content": "Compute the value"}],
        ),
        error=ToolError(code="timeout", message="Task admission/execution time budget exceeded"),
    )
    outcome = agentgs_batch.classify_result(result)
    assert outcome["status"] == "failed"
    assert outcome["answer"] == "partial"
    assert "timeout" in outcome["error"]


def test_retries_aggregate_usage_and_keep_the_final_tools(tmp_path, monkeypatch):
    calls = {"n": 0}

    async def execute(item, attempt):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "answer": "",
                "error": "worker_exit: boom",
                "conversation": conversation_with(3, "first", tool="Other"),
            }
        return {
            "answer": "2",
            "conversation": conversation_with(5, "second", tool="DeepEarthQueryLITHO1"),
        }

    async def sleep(_delay):
        return None

    monkeypatch.setattr(agentgs_batch.asyncio, "sleep", sleep)
    assert run(tmp_path, [row(1)], execute, retries=3, concurrency=1) == 0
    assert calls["n"] == 2
    saved = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert saved["final_answer"] == "2"
    assert "attempt_conversation" not in saved
    usage = conversation_usage(saved["conversation"])
    assert usage["reported_tokens"] == 8
    assert usage["usage_complete"] is True
    assert [item["tool_name"] for item in tool_executions(saved["conversation"])] == [
        "DeepEarthQueryLITHO1"
    ]
    attempts = [
        json.loads(line)
        for line in (tmp_path / "predictions.run/attempts.jsonl").read_text().splitlines()
    ]
    assert len(attempts) == 2
    local = json.loads((Path(attempts[0]["attempt_directory"]) / "result.json").read_text())
    assert "Other" in json.dumps(local["attempt_conversation"])
    assert run(tmp_path, [row(1)], execute, retries=3, concurrency=1) == 0
    assert calls["n"] == 2


def test_interruption_keeps_the_retry_budget(tmp_path, monkeypatch):
    calls = {"n": 0}
    interrupted = {"done": False}

    async def execute(item, attempt):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"answer": "", "error": "worker_exit: boom", "conversation": Conversation()}
        return {"answer": "2", "conversation": conversation_with(4, "ok")}

    async def sleep(_delay):
        if not interrupted["done"]:
            interrupted["done"] = True
            raise RuntimeError("interrupted after durable attempt")

    monkeypatch.setattr(agentgs_batch.asyncio, "sleep", sleep)
    with pytest.raises(RuntimeError, match="interrupted after durable attempt"):
        run(tmp_path, [row(1)], execute, retries=3, concurrency=1)
    assert not (tmp_path / "predictions.run/run.lock").exists()
    monkeypatch.setattr(agentgs_batch.asyncio, "sleep", lambda _delay: asyncio.sleep(0))
    assert run(tmp_path, [row(1)], execute, retries=3, concurrency=1) == 0
    saved = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert saved["automatic_attempts_used"] == 2
    assert saved["status"] == "succeeded"
    assert calls["n"] == 2


def test_authentication_failure_does_not_start_later_questions(tmp_path):
    called = []

    async def execute(item, attempt):
        called.append(item.question_id)
        return {
            "answer": "",
            "error": "provider: HTTP 401 unauthorized",
            "conversation": Conversation(),
        }

    assert run(tmp_path, [row(1), row(2)], execute, retries=3, concurrency=1) == 1
    assert called == [1]
    saved = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert saved["question_id"] == 1
    assert saved["retry_pending"] is False
    report = json.loads((tmp_path / "predictions.run/summary.json").read_text())
    assert report["stopped_for_auth"] is True
    assert report["failed"] == 2


def test_parallel_attempts_and_identity_lock(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()
    active = {"now": 0, "peak": 0}

    async def execute(item, attempt):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        if active["now"] == 1:
            started.set()
            await release.wait()
        else:
            await started.wait()
            release.set()
        active["now"] -= 1
        return {"answer": "2", "conversation": Conversation()}

    assert run(tmp_path, [row(1), row(2)], execute, retries=0, concurrency=2) == 0
    assert active["peak"] == 2
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == 2
    with (tmp_path / "predictions.jsonl").open("ab") as stream:
        stream.write(b'{"interrupted":')
    assert run(tmp_path, [row(1), row(2)], execute, retries=0, concurrency=2) == 0
    assert [
        json.loads(line)["question_id"]
        for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ] == [1, 2]
    (tmp_path / "predictions.run/manifest.json").write_text('{"protocol": "other"}')
    with pytest.raises(ValueError, match="identity differs"):
        run(tmp_path, [row(1)], execute, retries=0, concurrency=1)
    assert not (tmp_path / "predictions.run/run.lock").exists()


def test_corrupt_middle_record_is_rejected(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(b'{"broken":\n{"question_id": 1}\n')
    with pytest.raises(ValueError, match="corrupt non-final"):
        agentgs_batch._load_latest(path)


def test_other_host_lock_is_kept(tmp_path):
    output = tmp_path / "predictions.jsonl"
    lock = tmp_path / "predictions.run/run.lock"
    lock.parent.mkdir()
    lock.write_text(json.dumps({"pid": os.getpid(), "hostname": "other-host"}))

    async def execute(item, attempt):
        return {"answer": "2", "conversation": Conversation()}

    with pytest.raises(RuntimeError, match="another host"):
        run(tmp_path, [row(1)], execute, retries=0, concurrency=1)
    assert lock.is_file()
    assert not output.exists()


def test_identity_records_effective_algorithm_without_secrets(tmp_path):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("{}\n")
    source = tmp_path / "src/agentgs"
    source.mkdir(parents=True)
    (source / "marker.py").write_text("VALUE = 1\n")
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / "raw/secret.bin").write_text("nope")
    (data / "kept.txt").write_text("kept")
    identity = agentgs_batch.build_identity(
        tmp_path,
        dataset,
        planning_mode="data_only",
        allow_rule_planning=True,
        plan_attempts=None,
        tool_attempts=None,
        command_attempts=None,
        modules=None,
        ablations=[],
        answer_contracts=False,
        contract_sha256=None,
        retries=3,
        runtime={"max_concurrent_tasks": 2},
        model_key="primary",
        model_id="demo",
        base_url="http://model.example/v1",
        benchmark_scope="common_data",
    )
    encoded = json.dumps(identity)
    assert "api_key" not in encoded
    assert "secret" not in identity["base_url_sha256"]
    assert identity["allow_rule_planning"] is False
    assert identity["planning_mode"] == "data_only"
    assert list(identity["prepared_inputs"]) == ["kept.txt"]
    again = agentgs_batch.build_identity(
        tmp_path,
        dataset,
        planning_mode="data_only",
        allow_rule_planning=True,
        plan_attempts=None,
        tool_attempts=None,
        command_attempts=None,
        modules=None,
        ablations=[],
        answer_contracts=False,
        contract_sha256=None,
        retries=3,
        runtime={"max_concurrent_tasks": 2},
        model_key="primary",
        model_id="demo",
        base_url="http://model.example/v1",
        benchmark_scope="common_data",
    )
    assert again == identity


def test_cli_help():
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/benchmark/agentgs/run.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--planning-mode" in result.stdout
    assert "--retries" in result.stdout


@pytest.mark.parametrize(
    "relative",
    [
        "skills/resources/definitions.json",
        "skills/resources/compositions.json",
        "skills/resources/capability_extensions.json",
        "contracts/catalog.json",
        "contracts/tool_scope.json",
        "skills/implementations/example/prompt.md",
    ],
)
def test_resume_rejects_changed_runtime_declarations(tmp_path, relative):
    resource = tmp_path / "src/agentgs" / relative
    resource.parent.mkdir(parents=True)
    resource.write_text("{}")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")

    def identity():
        return agentgs_batch.build_identity(
            tmp_path,
            dataset,
            planning_mode="skill_guided",
            allow_rule_planning=False,
            plan_attempts=None,
            tool_attempts=None,
            command_attempts=None,
            modules=None,
            ablations=[],
            answer_contracts=False,
            contract_sha256=None,
            retries=0,
            runtime={},
            model_key="test",
            model_id="test",
            base_url="http://fixture.invalid",
            benchmark_scope="full",
        )

    async def execute(item, attempt):
        return {"answer": "2", "conversation": conversation_with(1, "first")}

    def batch(policy):
        return agentgs_batch.run_batch(
            rows=[row(1)],
            output=tmp_path / "predictions.jsonl",
            identity=policy,
            retries=0,
            retry_failed=False,
            concurrency=1,
            opener=lambda: yielding(execute),
        )

    before = identity()
    assert batch(before) == 0
    resource.write_text('{"changed": true}')
    after = identity()
    assert before != after
    with pytest.raises(ValueError):
        batch(after)
