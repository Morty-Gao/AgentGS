"""Retry class, slot release, and failure_class on benchmark runners."""

import argparse
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/benchmark"))
import _agentgs as agentgs_batch  # noqa: E402
import _runtime as runtime  # noqa: E402

from agentgs.messages import Conversation  # noqa: E402


@asynccontextmanager
async def yielding(function):
    yield function


def run_agentgs(tmp_path, rows, execute, **kwargs):
    return agentgs_batch.run_batch(
        rows=rows,
        output=tmp_path / "predictions.jsonl",
        identity={"protocol": "agentgs-geotooleval-v1"},
        retries=kwargs.pop("retries", 3),
        retry_failed=False,
        concurrency=kwargs.pop("concurrency", 1),
        opener=lambda: yielding(execute),
    )


def row(question_id):
    return argparse.Namespace(question_id=question_id, category_id=1)


def saved_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize(
    ("error", "klass", "limit", "attempts"),
    [
        ("ReadTimeout: timed out", "model_timeout", 5, 6),
        ("ConnectError: reset", "model_transport", 5, 6),
        ("timeout: Tool invocation exceeded its time budget", "other_timeout", 10, 4),
        ("worker_exit: boom", "other", 10, 4),
    ],
)
def test_retry_class_budgets(tmp_path, monkeypatch, error, klass, limit, attempts):
    calls = {"n": 0}

    async def execute(item, attempt):
        calls["n"] += 1
        return {"answer": "", "error": error, "conversation": Conversation()}

    async def sleep(_delay):
        return None

    monkeypatch.setattr(agentgs_batch.asyncio, "sleep", sleep)
    assert run_agentgs(tmp_path, [row(1)], execute, retries=limit) == 1
    assert calls["n"] == attempts
    saved = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert saved["failure_class"] == klass
    assert saved["automatic_attempts_used"] == attempts
    assert saved["retry_pending"] is False


def test_infrastructure_retries_do_not_consume_the_budget(tmp_path, monkeypatch):
    calls = {"n": 0}

    async def execute(item, attempt):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "answer": "",
                "error": "OperationalError: database is locked",
                "conversation": Conversation(),
            }
        return {"answer": "2", "conversation": Conversation()}

    async def sleep(_delay):
        return None

    monkeypatch.setattr(agentgs_batch.asyncio, "sleep", sleep)
    assert run_agentgs(tmp_path, [row(1)], execute, retries=0) == 0
    assert calls["n"] == 2
    history = saved_lines(tmp_path / "predictions.run/attempts.jsonl")
    assert history[0]["failure_class"] == "infrastructure"
    assert history[0]["automatic_attempts_used"] == 0
    assert history[0]["retry_pending"] is True
    saved = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert saved["status"] == "succeeded"
    assert saved["automatic_attempts_used"] == 1
    assert "failure_class" not in saved


def test_model_timeout_releases_the_slot_before_the_next_attempt(tmp_path, monkeypatch):
    order = []
    counts = {1: 0, 2: 0}

    async def execute(item, attempt):
        counts[item.question_id] += 1
        order.append((item.question_id, counts[item.question_id]))
        if item.question_id == 1 and counts[1] < 5:
            return {"answer": "", "error": "ReadTimeout: timed out", "conversation": Conversation()}
        return {"answer": "2", "conversation": Conversation()}

    async def sleep(_delay):
        return None

    monkeypatch.setattr(agentgs_batch.asyncio, "sleep", sleep)
    assert run_agentgs(tmp_path, [row(1), row(2)], execute, retries=10, concurrency=1) == 0
    assert order.index((2, 1)) < order.index((1, 3))


def test_failure_class_matches_the_shared_labels():
    samples = {
        "ReadTimeout: timed out": "model_timeout",
        "WriteTimeout: timed out": "model_timeout",
        "ConnectTimeout: timed out": "model_timeout",
        "PoolTimeout: timed out": "model_timeout",
        "TransportFailure: LLM request deadline exceeded": "model_timeout",
        "TransportFailure: LLM transport/retry budget exhausted": "model_timeout",
        "RemoteProtocolError: peer closed": "model_transport",
        "ReadError: incomplete": "model_transport",
        "ConnectError: refused": "model_transport",
        "TransportFailure: connection reset": "model_transport",
        "timeout: Tool invocation exceeded its time budget": "other_timeout",
        "timeout: Task admission/execution time budget exceeded": "other_timeout",
        "Tool-call budget exhausted without a final answer": "other_timeout",
        "Execution of the code snippet has timed out for exceeding 20 seconds.": "other_timeout",
        "TimeoutExpired: program": "other_timeout",
        "probe timed out": "other_timeout",
        "QueueFull: Runtime task queue is full": "infrastructure",
        "OperationalError: database is locked": "infrastructure",
        "OperationalError: database is busy": "infrastructure",
        "runtime_error: OperationalError: database is locked": "infrastructure",
        "provider: HTTP 401 unauthorized": "authentication",
        "provider: HTTP 403 forbidden": "authentication",
        "worker_exit: boom": "other",
    }
    for text, klass in samples.items():
        assert runtime.failure_class(text) == klass


def _fork_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    source = project / "forks/agent/run.py"
    source.parent.mkdir(parents=True)
    source.write_text("# agent")
    source.with_name("geotool_bridge.py").write_text("# bridge")
    support = project / "scripts/benchmark/_runtime.py"
    support.parent.mkdir(parents=True)
    support.write_text("# runtime")
    dataset = project / "data/geotooleval.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("dataset")
    monkeypatch.setattr(runtime, "ROOT", project)
    monkeypatch.setattr(runtime, "catalog", lambda: [])
    monkeypatch.setattr(
        runtime,
        "environment",
        lambda args: {"LLM_MODEL": "test", "LLM_BASE_URL": "http://test/v1"},
    )
    return project, source


def _fork_args(project, **kwargs):
    return argparse.Namespace(
        questions=None,
        output=project / "predictions.jsonl",
        ids=None,
        concurrency=kwargs.get("concurrency", 1),
        request_timeout=5,
        max_output_tokens=50,
        retry_failed=False,
        retries=kwargs.get("retries", 3),
    )


def test_fork_class_budgets_and_slot_release(tmp_path, monkeypatch):
    project, source = _fork_project(tmp_path, monkeypatch)
    rows = [
        {
            "question_id": qid,
            "category_id": 1,
            "question": str(qid),
            "input_resource": "data/input",
            "answer_template": "",
            "answer_instruction": "",
            "expected_answer": "hidden",
        }
        for qid in (1, 2)
    ]
    monkeypatch.setattr(runtime, "load_questions", lambda *args: rows)
    monkeypatch.setattr(runtime.time, "sleep", lambda _delay: None)
    order = []
    counts = {1: 0, 2: 0}

    def run(env, row, attempt, client, events):
        qid = row["question_id"]
        counts[qid] += 1
        order.append((qid, counts[qid]))
        failed = qid == 1 and counts[qid] < 5
        return {
            "answer": "" if failed else "42",
            "status": "failed" if failed else "succeeded",
            "error": "ReadTimeout: timed out" if failed else None,
            "trace": {"events": events},
            "conversation": runtime.conversation(str(qid), events),
        }

    args = _fork_args(project, concurrency=1, retries=10)
    assert runtime.batch(args, "test", source, run) == 0
    assert order.index((2, 1)) < order.index((1, 3))
    saved = {row["question_id"]: row for row in saved_lines(args.output)}
    assert saved[1]["automatic_attempts_used"] == 5
    assert "failure_class" not in saved[1]

    counts = {3: 0}

    def exhaust(env, row, attempt, client, events):
        counts[3] += 1
        return {
            "answer": "",
            "status": "failed",
            "error": "timeout: Task admission/execution time budget exceeded",
            "trace": {"events": events},
            "conversation": runtime.conversation("3", events),
        }

    project = tmp_path / "other"
    project.mkdir()
    source = project / "forks/agent/run.py"
    source.parent.mkdir(parents=True)
    source.write_text("# agent")
    source.with_name("geotool_bridge.py").write_text("# bridge")
    monkeypatch.setattr(runtime, "ROOT", project)
    (project / "scripts/benchmark").mkdir(parents=True)
    (project / "scripts/benchmark/_runtime.py").write_text("# runtime")
    (project / "data").mkdir()
    (project / "data/geotooleval.jsonl").write_text("dataset")
    monkeypatch.setattr(
        runtime,
        "load_questions",
        lambda *args: [
            {
                "question_id": 3,
                "category_id": 1,
                "question": "3",
                "input_resource": "data/input",
                "answer_template": "",
                "answer_instruction": "",
                "expected_answer": "hidden",
            }
        ],
    )
    args = _fork_args(project, retries=10)
    args.output = project / "predictions.jsonl"
    assert runtime.batch(args, "test", source, exhaust) == 1
    assert counts[3] == 4
    record = json.loads(args.output.read_text())
    assert record["failure_class"] == "other_timeout"
    assert record["automatic_attempts_used"] == 4
    assert record["retry_pending"] is False
