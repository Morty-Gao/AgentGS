"""Execution evidence, search decisions and durable benchmark batches."""

import argparse
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/benchmark"))
import _runtime as runtime  # noqa: E402


@pytest.fixture
def geo():
    spec = importlib.util.spec_from_file_location(
        "tested_geoagent", ROOT / "forks/GeoAgent/run_geotooleval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.modules.pop("geotool_bridge", None)
    sys.path.insert(0, str(ROOT / "forks/GeoAgent"))
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(ROOT / "forks/GeoAgent"))
    sys.modules.pop("geotool_bridge", None)
    if previous is not None:
        sys.modules["geotool_bridge"] = previous


@pytest.fixture
def row():
    return {
        "question_id": 1,
        "category_id": 1,
        "question": "Calculate a value",
        "input_resource": "data/input",
        "answer_template": "",
        "answer_instruction": "",
        "expected_answer": "private reference",
    }


def test_timeout_retains_partial_output(geo, tmp_path):
    result = geo._execute("import time\nprint('partial', flush=True)\ntime.sleep(5)", tmp_path, 0.1)
    assert not result.ok
    assert "partial" in result.stdout
    assert "TimeoutError" in result.stderr
    assert (Path(result.directory) / "program.py").exists()


def test_search_rejects_final_from_failed_program(geo, row, tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "decompose", lambda *a: (["calculate"], -0.1))
    monkeypatch.setattr(
        geo,
        "_propose",
        lambda *a: [("print('FINAL: invalid')\nraise RuntimeError('failed')", -0.1)],
    )
    result = geo.search({}, row, tmp_path, 1, 1, 0, 5, None)
    assert result["status"] == "failed"
    assert result["answer"] == ""
    assert result["trace"]["events"][-1]["reward"] == 0


def test_unusable_sample_does_not_discard_other_candidates(geo, row, monkeypatch):
    generated = []

    def generate(*args):
        generated.append(1)
        if len(generated) == 1:
            raise geo.GenerationError("truncated")
        return {"code": "print('FINAL: 42')"}, -0.2

    monkeypatch.setattr(geo, "_generate", generate)
    events = []
    candidates = geo._propose({}, row, "calculate", "", 2, None, events)
    assert len(generated) == 2
    assert candidates == [("print('FINAL: 42')", -0.2)]
    assert events[0]["phase"] == "generation_error"


def test_success_preserves_multiline_answer_and_code(geo, row, tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "decompose", lambda *a: (["calculate"], -0.1))
    monkeypatch.setattr(geo, "_propose", lambda *a: [("print('FINAL: first\\nsecond')", -0.2)])
    result = geo.search({}, row, tmp_path, 1, 1, 0, 5, None)
    assert result["answer"] == "first\nsecond"
    assert result["selected_execution_id"] == result["trace"]["selected_execution_id"]
    assert (Path(result["artifacts_directory"]) / "program.py").read_text() == result["code"]


def test_repairs_are_bounded_before_remaining_replan(geo, row, tmp_path, monkeypatch):
    repairs, plans = [], []

    def plan(*args):
        plans.append(args[4] if len(args) > 4 else "")
        return (["finish"] if len(plans) > 1 else ["initialize", "broken"], -0.1)

    def propose(env, row, step, *args):
        return [
            (
                "x = 42"
                if step == "initialize"
                else "print('FINAL:', x)"
                if step == "finish"
                else "raise RuntimeError('bad')",
                -0.1,
            )
        ]

    def repair(*args):
        repairs.append(1)
        return "x = 42\nraise RuntimeError('still bad')", -0.1

    monkeypatch.setattr(geo, "decompose", plan)
    monkeypatch.setattr(geo, "_propose", propose)
    monkeypatch.setattr(geo, "_repair", repair)
    monkeypatch.setattr(geo, "_complete", lambda *a: ["raise RuntimeError('bad')"])
    result = geo.search({}, row, tmp_path, 7, 1, 2, 5, None)
    assert len(repairs) == 2
    assert "x = 42" in plans[1]
    assert "raise" not in plans[1]
    assert result["answer"] == "42"
    assert result["trace"]["pruned_trees"]


def test_simulation_reward_counts_successful_steps(geo, row, tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "decompose", lambda *a: (["prefix", "one", "two", "three"], -0.1))
    monkeypatch.setattr(geo, "_propose", lambda *a: [("x = 1", -0.1)])
    monkeypatch.setattr(
        geo, "_complete", lambda *a: ["x += 1", "raise RuntimeError('failure')", "x += 1"]
    )
    result = geo.search({}, row, tmp_path, 1, 1, 0, 5, None)
    rewards = [e["reward"] for e in result["trace"]["events"] if e["phase"] == "backprop"]
    assert rewards == [pytest.approx(1 / 3)]


def test_code_probability_uses_api_field_tokens():
    text = '{"code": "print(42)"}'
    start = text.index('"print')
    tokens = [
        {
            "token": c,
            "bytes": list(c.encode()),
            "logprob": -0.2 if start <= i < len(text) - 1 else -5.0,
        }
        for i, c in enumerate(text)
    ]
    event = {"response": {"content": text}, "logprobs": {"content": tokens}}
    assert runtime.average_logprob(event, "code") == pytest.approx(-0.2)
    event["logprobs"] = None
    with pytest.raises(RuntimeError, match="requires API token logprobs"):
        runtime.average_logprob(event, "code")


def test_failed_http_request_has_durable_usage_evidence(tmp_path):
    events = []
    token = runtime._checkpoint.set(tmp_path / "events.json")
    try:
        with httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(500))
        ) as client:
            with pytest.raises(httpx.HTTPStatusError):
                runtime.chat(
                    {"LLM_BASE_URL": "http://model/v1", "LLM_MODEL": "test"},
                    [{"role": "user", "content": "test"}],
                    client,
                    events,
                    stage="test",
                )
    finally:
        runtime._checkpoint.reset(token)
    assert json.loads((tmp_path / "events.json").read_text())[0]["status"] == "failed"
    conv = runtime.conversation("test", events)
    from agentgs.evaluation.analysis.conversation import conversation_usage

    assert conversation_usage(conv)["usage_complete"] is False


def test_request_archive_is_independent_of_later_dialogue_mutation():
    captured = []

    def transport(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}
                ],
                "usage": {"total_tokens": 4},
            },
        )

    messages = [{"role": "user", "content": "calculate"}]
    events = []
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        runtime.chat(
            {"LLM_BASE_URL": "http://model/v1", "LLM_MODEL": "test"},
            messages,
            client,
            events,
            stage="react",
        )
    messages[0]["content"] = "changed"
    messages.append({"role": "assistant", "content": "42"})
    assert events[0]["request"] == captured[0]


def test_provider_call_extensions_are_retained_in_trace_and_normalized_for_scoring():
    from agentgs.evaluation.analysis.conversation import tool_executions

    raw = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call",
                "index": 0,
                "type": "function",
                "function": {"name": "Tool", "arguments": "{}"},
            }
        ],
    }
    reply = {
        "role": "tool",
        "tool_call_id": "call",
        "content": "42",
        "metadata": {"tool_input": {"x": 1}, "status": "success"},
    }
    conv = runtime.conversation(
        "test", [], dialogue=[{"role": "user", "content": "test"}, raw, reply]
    )
    assert raw["tool_calls"][0]["index"] == 0
    assert tool_executions(conv)[0]["tool_input"] == {"x": 1}


def test_caught_tool_failure_invalidates_execution(geo, tmp_path):
    result = geo._execute(
        "from geotool_bridge import call_tool\ntry:\n call_tool('NoSuchTool', {})\nexcept Exception:\n pass\nprint('FINAL: fabricated')",
        tmp_path,
        5,
    )
    assert not result.ok
    assert result.calls[0]["status"] == "error"
    assert result.calls[0]["name"] == "NoSuchTool"


def test_concurrent_scientific_artifacts_and_receipts_are_isolated(tmp_path, monkeypatch):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    project = tmp_path / "project"
    data = project / "data"
    data.mkdir(parents=True)
    source = data / "input.tif"
    pixels = np.array([[1, 2], [3, 4]], dtype="float64")
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float64",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
    ) as target:
        target.write(pixels, 1)
    monkeypatch.setattr(runtime, "ROOT", project)

    def run(index):
        work = project / "outputs" / str(index)
        receipt = work / "calls.jsonl"
        with runtime.tool_workspace(work, receipt):
            result = runtime.call_tool(
                "RemoteSensingTiffAverage",
                {"file_list": [str(source)], "output_file": "same.tif"},
                strict=True,
            )
        return result, json.loads(receipt.read_text())

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    paths = [Path(r[0]["output_path"]) for r in results]
    assert paths[0] != paths[1]
    for index, (result, receipt) in enumerate(results):
        assert Path(result["output_path"]).is_relative_to(project / "outputs" / str(index))
        assert receipt["status"] == "success"
        assert receipt["result"] == result
        with rasterio.open(result["output_path"]) as saved:
            np.testing.assert_array_equal(saved.read(1), pixels)


def test_603_unique_questions_are_required(tmp_path):
    rows = [json.loads(line) for line in (ROOT / "data/geotooleval.jsonl").read_text().splitlines()]
    rows[-1] = rows[0]
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="duplicate"):
        runtime.load_questions(path)


def test_parallel_batch_resume_retry_and_identity(tmp_path, row, monkeypatch):
    project = tmp_path / "project"
    source = project / "forks/agent/run.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source")
    source.with_name("geotool_bridge.py").write_text("# bridge")
    support = project / "scripts/benchmark/_runtime.py"
    support.parent.mkdir(parents=True)
    support.write_text("# runtime")
    data = project / "data/geotooleval.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text("dataset")
    monkeypatch.setattr(runtime, "ROOT", project)
    monkeypatch.setattr(runtime, "catalog", lambda: [])
    monkeypatch.setattr(
        runtime, "environment", lambda args: {"LLM_MODEL": "test", "LLM_BASE_URL": "http://test/v1"}
    )
    rows = [{**row, "question_id": i} for i in range(1, 5)]
    monkeypatch.setattr(runtime, "load_questions", lambda *a: rows)
    args = argparse.Namespace(
        questions=None,
        output=project / "outputs/predictions.jsonl",
        ids=None,
        concurrency=2,
        request_timeout=5,
        max_output_tokens=100,
        retry_failed=False,
        retries=0,
    )
    active, peak, executions = 0, 0, []
    lock = threading.Lock()

    def run(env, row, attempt, client, events):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            executions.append(attempt)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {
            "answer": "" if row["question_id"] == 2 and not args.retry_failed else "42",
            "conversation": runtime.conversation("test", []),
        }

    assert runtime.batch(args, "test", source, run) == 1
    assert peak == 2
    first = args.output.read_bytes()
    unrelated = project / "src/agentgs/visualization/backend/routes.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("# unrelated interface change")
    assert runtime.batch(args, "test", source, run) == 1
    assert args.output.read_bytes() == first
    assert len(executions) == 4
    with args.output.open("ab") as stream:
        stream.write(b'{"interrupted":')
    args.retry_failed = True
    assert runtime.batch(args, "test", source, run) == 0
    assert len(executions) == 5
    records = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len(records) == 4
    history = args.output.parent / "predictions.run/attempts.jsonl"
    attempts = [json.loads(line) for line in history.read_text().splitlines()]
    assert len(attempts) == 5
    assert len({r["attempt_directory"] for r in attempts}) == 5
    assert all((p / "conversation.json").exists() for p in executions)
    source.write_text("# changed source")
    with pytest.raises(ValueError, match="identity differs"):
        runtime.batch(args, "test", source, run)
    assert not (args.output.parent / "predictions.run/run.lock").exists()


@pytest.mark.parametrize("success_at", [1, 2, 4, None])
@pytest.mark.parametrize("unknown_first_usage", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_automatic_retries_preserve_budget_concurrency_and_all_usage(
    tmp_path, row, monkeypatch, success_at, unknown_first_usage, interrupted
):
    from agentgs.evaluation.analysis.conversation import conversation_usage

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
        runtime, "environment", lambda args: {"LLM_MODEL": "test", "LLM_BASE_URL": "http://test/v1"}
    )
    rows = [{**row, "question_id": qid, "question": str(qid)} for qid in [1, 2]]
    monkeypatch.setattr(runtime, "load_questions", lambda *a: rows)
    interruption_seen = False

    def sleep(duration):
        nonlocal interruption_seen
        if interrupted and not interruption_seen:
            interruption_seen = True
            raise RuntimeError("interrupted after durable attempt")

    monkeypatch.setattr(runtime.time, "sleep", sleep)
    attempts = {1: 0, 2: 0}
    lock = threading.Lock()
    first_requests = threading.Barrier(2)
    active, peak = 0, 0

    def transport(request):
        nonlocal active, peak
        qid = int(json.loads(request.content)["messages"][0]["content"])
        with lock:
            active += 1
            peak = max(peak, active)
        if attempts[qid] == 1:
            first_requests.wait(timeout=5)
        with lock:
            active -= 1
        if attempts[qid] == 1 and unknown_first_usage:
            return httpx.Response(500)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}
                ],
                "usage": {"total_tokens": 5},
            },
        )

    original_client = httpx.Client
    monkeypatch.setattr(
        runtime.httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(transport), **kwargs),
    )
    args = argparse.Namespace(
        questions=None,
        output=project / "predictions.jsonl",
        ids=None,
        concurrency=2,
        request_timeout=5,
        max_output_tokens=50,
        retry_failed=False,
        retries=3,
    )

    def run(env, row, attempt, client, events):
        qid = row["question_id"]
        attempts[qid] += 1
        runtime.chat(env, [{"role": "user", "content": str(qid)}], client, events, stage="test")
        succeeded = success_at is not None and attempts[qid] >= success_at
        return {
            "answer": "42" if succeeded else "",
            "status": "succeeded" if succeeded else "failed",
            "trace": {"events": events},
            "conversation": runtime.conversation(str(qid), events),
        }

    try:
        code = runtime.batch(args, "test", source, run)
    except RuntimeError as exc:
        assert interrupted and "interrupted after durable attempt" in str(exc)
        code = runtime.batch(args, "test", source, run)
    expected = max(success_at or 4, 2 if unknown_first_usage else 1)
    assert attempts == {1: expected, 2: expected}
    assert peak == 2
    assert code == (1 if success_at is None else 0)
    records = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len(records) == 2
    history = args.output.parent / "predictions.run/attempts.jsonl"
    assert len(history.read_text().splitlines()) == 2 * expected
    for record in records:
        assert record["automatic_attempts_used"] == expected
        assert not record["retry_pending"]
        assert len(record["attempt_directories"]) == expected
        assert all((Path(p) / "result.json").exists() for p in record["attempt_directories"])
        usage = conversation_usage(record["conversation"])
        assert usage["reported_tokens"] == 5 * (expected - int(unknown_first_usage))
        assert usage["usage_complete"] is not unknown_first_usage
    before = args.output.read_bytes()
    assert runtime.batch(args, "test", source, run) == code
    assert args.output.read_bytes() == before
    assert attempts == {1: expected, 2: expected}


@pytest.mark.parametrize(
    "text,answer",
    [
        ("FINAL: 42", "42"),
        ("  final : 42\r\n", "42"),
        ("FinalAnswer: 42", "42"),
        ("## **Final answer:** 42", "42"),
        ("**FINAL**：42", "42"),
        ("最终答案：42", "42"),
        ("```text\nfinal: first\nsecond\n```", "first\nsecond"),
        ("diagnostic\nFinal Answer:\nfirst\nsecond", "first\nsecond"),
        ('{"final_answer": "first\\nsecond"}', "first\nsecond"),
        ('```json\n{"answer": 0}\n```', "0"),
        ('FINAL: {"answer": "42"}', "42"),
        ("FINAL: old\nfinal: revised", "revised"),
    ],
)
def test_final_output_wrappers_are_extracted_without_reference(text, answer):
    from _answers import extract_answer

    assert extract_answer(text)["answer"] == answer


@pytest.mark.parametrize(
    "text",
    [
        "",
        "FINAL:",
        "```text\nFINAL:\n```",
        '{"answer": null}',
        '{"answer":"42","final_answer":"43"}',
    ],
)
def test_empty_or_conflicting_final_output_does_not_fall_back(text):
    from _answers import extract_answer, extract_program_answer

    assert not extract_answer(text, allow_plain=True)["answer"]
    assert not extract_program_answer(text, terminal=True)["answer"]


def test_plain_program_output_requires_terminal_boundary_and_new_output():
    from _answers import extract_program_answer

    assert not extract_program_answer("42\n")["answer"]
    assert extract_program_answer("42\n", terminal=True)["answer"] == "42"
    assert not extract_program_answer("42\n", terminal=True, prefix_stdout="42\n")["answer"]
    assert not extract_program_answer("43\n", terminal=True, prefix_stdout="42\n")["answer"]
    result = extract_program_answer(
        "debug 10\nfirst\nsecond\n", terminal=True, prefix_stdout="debug 10\n"
    )
    assert result == {"answer": "first\nsecond", "method": "terminal_stdout"}


def test_unmarked_program_output_is_only_accepted_after_all_steps(geo, row, tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "decompose", lambda *a: (["prepare", "calculate", "finish"], -0.1))
    monkeypatch.setattr(geo, "_propose", lambda *a: [("print(41)", -0.1)])
    monkeypatch.setattr(geo, "_complete", lambda *a: ["print(42)", "print(43)"])
    result = geo.search({}, row, tmp_path, 1, 1, 0, 5, None)
    assert result["answer"] == "43" and result["status"] == "succeeded"
    assert result["answer_extraction"]["method"] == "terminal_stdout"
    assert result["output"] == "41\n42\n43"
    assert sum(e["phase"] == "execute" for e in result["trace"]["events"]) == 3


def test_completed_plan_cannot_reuse_an_intermediate_print_as_answer(
    geo, row, tmp_path, monkeypatch
):
    monkeypatch.setattr(geo, "decompose", lambda *a: (["prepare", "finish"], -0.1))
    monkeypatch.setattr(geo, "_propose", lambda *a: [("print(42)", -0.1)])
    monkeypatch.setattr(geo, "_complete", lambda *a: ["x = 1"])
    result = geo.search({}, row, tmp_path, 1, 1, 0, 5, None)
    assert result["answer"] == "" and result["status"] == "failed"
    assert result["output"] == "42"


@pytest.fixture
def earth():
    spec = importlib.util.spec_from_file_location(
        "tested_earthagent", ROOT / "forks/Earth-Agent/run_geotooleval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.modules.pop("geotool_bridge", None)
    sys.path.insert(0, str(ROOT / "forks/Earth-Agent"))
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(ROOT / "forks/Earth-Agent"))
    sys.modules.pop("geotool_bridge", None)
    if previous is not None:
        sys.modules["geotool_bridge"] = previous


def test_earth_retains_raw_message_and_extracts_final_answer(earth, row, tmp_path, monkeypatch):
    raw = "## **Final answer:**\n67°45′0.00″ N"
    monkeypatch.setattr(
        earth,
        "chat",
        lambda *a, **k: {
            "response": {"role": "assistant", "content": raw, "reasoning_content": "original"},
            "finish_reason": "stop",
        },
    )
    outcome = earth.answer_question({}, [], row, 1, None, tmp_path, [])
    assert outcome["answer"] == "67°45′0.00″ N"
    assert outcome["status"] == "succeeded"
    assert outcome["trace"]["turns"][0]["assistant"]["content"] == raw
    assert outcome["trace"]["turns"][0]["assistant"]["reasoning_content"] == "original"
    assert outcome["conversation"]["messages"][-1]["content"] == raw


@pytest.mark.parametrize("kind", ["earth", "geo"])
def test_fork_predictions_reach_shared_robust_scorer(kind, earth, geo, row, tmp_path, monkeypatch):
    from agentgs.benchmark.score import score_predictions

    raw = "  final answer：67°45′0.00″ N"
    if kind == "earth":
        monkeypatch.setattr(
            earth,
            "chat",
            lambda *a, **k: {
                "response": {"role": "assistant", "content": raw},
                "finish_reason": "stop",
            },
        )
        outcome = earth.answer_question({}, [], row, 1, None, tmp_path, [])
    else:
        monkeypatch.setattr(geo, "decompose", lambda *a: (["finish"], -0.1))
        monkeypatch.setattr(geo, "_propose", lambda *a: [(f"print({raw!r})", -0.1)])
        outcome = geo.search({}, row, tmp_path, 1, 1, 0, 5, None)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"question_id": 447, **outcome}) + "\n")
    report = score_predictions(
        ROOT / "data/geotooleval.jsonl", predictions, tmp_path / "scores.json", allow_partial=True
    )
    assert report["results"][0]["answer_score"]["method"] == "dms_coordinate"
    assert report["micro_fas"] == 1
    assert report["legacy_micro_fas"] == 0


def test_json_answer_wrapper_preserves_explicit_unit_for_scoring():
    from _answers import extract_answer

    from agentgs.evaluation.datasets.base import Example
    from agentgs.evaluation.metrics.answers import score_answer

    answer = extract_answer('{"answer":46.54,"unit":"m"}')["answer"]
    task = Example(1, 1, "fixture", "Compute distance", "46.54 km", "single_number")
    assert score_answer(task, answer)["score"] == 0
