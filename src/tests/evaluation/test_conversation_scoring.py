import json
from dataclasses import replace

import pytest

from agentgs.evaluation.benchmark import public_input, score_prediction, score_predictions
from agentgs.evaluation.conversation import conversation_usage, tool_executions, usage_summary
from agentgs.evaluation.datasets.geotooleval import EvalTask, load_tasks
from agentgs.evaluation.metrics.tools import SCORING_PROTOCOL, score_task
from agentgs.messages import Conversation, Message


def task():
    return EvalTask(
        1,
        1,
        "DeepEarthTools",
        "Compute the value",
        "2",
        "单数值结果",
        "",
        "single",
        "T",
        ({"tool_name": "T", "arguments": {"x": [1, 3]}},),
    )


def request(conversation, request_id, total, *, attempts=1, status="completed"):
    conversation.record_event("model.started", {"request_id": request_id})
    if status == "started":
        return
    if status in {"failed", "cancelled"}:
        conversation.record_event(f"model.{status}", {"request_id": request_id})
        return
    metadata = {
        "request_id": request_id,
        "usage": {"total_tokens": total},
        "attempts": attempts,
        "status": "completed",
    }
    conversation.append(
        Message(role="assistant", content="2", reasoning="original reasoning", metadata=metadata)
    )
    conversation.record_event("model.completed", metadata)


def native_history():
    conversation = Conversation.from_openai([{"role": "user", "content": "Compute the value"}])
    planning = conversation.begin_tool("submit_tool_plan", '{"tools":["T"]}', call_id="plan")
    conversation.finish_tool(planning, "accepted", operation="planning", status="success")
    rejected = conversation.begin_tool("submit_tool_plan", '{"tools":[]}', call_id="bad-plan")
    conversation.finish_tool(rejected, "rejected", operation="planning", status="failed")
    failed = conversation.begin_tool("T", '{"x":[]}', call_id="retry")
    conversation.finish_tool(failed, "invalid", status="retrying")
    succeeded = conversation.begin_tool("T", '{"artifact_ref":"source"}', call_id="success")
    conversation.finish_tool(succeeded, '{"value":2}', status="success", tool_input='{"x":[1,3]}')
    request(conversation, "final", 30)
    return conversation


@pytest.mark.parametrize("form", ["object", "messages", "serialized", "jsonl"])
def test_native_scoring_uses_call_links_bound_arguments_and_retries(form):
    conversation = native_history()
    before = conversation.model_dump_json()
    value = {
        "object": conversation,
        "messages": conversation.messages,
        "serialized": conversation.model_dump(mode="json"),
        "jsonl": Conversation.from_jsonl(conversation.to_jsonl()),
    }[form]
    scored = score_prediction(task(), "2", value)["scores"]
    assert scored["protocol"] == SCORING_PROTOCOL
    assert scored["tsf1"] == scored["tos"] == scored["pas"] == 1
    assert scored["physical_tool_call_attempts"] == 2
    assert scored["ecr"] == scored["tce"] == 0.5
    assert scored["model_total_tokens"] == 30
    assert scored["accuracy_score"] == 95
    assert scored["cost_score"] == 80
    assert scored["final_score"] == 92.75
    assert conversation.model_dump_json() == before


def test_previous_turn_and_nested_comparison_do_not_change_task_scores():
    conversation = Conversation.from_openai([{"role": "user", "content": "Old question"}])
    call_id = conversation.begin_tool("OtherTool", "{}")
    conversation.finish_tool(call_id, "old result", status="success")
    request(conversation, "previous", 900)
    current = native_history()
    for message in current.messages:
        conversation.append(message)
    conversation.record_event(
        "shared_bridge_path_comparison",
        {
            "content": {"messages": current.model_dump(mode="json")["messages"]},
        },
    )
    scored = score_prediction(task(), "2", conversation)["scores"]
    assert scored["physical_tool_call_attempts"] == 2
    assert scored["model_total_tokens"] == 30
    assert scored["tsf1"] == 1


def test_parallel_tool_replies_are_scored_in_invocation_order():
    conversation = Conversation.from_openai(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "A", "arguments": "{}"}},
                    {"id": "b", "type": "function", "function": {"name": "B", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "2"},
            {"role": "tool", "tool_call_id": "a", "content": "1"},
        ]
    )
    scored = score_task(
        gold_steps=[{"tool_name": "A"}, {"tool_name": "B"}],
        messages=conversation,
        answer_score=1,
    )
    assert scored["tos"] == scored["tsf1"] == 1


def test_plain_openai_planning_function_is_excluded_without_runtime_metadata():
    conversation = Conversation.from_openai(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "plan",
                        "type": "function",
                        "function": {
                            "name": "submit_tool_plan",
                            "arguments": '{"tools":["T"]}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "plan", "content": "accepted"},
        ]
    )
    assert tool_executions(conversation) == []


def test_old_tool_annotations_cannot_duplicate_native_invocations():
    conversation = native_history()
    conversation.record_event(
        "tool_call",
        {
            "call_id": "success",
            "tool_name": "OtherTool",
            "status": "success",
            "tool_input": "{}",
            "tool_output": "fake",
        },
    )
    scored = score_prediction(task(), "2", conversation)["scores"]
    assert scored["physical_tool_call_attempts"] == 2
    assert scored["tsf1"] == scored["pas"] == 1


@pytest.mark.parametrize(
    "status,attempts",
    [
        ("started", 1),
        ("failed", 1),
        ("cancelled", 1),
        ("completed", 2),
    ],
)
def test_conversation_usage_overrides_claimed_complete_summary(status, attempts):
    conversation = native_history()
    request(conversation, "unfinished", 10, status=status, attempts=attempts)
    usage = conversation_usage(conversation, {"usage_complete": True, "total_tokens": 999})
    assert not usage["usage_complete"]
    assert usage["total_tokens"] is None
    assert usage["reported_tokens"] == (40 if status == "completed" else 30)
    scored = score_prediction(task(), "2", conversation)["scores"]
    assert scored["final_score"] is None


def test_checkpoint_reply_before_completed_event_is_counted_once():
    conversation = native_history()
    conversation.messages.pop()  # model.completed is not persisted yet.
    assert conversation_usage(conversation)["total_tokens"] == 30


def test_pending_calls_are_preserved_and_do_not_pass_success_gate():
    conversation = native_history()
    conversation.begin_tool("T", '{"x":[1,3]}', call_id="pending")
    scored = score_prediction(task(), "2", conversation)["scores"]
    assert scored["pending_function_calls"] == 1
    assert not scored["task_success_gate"]
    assert scored["cost_score"] == 0
    assert [event["call_id"] for event in tool_executions(conversation)] == ["retry", "success"]


def test_explicit_tool_result_envelope_supplies_upstream_arguments():
    conversation = Conversation()
    first = conversation.begin_tool("Source", "{}")
    conversation.finish_tool(
        first,
        json.dumps(
            {
                "call_id": first,
                "tool": "Source",
                "succeeded": True,
                "data": {"x": [1, 3]},
            }
        ),
    )
    second = conversation.begin_tool("T", '{"x":[1,3]}')
    conversation.finish_tool(second, "2")
    scored = score_task(
        gold_steps=[
            {"tool_name": "Source", "arguments": {}},
            {"tool_name": "T", "arguments": {"x": "upstream"}, "consumes": ["x"]},
        ],
        messages=conversation,
        answer_score=1,
    )
    assert scored["pas"] == 1


def test_offline_rescoring_round_trips_conversation_and_raw_usage(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "题目序号": 1,
                "所属类别": 1,
                "题目正文": "Compute the value",
                "题目答案": "2",
                "答案类型": "单数值结果",
                "输入资源": {"source": 1, "gold_answer": "PRIVATE"},
                "标准工具链调用": "T",
                "单/多工具链调用": "single",
                "标准工具调用明细": list(task().gold_steps),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loaded = load_tasks(dataset)[0]
    assert public_input(loaded).input_resource == {"source": 1}
    conversation = native_history()
    prediction = tmp_path / "predictions.jsonl"
    prediction.write_text(
        json.dumps(
            {
                "question_id": 1,
                "final_answer": "2",
                "conversation": conversation.model_dump(mode="json"),
                "usage": [{"usage": {"total_tokens": 999}, "attempts": 1}],
                "trace": [{"type": "tool_call", "tool_name": "OtherTool"}],
            }
        ),
        encoding="utf-8",
    )
    report = score_predictions(dataset, prediction, tmp_path / "scores.json")
    assert report["results"][0]["scores"] == score_prediction(loaded, "2", conversation)["scores"]
    assert report["scoring_protocol"] == SCORING_PROTOCOL
    prediction.write_text(
        json.dumps({"question_id": 1, "answer": "2", "trace": [{"type": "tool_call"}]})
    )
    with pytest.raises(ValueError, match="Trace-only"):
        score_predictions(dataset, prediction, tmp_path / "scores.json")


def test_raw_legacy_usage_is_normalized_when_request_evidence_is_unavailable():
    conversation = Conversation.from_openai(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {
                            "name": "T",
                            "arguments": '{"x":[1,3]}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c", "content": "2"},
        ]
    )
    raw = [{"usage": {"total_tokens": 5}, "attempts": 1}]
    assert score_prediction(task(), "2", conversation, raw)["scores"]["model_total_tokens"] == 5


@pytest.mark.parametrize("usage", [None, -1, True, "10"])
def test_invalid_usage_cannot_be_reported_as_complete(usage):
    assert not usage_summary([{"usage": {"total_tokens": usage}}])["usage_complete"]


def test_native_public_resource_retains_types_and_strips_nested_gold():
    resource = {
        "items": [{"source_id": 1, "expected_tools": ["PRIVATE"], "gold_answer": "PRIVATE"}]
    }
    assert public_input(replace(task(), input_resource=resource)).input_resource == {
        "items": [{"source_id": 1}],
    }
