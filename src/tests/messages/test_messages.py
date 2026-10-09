import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentgs.agent.state import RunState
from agentgs.client.models import ChatReply
from agentgs.contracts.models import AgentResult, TaskInput
from agentgs.messages import Conversation, Message


def test_native_call_index_is_accepted_but_not_sent_back():
    message = Message.from_openai(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "index": 0,
                    "type": "function",
                    "function": {"name": "Mean", "arguments": '{"x":[1,3]}'},
                }
            ],
        }
    )
    assert message.tool_calls[0].index == 0
    assert "index" not in message.to_openai()["tool_calls"][0]
    assert Message.model_validate_json(message.model_dump_json()) == message


def test_multiturn_parallel_functions_reasoning_and_serialization(tmp_path):
    original = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Compare two values."},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Two independent calculations are needed.",
            "reasoning_details": [{"type": "encrypted", "data": "opaque"}],
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "T", "arguments": '{"x":1}'}},
                {"id": "b", "type": "function", "function": {"name": "T", "arguments": '{"x":2}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "b", "content": "2"},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "assistant", "content": "The second is larger."},
        {"role": "user", "content": "By how much?"},
    ]
    conversation = Conversation.from_openai(original)
    assert conversation.messages[2].reasoning == original[2]["reasoning_content"]
    assert conversation.to_openai(include_reasoning=True) == original
    exported = conversation.to_openai(include_reasoning=False)
    assert "reasoning_content" not in exported[2] and "reasoning_details" not in exported[2]
    conversation.record_event("validation", {"valid": True})
    assert len(conversation.to_openai()) == len(original)
    conversation.save(tmp_path / "messages.json")
    assert Conversation.load(tmp_path / "messages.json") == conversation
    assert Conversation.from_jsonl(conversation.to_jsonl()) == conversation
    assert [
        event["tool_input"]
        for event in conversation.execution_events()
        if event["type"] == "tool_call"
    ] == ['{"x":2}', '{"x":1}']
    assert "trace" not in AgentResult(answer="ok", messages=conversation.messages).model_dump()


def test_multimodal_and_malformed_function_arguments_are_preserved():
    parts = [
        {"type": "text", "text": "Describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    assert Message.from_openai({"role": "user", "content": parts}).to_openai()["content"] == parts
    call = {
        "role": "assistant",
        "tool_calls": [
            {"id": "bad", "type": "function", "function": {"name": "T", "arguments": "{broken"}}
        ],
    }
    assert Message.from_openai(call).tool_calls[0].function.arguments == "{broken"
    with pytest.raises(ValidationError):
        Message(role="tool", content="orphan")
    with pytest.raises(ValidationError):
        AgentResult(answer="ok", trace=[])


def test_exports_and_derived_events_do_not_mutate_stored_messages():
    original = {"role": "user", "content": [{"type": "text", "text": "original"}]}
    message = Message.from_openai(original)
    original["content"][0]["text"] = "changed at source"
    exported = message.to_openai()
    exported["content"][0]["text"] = "changed at destination"
    assert message.content[0]["text"] == "original"
    conversation = Conversation()
    conversation.record_event("validation", {"details": {"valid": True}})
    conversation.execution_events()[0]["details"]["valid"] = False
    assert conversation.messages[0].metadata["details"]["valid"] is True


def test_function_links_reject_orphans_duplicates_and_allow_pending_snapshots():
    conversation = Conversation()
    call_id = conversation.begin_tool("T", "{}")
    # A partial result may legitimately contain a call whose worker is still running.
    assert Conversation.model_validate_json(conversation.model_dump_json()) == conversation
    conversation.finish_tool(call_id, "ok")
    with pytest.raises(ValueError, match="Duplicate tool result"):
        conversation.finish_tool(call_id, "again")
    with pytest.raises(ValueError, match="Unknown tool_call_id"):
        conversation.finish_tool("missing", "result")
    with pytest.raises(ValueError, match="Duplicate function call"):
        conversation.begin_tool("T", "{}", call_id=call_id)
    with pytest.raises(ValidationError, match="Unknown tool_call_id"):
        Conversation(messages=[Message(role="tool", tool_call_id="missing", content="orphan")])


class Events:
    def emit(self, kind, payload):
        pass


async def test_staged_requests_reconstruct_exact_inputs_and_preserve_raw_reasoning():
    requests = []
    checkpoints = []

    class Client:
        async def chat(self, messages, **kwargs):
            requests.append(messages)
            return ChatReply(
                message={
                    "role": "assistant",
                    "content": "Plan: T",
                    "reasoning_content": "Check the evidence.",
                },
                raw_response={"choices": [{"message": {"content": "Plan: T"}}]},
                request_payload={"model": "fixture", "messages": messages, "temperature": 0.2},
                usage={"total_tokens": 10},
                response_id="response-1",
            )

    history = Conversation.from_openai(
        [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Follow-up"},
        ]
    )
    task = TaskInput(messages=history.messages)
    assert task.question == "Follow-up"
    context = SimpleNamespace(
        client=Client(),
        options=None,
        events=Events(),
        checkpoint=lambda result: checkpoints.append(result.model_dump_json()),
    )
    state = RunState(context, task)
    assert await state.model("Stage prompt", stage="tool_planning") == "Plan: T"
    result = AgentResult.model_validate_json(checkpoints[-1])
    conversation = Conversation(messages=result.messages)
    started = next(
        event for event in conversation.execution_events() if event["type"] == "model.started"
    )
    assert conversation.request_messages(started["request_id"]) == requests[0]
    contents = [row["content"] for row in requests[0]]
    assert "Earlier question" in contents and "Earlier answer" in contents
    assert "Stage prompt" in requests[0][-1]["content"]
    reply = next(
        m for m in result.messages if m.role == "assistant" and m.metadata.get("request_id")
    )
    assert reply.reasoning == "Check the evidence."
    assert reply.metadata["request_options"] == {"model": "fixture", "temperature": 0.2}
    assert reply.metadata["response_id"] == "response-1"
    assert reply.metadata["raw_response"]["choices"][0]["message"]["content"] == "Plan: T"
    assert "trace" not in json.loads(checkpoints[-1])


async def test_failed_model_request_is_durable_without_an_invented_reply():
    checkpoints = []

    class Client:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("offline")

    state = RunState(
        SimpleNamespace(
            client=Client(),
            options=None,
            events=Events(),
            checkpoint=lambda r: checkpoints.append(r.model_dump_json()),
        ),
        TaskInput(question="q"),
    )
    with pytest.raises(RuntimeError, match="offline"):
        await state.model("Full prompt", stage="planning")
    conversation = Conversation(messages=AgentResult.model_validate_json(checkpoints[-1]).messages)
    assert any(m.role == "user" and "Full prompt" in m.content for m in conversation.messages)
    assert not any(m.role == "assistant" for m in conversation.messages)
    assert conversation.execution_events()[-1]["type"] == "model.failed"


async def test_tool_retries_and_checkpoints_use_function_messages():
    from agentgs.agent.execution.feedback import run_tool_with_feedback

    conversation = Conversation()
    checkpoints = []

    async def provide(attempt, history):
        return json.dumps({"attempt": attempt})

    async def run(name, arguments, files):
        return (
            "Tool function execution error, retry"
            if json.loads(arguments)["attempt"] == 1
            else "42"
        )

    result = await run_tool_with_feedback(
        tool_name="T",
        file_path_list=[],
        input_provider=provide,
        tool_runner=run,
        messages=conversation,
        checkpoint=lambda: checkpoints.append(conversation.model_dump_json()),
    )
    assert result.succeeded
    tools = [m for m in conversation.messages if m.role == "tool"]
    assert [m.metadata["status"] for m in tools] == ["retrying", "success"]
    assert tools[0].tool_call_id != tools[1].tool_call_id
    assert (
        len(
            [
                m
                for m in Conversation.model_validate_json(checkpoints[0]).messages
                if m.role == "assistant" and m.tool_calls
            ]
        )
        == 1
    )
    assert not any(
        m.role == "tool" for m in Conversation.model_validate_json(checkpoints[0]).messages
    )
    assert [
        event["type"]
        for event in conversation.execution_events()
        if event["type"] in {"tool_call", "tool_retry"}
    ] == ["tool_retry", "tool_call"]


async def test_dialogue_continues_history_with_parallel_calls_and_reasoning():
    from agentgs.contracts.models import ToolCall, ToolResult

    requests = []
    snapshots = []
    history = Conversation.from_openai(
        [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
    )

    class Client:
        async def chat(self, messages, **kwargs):
            requests.append(messages)
            if len(requests) == 1:
                return ChatReply(
                    message={
                        "role": "assistant",
                        "reasoning_content": "Calculate both.",
                        "tool_calls": [
                            {
                                "id": key,
                                "type": "function",
                                "function": {"name": "T", "arguments": json.dumps({"x": x})},
                            }
                            for key, x in [("a", 1), ("b", 2)]
                        ],
                    }
                )
            assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["a", "b"]
            call_message = next(m for m in messages if m.get("tool_calls"))
            assert call_message["reasoning_content"] == "Calculate both."
            return ChatReply(
                message={"content": "The difference is one.", "thinking": "Compare results."}
            )

    async def execute(call):
        return ToolResult(call_id=call.id, tool=call.tool, succeeded=True, data=call.arguments)

    context = SimpleNamespace(
        client=Client(),
        tools=SimpleNamespace(execute=execute),
        events=Events(),
        checkpoint=lambda result: snapshots.append(result.model_dump_json()),
    )
    state = RunState(context, TaskInput(question="Compare", messages=history.messages))
    for _ in range(2):
        inputs = [m for m in state.conversation.messages if m.role != "event"]

        async def invoke():
            return await context.client.chat([m.to_openai() for m in inputs])

        message = await state.call_model(inputs, stage="tool_calling", invoke=invoke)
        for call in message.tool_calls:
            await state.execute(
                ToolCall(
                    id=call.id,
                    tool=call.function.name,
                    arguments=json.loads(call.function.arguments),
                )
            )
        if not message.tool_calls:
            state.answer = message.content
            await state.save()
    result = state.result()
    assert [m["content"] for m in requests[0] if m["role"] != "system"] == [
        "Previous question",
        "Previous answer",
        "Compare",
    ]
    assert result.answer == "The difference is one."
    persisted = AgentResult.model_validate_json(snapshots[-1])
    assert [
        m.reasoning for m in persisted.messages if m.role == "assistant" and m.reasoning is not None
    ] == [
        "Calculate both.",
        "Compare results.",
    ]
    assert [
        event["tool_name"] for event in persisted.execution_events() if event["type"] == "tool_call"
    ] == ["T", "T"]
    assert "trace" not in json.loads(snapshots[-1])


def test_benchmark_derives_successes_and_retries_from_messages():
    from agentgs.evaluation.metrics.tools import score_task

    conversation = Conversation()
    failed = conversation.begin_tool("T", '{"x":1}')
    conversation.finish_tool(failed, "failed", status="retrying")
    succeeded = conversation.begin_tool("T", '{"x":1}')
    conversation.finish_tool(succeeded, "1", status="success")
    scored = score_task(
        gold_steps=[{"tool_name": "T", "arguments": {"x": 1}}],
        messages=conversation.messages,
        answer_score=1,
        token_usage={"usage_complete": True, "total_tokens": 10},
    )
    assert scored["tsf1"] == 1
    assert scored["physical_tool_call_attempts"] == 2
    assert scored["ecr"] == 0.5
    assert scored["tce"] == 0.5


async def test_checkpoint_error_does_not_break_tool_message_pairing():
    from agentgs.agent.execution.feedback import run_tool_with_feedback
    from agentgs.agent.execution.service import ExecutionService
    from agentgs.contracts.models import ToolResult, ToolSpec

    async def execute(call):
        return ToolResult(call_id=call.id, tool=call.tool, succeeded=True, data={"value": 42})

    def fail_checkpoint(result):
        raise OSError("unavailable")

    async def provide(attempt, history):
        return "{}"

    state = RunState(
        SimpleNamespace(
            tools=SimpleNamespace(
                execute=execute,
                specs=lambda: [
                    ToolSpec(id="T", description="fixture", input_schema={"type": "object"})
                ],
            ),
            events=Events(),
            checkpoint=fail_checkpoint,
        ),
        TaskInput(question="q"),
    )
    result = await run_tool_with_feedback(
        tool_name="T",
        file_path_list=[],
        input_provider=provide,
        tool_runner=ExecutionService(state, None).physical,
        messages=state.conversation,
        checkpoint=state.save,
    )
    assert result.succeeded
    assert not state.conversation.pending_tool_calls()
    assert state.results[0].call_id == next(
        m.tool_call_id for m in state.conversation.messages if m.role == "tool"
    )
