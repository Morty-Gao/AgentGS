import json
from types import SimpleNamespace

import httpx
import pytest

from agentgs.agent.state import RunState
from agentgs.client.models import ChatReply, LLMConfig
from agentgs.client.openai_compatible import OpenAICompatibleClient
from agentgs.contracts.models import TaskInput
from agentgs.messages import Conversation, Message

REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_details", "thinking")


def assistant(content, label, *, call_id=None):
    message = {
        "role": "assistant",
        "content": content,
        "reasoning": f"{label} reasoning",
        "reasoning_content": f"{label} reasoning content",
        "reasoning_details": [{"type": "encrypted", "data": label}],
        "thinking": {"text": f"{label} thinking", "signature": label},
    }
    if call_id:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "T", "arguments": '{"x":1}'},
            }
        ]
    return message


@pytest.fixture
def conversation():
    conversation = Conversation.from_openai(
        [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Earlier question"},
            assistant("Earlier answer", "old"),
            {"role": "user", "content": "Current question"},
            assistant(None, "current-a", call_id="a"),
            {"role": "tool", "tool_call_id": "a", "content": "first result"},
            {"role": "developer", "content": "Continue using the results."},
            assistant(None, "current-b", call_id="b"),
            {"role": "tool", "tool_call_id": "b", "content": "second result"},
        ]
    )
    conversation.record_event("tool.completed", {"turn": "user", "status": "success"})
    return conversation


def test_request_prunes_only_reasoning_before_latest_user(conversation):
    stored = conversation.model_dump_json()
    expected = conversation.to_openai()
    for key in REASONING_FIELDS:
        expected[2].pop(key)
    sent = conversation.to_request_messages()
    assert sent == expected
    for index in (4, 7):
        assert all(key in sent[index] for key in REASONING_FIELDS)
    assert conversation.model_dump_json() == stored
    assert conversation.to_openai()[2]["reasoning_content"] == "old reasoning content"


def test_serialization_retains_all_reasoning_after_request_preparation(conversation, tmp_path):
    complete = conversation.to_openai()
    conversation.to_request_messages()
    path = tmp_path / "messages.json"
    conversation.save(path)
    for restored in (
        Conversation.load(path),
        Conversation.model_validate_json(conversation.model_dump_json()),
        Conversation.from_jsonl(conversation.to_jsonl()),
    ):
        assert restored.to_openai() == complete
        assert restored.to_request_messages() == conversation.to_request_messages()
        for index in (2, 4, 7):
            assert all(key in restored.to_openai()[index] for key in REASONING_FIELDS)


def test_request_converts_content_and_calls_without_sending_internal_fields(conversation):
    conversation.messages[3].content = [
        {"type": "text", "text": "Current question"},
        {"type": "image_url", "image_url": {"url": "https://fixture/image.png"}},
    ]
    sent = conversation.to_request_messages()
    assert sent[3]["content"] == conversation.messages[3].content
    assert sent[4]["tool_calls"][0] == {
        "id": "a",
        "type": "function",
        "function": {"name": "T", "arguments": '{"x":1}'},
    }
    assert sent[5]["tool_call_id"] == "a"
    assert all(message["role"] != "event" for message in sent)
    assert all(
        not any(key in message for key in ("id", "created_at", "metadata", "provider_fields"))
        for message in sent
    )
    sent[3]["content"][0]["text"] = "changed request"
    assert conversation.messages[3].content[0]["text"] == "Current question"


@pytest.mark.parametrize("content", ["Next question", [{"type": "text", "text": "Next question"}]])
def test_new_user_turn_prunes_all_earlier_reasoning(conversation, content):
    conversation.append(Message(role="user", content=content))
    sent = conversation.to_request_messages()
    assert all(not any(key in message for key in REASONING_FIELDS) for message in sent)
    assert sent[-1] == {"role": "user", "content": content}
    # Preparing another turn does not delete the durable reasoning record.
    assert (
        conversation.messages[4].provider_fields["reasoning_content"]
        == "current-a reasoning content"
    )


def test_without_user_message_no_reasoning_is_pruned():
    conversation = Conversation.from_openai(
        [
            {"role": "developer", "content": "Continue"},
            assistant("Answer", "current"),
        ]
    )
    assert conversation.to_request_messages() == conversation.to_openai()
    assert Conversation().to_request_messages() == []


def test_request_reconstruction_uses_that_requests_user_turn(conversation):
    inputs = [message for message in conversation.messages if message.role != "event"]
    expected = conversation.to_request_messages()
    conversation.record_event(
        "model.started",
        {
            "request_id": "first",
            "input_message_ids": [message.id for message in inputs],
        },
    )
    conversation.append(Message(role="user", content="A later turn"))
    assert conversation.request_messages("first") == expected
    assert (
        conversation.request_messages("first")[4]["reasoning_content"]
        == "current-a reasoning content"
    )


@pytest.mark.parametrize("input_kind", ["conversation", "typed_messages", "openai_messages"])
async def test_transport_applies_turn_policy_and_retries_identical_payloads(
    conversation, input_kind
):
    config = LLMConfig.model_validate(
        {
            "providers": {
                "p": {"base_url": "https://fixture/v1", "api_key": "fixture", "max_attempts": 2}
            },
            "models": {
                "m": {
                    "provider": "p",
                    "model": "fixture",
                    "context_window": 128000,
                    "max_output_tokens": 100,
                }
            },
            "roles": {"planner": "m"},
        }
    )
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"choices": [{"message": assistant("Done", "reply")}]})

    client = OpenAICompatibleClient(config, transport=httpx.MockTransport(handler))
    source = {
        "conversation": conversation,
        "typed_messages": conversation.messages,
        "openai_messages": conversation.to_openai(),
    }[input_kind]
    stored = conversation.model_dump_json()
    try:
        reply = await client.chat(source)
    finally:
        await client.aclose()
    assert requests[0] == requests[1]
    assert requests[0]["messages"] == conversation.to_request_messages()
    assert reply.request_payload == requests[0]
    assert reply.message.provider_fields["reasoning_content"] == "reply reasoning content"
    assert conversation.model_dump_json() == stored


async def test_agent_preserves_reasoning_across_tools_then_prunes_on_next_user():
    requests = []

    class Client:
        async def chat(self, messages, **kwargs):
            assert all(isinstance(message, dict) for message in messages)
            requests.append(messages)
            if len(requests) == 1:
                return ChatReply(message=assistant(None, "current", call_id="call"))
            return ChatReply(message=assistant("Done", "final"))

    history = Conversation.from_openai(
        [
            {"role": "user", "content": "Earlier question"},
            assistant("Earlier answer", "old"),
            {"role": "user", "content": "Current question"},
        ]
    )
    context = SimpleNamespace(
        client=Client(),
        options=None,
        checkpoint=None,
        events=SimpleNamespace(emit=lambda *args: None),
    )
    state = RunState(context, TaskInput(messages=history.messages))
    call = await state.model("Use T", stage="tool_calling", tools=[{"type": "function"}])
    assert call.tool_calls[0].id == "call"
    state.conversation.finish_tool("call", "tool result", status="success")
    assert (
        await state.model("Answer using the tool result", stage="final_answer_generation") == "Done"
    )
    prior = next(message for message in requests[1] if message.get("content") == "Earlier answer")
    current = next(message for message in requests[1] if message.get("tool_calls"))
    assert not any(key in prior for key in REASONING_FIELDS)
    assert all(key in current for key in REASONING_FIELDS)
    state.conversation.append(Message(role="user", content="Next question"))
    await state.model("Answer the next question", stage="final_answer_generation")
    assert all(not any(key in message for key in REASONING_FIELDS) for message in requests[2])
    conversation = Conversation(messages=state.result().messages)
    request_ids = [
        event["request_id"]
        for event in conversation.execution_events()
        if event["type"] == "model.started"
    ]
    assert [conversation.request_messages(request_id) for request_id in request_ids] == requests
    assert (
        next(message for message in state.result().messages if message.tool_calls).provider_fields[
            "reasoning_content"
        ]
        == "current reasoning content"
    )
