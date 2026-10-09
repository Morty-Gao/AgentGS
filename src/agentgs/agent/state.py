"""Observable algorithm decisions and durable partial results."""

import json
import logging
from dataclasses import dataclass, field
from uuid import uuid4

from agentgs.contracts.models import AgentResult
from agentgs.messages import Conversation, Message

logger = logging.getLogger(__name__)


class AgentExecutionError(RuntimeError):
    def __init__(self, message, result, status_code=None):
        self.result, self.status_code = result, status_code
        super().__init__(message)


@dataclass
class RunState:
    context: object
    task: object
    conversation: Conversation = field(default_factory=Conversation)
    results: list = field(default_factory=list)
    usage: list = field(default_factory=list)
    model_calls: int = 0
    answer: str = ""
    completion_status: str = "Not Completed"
    tool_chain_completed: bool = False
    answer_generation_status: str = "not_started"

    def __post_init__(self):
        self.conversation = Conversation(
            messages=[m.model_copy(deep=True) for m in self.task.messages]
        )
        last_user = next((m for m in reversed(self.conversation.messages) if m.role == "user"), None)
        if self.task.question.strip() and (last_user is None or last_user.content != self.task.question):
            self.conversation.append(
                Message(role="user", content=self.task.question, metadata={"source": "task"})
            )

    def execution_events(self):
        return self.conversation.execution_events()

    def record(self, event):
        event = dict(event)
        self.conversation.record_event(event.pop("type"), event)
        self.save()

    def emit(self, kind, payload):
        self.conversation.record_event(kind, payload)
        try:
            self.context.events.emit(kind, payload)
        except Exception as exc:
            logger.warning("Auxiliary agent event could not be saved: %s", type(exc).__name__)
            self.record(
                {
                    "type": "event_recording_error",
                    "event_type": kind,
                    "error_type": type(exc).__name__,
                }
            )
        self.save()

    def result(self):
        return AgentResult(
            answer=self.answer,
            tool_results=self.results,
            usage=self.usage,
            messages=self.conversation.messages,
            completion_status=self.completion_status,
            tool_chain_completed=self.tool_chain_completed,
            answer_generation_status=self.answer_generation_status,
            algorithm_id="geotoolagent-native-function-calling-v1",
        )

    def save(self):
        if self.context.checkpoint is not None:
            try:
                self.context.checkpoint(self.result())
            except Exception as exc:
                logger.warning("Agent checkpoint could not be saved: %s", type(exc).__name__)
                if not any(
                    row.get("type") == "checkpoint_error" for row in self.execution_events()
                ):
                    self.conversation.record_event(
                        "checkpoint_error", {"error_type": type(exc).__name__}
                    )

    async def execute(self, call):
        """Record explicit tools with the same function-call protocol as model tools."""
        if call.id not in self.conversation.function_calls():
            self.conversation.begin_tool(
                call.tool, json.dumps(call.arguments, ensure_ascii=False), call_id=call.id
            )
        self.save()
        try:
            result = await self.context.tools.execute(call)
        except Exception as exc:
            self.conversation.finish_tool(
                call.id, str(exc), status="failed", error_type=type(exc).__name__
            )
            self.save()
            raise
        self.results.append(result)
        self.conversation.finish_tool(
            call.id,
            result.model_dump_json(),
            status="success" if result.succeeded else "failed",
        )
        self.save()
        return result

    async def model(self, prompt, *, stage, system_prompt="", tools=None, tool_choice=None):
        """Continue the same user turn, retaining every original assistant reply.

        Stage instructions use developer messages. Current-turn reasoning and
        call IDs are retained; earlier-turn reasoning is omitted from requests.
        """
        pending = set(self.conversation.function_calls()) - {
            m.tool_call_id for m in self.conversation.messages if m.role == "tool"
        }
        if pending:
            raise ValueError("Cannot request a model response before all tool calls are answered")
        instruction = (system_prompt + "\n\n" if system_prompt else "") + prompt
        self.conversation.append(Message(role="developer", content=instruction, metadata={"stage": stage}))
        inputs = [m for m in self.conversation.messages if m.role != "event"]

        async def invoke():
            return await self.context.client.chat(
                Conversation(messages=inputs).to_request_messages(), tools=tools, tool_choice=tool_choice,
                parallel_tool_calls=False, model=self.task.model, effort=self.task.effort,
                stage=stage, settings=self.context.options,
            )

        message = await self.call_model(inputs, stage=stage, invoke=invoke, tools=tools)
        if tools:
            return message
        if message.tool_calls:
            for call in message.tool_calls:
                self.conversation.finish_tool(call.id, json.dumps({"status": "error", "error": "Tools are unavailable during final answer generation"}), status="failed")
            self.save()
            raise ValueError("Final-answer model returned an unexpected tool call")
        if not isinstance(message.content, str):
            raise ValueError("Final-answer model returned no textual content")
        return message.content

    async def call_model(self, inputs, *, stage, invoke, tools=None):
        """Persist every request and original reply without dropping reasoning."""
        if self.task.max_model_calls is not None and self.model_calls >= self.task.max_model_calls:
            raise RuntimeError("Explicit model call budget exhausted")
        self.model_calls += 1
        request_id = uuid4().hex
        row = {
            "call": self.model_calls,
            "stage": stage,
            "usage": None,
            "attempts": None,
            "response_model": None,
            "response_id": None,
            "status": "started",
        }
        self.usage.append(row)
        self.emit(
            "model.started",
            {
                "index": self.model_calls,
                "stage": stage,
                "request_id": request_id,
                "input_message_ids": [m.id for m in inputs],
                "model": self.task.model,
                "effort": self.task.effort,
                "tools": tools,
            },
        )
        try:
            reply = await invoke()
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__)
            self.emit(
                "model.failed",
                {
                    "call": self.model_calls,
                    "stage": stage,
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                },
            )
            raise
        row.update(
            {
                "usage": reply.usage,
                "attempts": reply.attempts,
                "response_model": getattr(reply, "response_model", None),
                "response_id": getattr(reply, "response_id", None),
                "status": "completed",
            }
        )
        message = reply.message.model_copy(deep=True)
        message.metadata.update(
            **row,
            request_id=request_id,
            raw_response=getattr(reply, "raw_response", None),
            request_options={
                k: v
                for k, v in (getattr(reply, "request_payload", None) or {}).items()
                if k != "messages"
            },
        )
        self.conversation.append(message)
        self.emit("model.completed", {**row, "request_id": request_id})
        return message
