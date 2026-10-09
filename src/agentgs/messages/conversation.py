"""One durable message log for dialogue, model requests, tools and annotations."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentgs.messages.models import (
    Function,
    FunctionCall,
    LogEntry,
    Message,
    RuntimeEvent,
    parse_entry,
)


class Conversation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2] = 2
    messages: list[LogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_links(self):
        seen_messages = set()
        calls = set()
        results = set()
        for message in self.messages:
            if message.id in seen_messages:
                raise ValueError(f"Duplicate message id: {message.id}")
            seen_messages.add(message.id)
            if isinstance(message, RuntimeEvent):
                continue
            for call in message.tool_calls:
                if call.id in calls:
                    raise ValueError(f"Duplicate function call id: {call.id}")
                calls.add(call.id)
            if message.role == "tool":
                if message.tool_call_id not in calls:
                    raise ValueError(f"Unknown tool_call_id: {message.tool_call_id}")
                if message.tool_call_id in results:
                    raise ValueError(f"Duplicate tool result: {message.tool_call_id}")
                results.add(message.tool_call_id)
        return self

    def append(self, message: LogEntry | dict[str, Any]) -> LogEntry:
        row = parse_entry(message) if isinstance(message, dict) else message
        if any(existing.id == row.id for existing in self.messages):
            raise ValueError(f"Duplicate message id: {row.id}")
        if isinstance(row, RuntimeEvent):
            self.messages.append(row)
            return row
        if row.role == "tool":
            calls = self.function_calls()
            if row.tool_call_id not in calls:
                raise ValueError(f"Unknown tool_call_id: {row.tool_call_id}")
            if any(m.role == "tool" and m.tool_call_id == row.tool_call_id for m in self.messages):
                raise ValueError(f"Duplicate tool result: {row.tool_call_id}")
        if row.tool_calls:
            ids = [call.id for call in row.tool_calls]
            if len(set(ids)) != len(ids) or set(ids) & self.function_calls().keys():
                raise ValueError("Duplicate function call id")
        self.messages.append(row)
        return row

    @classmethod
    def from_openai(cls, messages: Iterable[dict[str, Any]]) -> "Conversation":
        conversation = cls()
        for message in messages:
            conversation.append(Message.from_openai(message))
        return conversation

    def to_openai(self, *, include_reasoning: bool = True) -> list[dict[str, Any]]:
        """Convert the stored log without applying request-time turn pruning."""
        return [
            m.to_openai(include_reasoning=include_reasoning)
            for m in self.messages
            if m.role != "event"
        ]

    def to_request_messages(self) -> list[dict[str, Any]]:
        """Strip reasoning from previous user turns and retain the current turn.

        Only a user message starts a new turn. Tool calls, tool results,
        and internal events continue the existing turn.
        With no user message there is no cutoff, so all reasoning is retained.
        The durable log is never modified by preparing a request.
        """
        last_user = next(
            (
                index
                for index in range(len(self.messages) - 1, -1, -1)
                if self.messages[index].role == "user"
            ),
            -1,
        )
        return [
            message.to_openai(include_reasoning=index >= last_user)
            for index, message in enumerate(self.messages)
            if message.role != "event"
        ]

    def record_event(self, kind: str, payload: dict[str, Any]) -> RuntimeEvent:
        return self.append(RuntimeEvent(metadata={"type": kind, **payload}))

    def function_calls(self) -> dict[str, FunctionCall]:
        return {
            call.id: call for m in self.messages if isinstance(m, Message) for call in m.tool_calls
        }

    def pending_tool_calls(self) -> list[FunctionCall]:
        completed = {m.tool_call_id for m in self.messages if m.role == "tool"}
        return [call for call in self.function_calls().values() if call.id not in completed]

    def begin_tool(self, name: str, arguments: str, *, call_id: str | None = None) -> str:
        call_id = call_id or uuid4().hex
        self.append(
            Message(
                role="assistant",
                tool_calls=[
                    FunctionCall(id=call_id, function=Function(name=name, arguments=arguments))
                ],
                metadata={"source": "runtime"},
            )
        )
        return call_id

    def finish_tool(self, call_id: str, output: str, **metadata: Any) -> Message:
        return self.append(
            Message(role="tool", tool_call_id=call_id, content=output, metadata=metadata)
        )

    def execution_events(self) -> list[dict[str, Any]]:
        """Derived execution view. Changing it cannot change the message log."""
        calls = self.function_calls()
        events = []
        for message in self.messages:
            if message.role == "event":
                events.append(deepcopy(message.metadata))
            elif message.role == "tool" and message.tool_call_id in calls:
                call = calls[message.tool_call_id]
                status = message.metadata.get("status", "success")
                events.append(
                    {
                        "type": "planning_call"
                        if message.metadata.get("operation") == "planning"
                        else "tool_call"
                        if status == "success"
                        else "tool_retry",
                        "tool_name": call.function.name,
                        "tool_input": call.function.arguments,
                        "tool_output": deepcopy(
                            message.metadata.get("raw_tool_output", message.content)
                        ),
                        "call_id": call.id,
                        "attempt": message.metadata.get("attempt", 1),
                        "status": status,
                        **deepcopy(message.metadata),
                    }
                )
        return events

    def request_messages(self, request_id: str) -> list[dict[str, Any]]:
        """Reconstruct the exact input messages for a recorded model request."""
        for entry in reversed(self.messages):
            if entry.role == "event" and entry.metadata.get("request_id") == request_id and entry.metadata.get("type") == "model.transport_request":
                return deepcopy(entry.metadata["payload"]["messages"])
        request = next(
            (
                m
                for m in self.messages
                if m.role == "event"
                and m.metadata.get("request_id") == request_id
                and m.metadata.get("type") == "model.started"
            ),
            None,
        )
        if request is None:
            raise KeyError(f"Unknown model request: {request_id}")
        if "request_messages" in request.metadata:
            return deepcopy(request.metadata["request_messages"])
        by_id = {m.id: m for m in self.messages}
        inputs = [by_id[mid] for mid in request.metadata["input_message_ids"]]
        return Conversation(messages=inputs).to_request_messages()

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> "Conversation":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_jsonl(self) -> str:
        return "".join(m.model_dump_json() + "\n" for m in self.messages)

    @classmethod
    def from_jsonl(cls, text: str) -> "Conversation":
        return cls(
            messages=[parse_entry(json.loads(line)) for line in text.splitlines() if line.strip()]
        )
