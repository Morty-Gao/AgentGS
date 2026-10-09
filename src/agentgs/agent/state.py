"""Observable algorithm decisions and durable partial results."""

import asyncio
import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

from agentgs.agent.prompts import system_prompt, task_context, tool_feedback
from agentgs.client import audit
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import AgentResult
from agentgs.messages import Conversation, Message

logger = logging.getLogger(__name__)


class AgentExecutionError(RuntimeError):
    def __init__(self, message, result, status_code=None):
        self.result, self.status_code = result, status_code
        super().__init__(message)


class ModelProtocolError(ValueError):
    """A model response cannot advance the current stage."""


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
    action_budget_exhausted: bool = False
    answer_evidence_status: str = "unverified"

    def __post_init__(self):
        self._model_lock = asyncio.Lock()
        self._save_lock = asyncio.Lock()
        self.conversation = Conversation(
            messages=[m.model_copy(deep=True) for m in self.task.messages]
        )
        self._finished = False
        self._stage_request = None
        self._protocol_remaining = None
        self._token_charge = 0
        self._http_attempts = 0
        self.physical_tool_calls = 0
        self.diagnostic_tool_calls = 0
        options = getattr(self.context, "options", None) or AlgorithmSettings()
        self.context.options = options
        expected_system = system_prompt(options.planning_mode, self.task.dataset_instructions)
        systems = [m for m in self.conversation.messages if m.role == "system"]
        if any(m.content != expected_system for m in systems):
            raise ValueError("Imported system prompt differs from the framework system prompt")
        self.conversation.messages = [m for m in self.conversation.messages if m.role != "system"]
        self.conversation.messages.insert(
            0, systems[0] if systems else Message(role="system", content=expected_system)
        )
        if self.conversation.pending_tool_calls():
            raise ValueError("Cannot start a user task with unresolved historical tool calls")
        dialogue = [m for m in self.conversation.messages if m.role not in {"system", "event"}]
        if dialogue and dialogue[-1].role == "tool":
            raise ValueError("Unfinished tool turns cannot be restarted as a new user task")
        if dialogue and dialogue[-1].role == "user" and dialogue[-1].metadata.get("sent_to_model"):
            raise ValueError("An unfinished model request requires an explicit new attempt")
        last_user = next(
            (m for m in reversed(self.conversation.messages) if m.role == "user"), None
        )
        if self.task.question.strip() and (
            last_user is None
            or last_user.metadata.get("original_question", last_user.content) != self.task.question
            or any(
                m.role == "assistant"
                for m in self.conversation.messages[
                    self.conversation.messages.index(last_user) + 1 :
                ]
            )
        ):
            self.conversation.append(
                Message(role="user", content=self.task.question, metadata={"source": "task"})
            )
        self._current_user = next(
            (m for m in reversed(self.conversation.messages) if m.role == "user"), None
        )

    def execution_events(self):
        return self.conversation.execution_events()

    async def cancel(self):
        """Close native invocations even when cancellation interrupts binding."""
        for call in self.conversation.pending_tool_calls():
            self.conversation.finish_tool(
                call.id, json.dumps({"status": "cancelled"}), status="cancelled"
            )
        for row in self.usage:
            if row["status"] == "started":
                row["status"] = "cancelled"
        self.completion_status = "Not Completed"
        if self.answer_generation_status == "not_started":
            self.answer_generation_status = "cancelled"
        await self.emit("run.cancelled", {})

    async def record(self, event):
        event = dict(event)
        self.conversation.record_event(event.pop("type"), event)
        await self.save()

    async def emit(self, kind, payload):
        self.conversation.record_event(kind, payload)
        try:
            await invoke(self.context.events.emit, kind, payload)
        except Exception as exc:
            logger.warning("Auxiliary agent event could not be saved: %s", type(exc).__name__)
            await self.record(
                {
                    "type": "event_recording_error",
                    "event_type": kind,
                    "error_type": type(exc).__name__,
                }
            )
        await self.save()

    def result(self):
        options = getattr(self.context, "options", None)
        return AgentResult(
            answer=self.answer,
            tool_results=self.results,
            usage=self.usage,
            messages=self.conversation.messages,
            completion_status=self.completion_status,
            tool_chain_completed=self.tool_chain_completed,
            answer_generation_status=self.answer_generation_status,
            action_budget_exhausted=self.action_budget_exhausted,
            answer_evidence_status=self.answer_evidence_status,
            algorithm_id="geotoolagent-native-function-calling-v1",
            provenance={
                "conversation_schema": 2,
                "execution_cost": {
                    "tool_executor_requests": self.physical_tool_calls,
                    "diagnostic_tool_executor_requests": self.diagnostic_tool_calls,
                    "model_requests": self.model_calls,
                },
                "budget": {
                    "http_attempts": self._http_attempts,
                    "token_charge": self._token_charge,
                    "token_accounting": "reported_usage_or_reserved_upper_bound",
                },
                "effective_algorithm": options.model_dump(mode="json")
                if options is not None
                else {},
                **(getattr(self, "provenance_extra", None) or {}),
            },
        )

    async def save(self):
        if self.context.checkpoint is not None:
            async with self._save_lock:
                try:
                    snapshot = self.result().model_copy(deep=True)
                    await invoke(self.context.checkpoint, snapshot)
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
        await self.save()
        try:
            self.physical_tool_calls += 1
            result = await self.context.tools.execute(call)
        except asyncio.CancelledError:
            self.conversation.finish_tool(
                call.id, json.dumps({"status": "cancelled"}), status="cancelled"
            )
            await self.save()
            raise
        except Exception as exc:
            self.conversation.finish_tool(
                call.id, str(exc), status="failed", error_type=type(exc).__name__
            )
            await self.save()
            raise
        self.results.append(result)
        self.conversation.finish_tool(
            call.id,
            result.model_dump_json(),
            status="success" if result.succeeded else "failed",
        )
        await self.save()
        return result

    async def execute_many(self, calls):
        for call in calls:
            if call.id not in self.conversation.function_calls():
                self.conversation.begin_tool(
                    call.tool, json.dumps(call.arguments, ensure_ascii=False), call_id=call.id
                )
        await self.save()

        async def record_result(result):
            self.results.append(result)
            self.conversation.finish_tool(
                result.call_id,
                result.model_dump_json(),
                status="success" if result.succeeded else "failed",
            )
            await self.save()

        try:
            self.physical_tool_calls += len(calls)
            results = await self.context.tools.execute_many(calls, on_result=record_result)
        except (Exception, asyncio.CancelledError) as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            for call in calls:
                if any(pending.id == call.id for pending in self.conversation.pending_tool_calls()):
                    self.conversation.finish_tool(
                        call.id,
                        json.dumps({"status": status, "error": type(exc).__name__}),
                        status=status,
                    )
            await self.save()
            raise
        batch_ids = {call.id for call in calls}
        self.results = [
            result for result in self.results if result.call_id not in batch_ids
        ] + results
        await self.save()
        return results

    async def model(self, prompt, *, stage, tools=None, tool_choice=None):
        async with self._model_lock:
            return await self._model(
                prompt,
                stage=stage,
                tools=tools,
                tool_choice=tool_choice,
            )

    def begin_protocol_stage(self):
        self._protocol_remaining = self.context.options.argument_attempts

    def protocol_failure(self):
        if self._protocol_remaining is not None:
            self._protocol_remaining -= 1
            if self._protocol_remaining <= 0:
                raise ModelProtocolError("Protocol repair budget exhausted")

    async def _model(self, prompt, *, stage, tools=None, tool_choice=None):
        """Finalize unsent feedback, then request one legal stage response."""
        if self._finished:
            raise ModelProtocolError("The assistant has already ended this user turn")
        pending = set(self.conversation.function_calls()) - {
            m.tool_call_id for m in self.conversation.messages if m.role == "tool"
        }
        if pending:
            raise ValueError("Cannot request a model response before all tool calls are answered")
        inputs = [m for m in self.conversation.messages if m.role != "event"]
        if self._stage_request is None:
            if self._current_user is None:
                raise ModelProtocolError("A model task requires a user message")
            user = self._current_user
            user.metadata.update(original_question=self.task.question, stage=stage)
            user.content = task_context(
                self.task.question,
                prompt,
                input_resource=self.task.input_resource,
                output_contract=self.task.output_contract,
            )
        elif inputs[-1].role == "tool" and not inputs[-1].metadata.get("sent_to_model"):
            feedback = inputs[-1]
            raw = feedback.metadata.setdefault("raw_tool_output", deepcopy(feedback.content))
            feedback.content = tool_feedback(raw, prompt, stage)
            feedback.metadata.update(stage=stage, reminder=getattr(prompt, "instruction", prompt))
        elif self._stage_request != (stage, prompt):
            raise ModelProtocolError("A new stage requires a fresh tool result")
        self._stage_request = (stage, prompt)
        for message in inputs:
            if message.role == "tool" and not message.metadata.get("sent_to_model"):
                raw = message.metadata.setdefault("raw_tool_output", deepcopy(message.content))
                if "reminder" not in message.metadata:
                    message.content = tool_feedback(raw)
            message.metadata["sent_to_model"] = True
        wire = Conversation(messages=inputs).to_request_messages()
        if wire[0]["role"] != "system" or sum(m["role"] == "system" for m in wire) != 1:
            raise ModelProtocolError("Agent requests require exactly one leading system message")
        await self.save()

        async def invoke():
            return await self.context.client.chat(
                deepcopy(wire),
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,
                model=self.task.model,
                effort=self.task.effort,
                stage=stage,
                settings=self.context.options,
            )

        def validate(message):
            if message.role != "assistant":
                raise ModelProtocolError("A model response must have the assistant role")
            if tools and tool_choice != "auto" and not message.tool_calls:
                raise ModelProtocolError("This stage requires a native function call")
            if not tools and message.tool_calls:
                raise ModelProtocolError("Tools are unavailable during final answer generation")
            if not message.tool_calls and (
                not isinstance(message.content, str) or not message.content.strip()
            ):
                raise ModelProtocolError("The model returned no answer or function call")

        for attempt in range(self.context.options.argument_attempts):
            try:
                message = await self.call_model(
                    inputs, stage=stage, invoke=invoke, tools=tools, validate=validate
                )
                break
            except ModelProtocolError:
                self.protocol_failure()
                if attempt + 1 == self.context.options.argument_attempts:
                    raise
        if tools:
            if not message.tool_calls:
                self._finished = True
                message.metadata["final_answer"] = True
                await self.save()
            return message
        self._finished = True
        message.metadata["final_answer"] = True
        await self.save()
        return message.content

    async def call_model(self, inputs, *, stage, invoke, tools=None, validate=None):
        """Persist every request and original reply without dropping reasoning."""
        if self.model_calls >= (self.task.max_model_calls or self.context.options.max_model_calls):
            from agentgs.client.errors import BudgetExceeded

            raise BudgetExceeded("Explicit model call budget exhausted")
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
        try:
            await self.emit(
                "model.started",
                {
                    "index": self.model_calls,
                    "stage": stage,
                    "request_id": request_id,
                    "input_message_ids": [m.id for m in inputs],
                    "request_messages": Conversation(messages=inputs).to_request_messages(),
                    "model": self.task.model,
                    "effort": self.task.effort,
                    "tools": tools,
                },
            )

            async def record_transport(kind, payload):
                from agentgs.client.context import token_bound
                from agentgs.client.errors import BudgetExceeded

                if kind == "request":
                    request = payload["payload"]
                    reserve = token_bound(request) + request.get(
                        "max_tokens", request.get("max_completion_tokens", 0)
                    )
                    if self._token_charge + reserve > self.context.options.max_total_tokens:
                        raise BudgetExceeded("Task token budget exhausted")
                    self._token_charge += reserve
                    self._pending_token_reserve = reserve
                    self._http_attempts += 1
                elif kind == "response":
                    try:
                        usage = json.loads(payload["body"]).get("usage", {})
                        total = usage.get("total_tokens")
                        if isinstance(total, int) and total >= 0:
                            self._token_charge += total - self._pending_token_reserve
                    except (ValueError, AttributeError):
                        pass
                await self.emit(
                    "model.transport_" + kind,
                    {
                        "request_id": request_id,
                        "stage": stage,
                        **deepcopy(payload),
                    },
                )

            token = audit.recorder.set(record_transport)
            try:
                reply = await invoke()
            finally:
                audit.recorder.reset(token)
        except asyncio.CancelledError:
            row.update(status="cancelled")
            await self.emit(
                "model.cancelled",
                {"call": self.model_calls, "stage": stage, "request_id": request_id},
            )
            raise
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__)
            await self.emit(
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
        request_payload = getattr(reply, "request_payload", None)
        await self.emit(
            "model.response",
            {
                **row,
                "request_id": request_id,
                "request_payload": request_payload,
                "response": message.model_dump(mode="json"),
                "raw_response": getattr(reply, "raw_response", None),
            },
        )
        if validate is not None:
            try:
                validate(message)
            except ModelProtocolError as exc:
                row.update(status="protocol_failed", error=str(exc))
                await self.emit("model.protocol_failed", {**row, "request_id": request_id})
                raise
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
        await self.emit("model.completed", {**row, "request_id": request_id})
        return message
