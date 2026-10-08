"""Observable algorithm decisions and durable partial results."""

import hashlib
import logging
from dataclasses import dataclass, field

from agentgs.contracts.models import AgentResult

logger = logging.getLogger(__name__)


class AgentExecutionError(RuntimeError):
    def __init__(self, message, result, status_code=None):
        self.result, self.status_code = result, status_code
        super().__init__(message)


@dataclass
class RunState:
    context: object
    task: object
    trace: list = field(default_factory=list)
    results: list = field(default_factory=list)
    usage: list = field(default_factory=list)
    model_calls: int = 0
    answer: str = ""
    completion_status: str = "Not Completed"
    tool_chain_completed: bool = False
    answer_generation_status: str = "not_started"

    def emit(self, kind, payload):
        self.trace.append({"type": kind, **payload})
        try:
            self.context.events.emit(kind, payload)
        except Exception as exc:
            logger.warning("Auxiliary agent event could not be saved: %s", type(exc).__name__)
            self.trace.append(
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
            trace=self.trace,
            completion_status=self.completion_status,
            tool_chain_completed=self.tool_chain_completed,
            answer_generation_status=self.answer_generation_status,
            algorithm_id="geotoolagent-geotooleval-101-snapshot-20261008",
        )

    def save(self):
        if self.context.checkpoint is not None:
            try:
                self.context.checkpoint(self.result())
            except Exception as exc:
                logger.warning("Agent checkpoint could not be saved: %s", type(exc).__name__)
                if not any(row.get("type") == "checkpoint_error" for row in self.trace):
                    self.trace.append(
                        {"type": "checkpoint_error", "error_type": type(exc).__name__}
                    )

    async def model(self, prompt, *, stage, system_prompt=""):
        if self.task.max_model_calls is not None and self.model_calls >= self.task.max_model_calls:
            raise RuntimeError("Explicit model call budget exhausted")
        self.model_calls += 1
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
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            },
        )
        client = self.context.client
        try:
            if hasattr(client, "text"):
                reply = await client.text(
                    prompt,
                    stage=stage,
                    settings=self.context.options,
                    model=self.task.model,
                    effort=self.task.effort,
                    system_prompt=system_prompt,
                )
            else:
                reply = await client.chat(
                    [
                        {
                            "role": "system",
                            "content": system_prompt
                            or "You are ChatGPT, a large language model trained by OpenAI.\n",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=self.task.model,
                    effort=self.task.effort,
                )
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__)
            self.emit(
                "model.failed",
                {"call": self.model_calls, "stage": stage, "error_type": type(exc).__name__},
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
        self.emit("model.completed", row)
        answer = reply.message.get("content")
        if not isinstance(answer, str):
            raise ValueError("Staged model protocol requires textual content")
        return answer
