"""Original result-classification policy, separated from transport and retries.

This classifies the legacy boundary representation. A typed ToolResult adapter
must preserve these distinctions before the policy is wired into the harness.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

INVALID_INPUT = re.compile(
    "^\\s*(?:input\\s+error|invalid\\s+input|parameter\\s+error)\\s*:", re.I
)
RUNTIME_ERROR = re.compile(
    "^\\s*(?:runtime\\s+error|execution\\s+error|tool\\s+error|error)\\s*:",
    re.I,
)


@dataclass(frozen=True)
class ExecutionFeedback:
    status: str
    score: float
    error_type: str
    summary: str
    suggestion: str
    retryable: bool

    def as_dict(self):
        return asdict(self)

    def to_event(self, *, tool_name: str, attempt: int, max_attempts: int) -> dict:
        return {
            "type": "execution_feedback",
            "tool_name": tool_name,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "status": self.status,
            "score": self.score,
            "error_type": self.error_type,
            "summary": self.summary,
            "suggestion": self.suggestion,
            "will_retry": self.retryable and attempt < max_attempts,
        }


def stringify(output):
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(output)


def error_envelope(text):
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("success") is False or str(payload.get("status", "")).casefold() in {
        "error",
        "failed",
        "failure",
    }:
        return True
    code = payload.get("status_code")
    if isinstance(code, int) and (not isinstance(code, bool)) and (code >= 400):
        return True
    error = payload.get("error")
    return bool(error) and isinstance(error, (str, dict, list))


def assess_tool_execution(output, *, input_format="", attempt=1, max_attempts=3):
    text = stringify(output).strip()
    can_retry = attempt < max_attempts

    def failure(kind, summary, suggestion):
        return ExecutionFeedback(
            "retrying" if can_retry else "failed", 0.0, kind, summary, suggestion, can_retry
        )

    shortened = text if len(text) <= 800 else text[:800] + "…"
    if not text:
        return failure(
            "empty_output",
            "The tool returned an empty result.",
            "Run the tool again; if this repeats, inspect the tool implementation and service logs.",
        )
    if text.lower().startswith("tool function execution error"):
        transport = any(
            (
                value in text.lower()
                for value in [
                    "http ",
                    "clientconnector",
                    "connection",
                    "timeout",
                    "non-json response",
                ]
            )
        )
        return failure(
            "transport_error" if transport else "tool_runtime_error",
            shortened,
            "Retry the same input after checking that the tool service on port 60002 is available."
            if transport
            else "Use the runtime message to correct the input or choose another tool.",
        )
    if INVALID_INPUT.search(text):
        expected = f" Expected input format: {input_format}." if input_format else ""
        return failure(
            "invalid_input",
            shortened,
            "Generate a corrected input from the user question and do not repeat the rejected input."
            + expected,
        )
    if (
        RUNTIME_ERROR.search(text)
        or text.lower().startswith("traceback (most recent call last)")
        or error_envelope(text)
    ):
        return failure(
            "tool_runtime_error",
            shortened,
            "Use this runtime error to repair the input; choose an alternative tool if repair is impossible.",
        )
    return ExecutionFeedback(
        "passed",
        1.0,
        "none",
        "Tool execution completed and returned a usable result.",
        "Continue to the next planned step.",
        False,
    )
