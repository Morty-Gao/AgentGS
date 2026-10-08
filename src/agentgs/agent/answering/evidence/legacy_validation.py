from __future__ import annotations

import json
import math
import re
from typing import Any

from agentgs.agent.answering.evidence.comparison import (
    _change_summary_option,
    _directional_comparison_option,
    _ordered_comparison_option,
    _semantic_option,
)
from agentgs.agent.answering.evidence.composition import structured_evidence_answer
from agentgs.agent.answering.evidence.core import (
    _last_structured_output,
    _numbers,
    evidence_numbers,
)
from agentgs.agent.answering.evidence.options import _best_option, _option_blocks
from agentgs.agent.answering.evidence.temporal import _temporal_option, _temporal_series_option


def validate_final_answer(
    question: str,
    final_answer: str,
    successful_outputs: list[Any] | tuple[Any, ...],
    *,
    completed: bool = True,
) -> tuple[str, dict[str, Any]]:
    evidence = evidence_numbers(successful_outputs)
    answer = str(final_answer or "").strip()
    event: dict[str, Any] = {
        "type": "final_answer_validation",
        "evidence_number_count": len(evidence),
        "evidence_numbers": evidence[:40],
        "status": "passed",
    }
    if not completed:
        event["status"] = "failed_propagated"
        return (answer or "Task execution did not complete.", event)
    structured_answer = structured_evidence_answer(successful_outputs, question)
    option_evidence = list(evidence)
    if structured_answer:
        option_evidence.extend(_numbers(str(structured_answer)))
    option = (
        _temporal_option(question, successful_outputs)
        or _temporal_series_option(question, successful_outputs)
        or _ordered_comparison_option(question, successful_outputs)
        or _change_summary_option(question, successful_outputs)
        or _directional_comparison_option(question, successful_outputs)
        or _semantic_option(question, successful_outputs)
        or _best_option(question, option_evidence)
    )
    if option:
        label, body = option
        selected = re.search("(?:^|\\b)([A-H])(?:\\b|[\\.、:：])", answer, re.I)
        if not selected or selected.group(1).upper() != label:
            answer = f"{label}. {body}"
            event.update({"status": "corrected_choice", "selected_option": label})
        return (answer, event)
    if _option_blocks(question):
        event["status"] = "no_supported_option"
        terminal = _last_structured_output(successful_outputs)
        evidence_text = (
            json.dumps(terminal, ensure_ascii=False, default=str)
            if terminal is not None
            else "No completed tool result."
        )
        return (
            "No unique option is supported by the executed tool evidence. Observed results: "
            + evidence_text,
            event,
        )
    if structured_answer and answer == str(structured_answer).strip():
        answer = structured_answer
        event.update(
            {"status": "structured_evidence_projection", "structured_answer": structured_answer}
        )
        return (answer, event)
    answer_numbers = _numbers(answer)
    if answer_numbers and evidence:
        unmatched = [
            number
            for number in answer_numbers
            if not any(
                (
                    math.isclose(number, observed, rel_tol=0.02, abs_tol=0.02)
                    for observed in evidence
                )
            )
        ]
        if unmatched and len(unmatched) == len(answer_numbers):
            try:
                value = (
                    json.loads(successful_outputs[-1])
                    if isinstance(successful_outputs[-1], str)
                    else successful_outputs[-1]
                )
                answer = json.dumps(value, ensure_ascii=False, default=str)
                event["status"] = "fallback_to_tool_evidence"
                return (answer, event)
            except Exception:
                pass
    if structured_answer:
        answer = structured_answer
        event.update(
            {"status": "structured_evidence_projection", "structured_answer": structured_answer}
        )
        return (answer, event)
    return (answer, event)
