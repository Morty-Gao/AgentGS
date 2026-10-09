"""Deterministic GeoToolEval accuracy and token-cost scoring."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any

from agentgs.evaluation.analysis.conversation import (
    as_conversation,
    conversation_usage,
    tool_executions,
)
from agentgs.messages import Conversation, Message

SCORING_PROTOCOL = "geotooleval-conversation-v3"

DEFAULT_TOKEN_BUDGET_PER_GOLD_STEP = 10000
_IGNORED_ARGUMENT_KEYS = {"_output_keys"}


def parse_tool_input(raw_input: Any) -> Any:
    if isinstance(raw_input, (dict, list, int, float, bool)) or raw_input is None:
        return raw_input
    text = str(raw_input).strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _clean_arguments(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _clean_arguments(item)
            for key, item in value.items()
            if str(key) not in _IGNORED_ARGUMENT_KEYS and (not str(key).startswith("_"))
        }
    if isinstance(value, list):
        return [_clean_arguments(item) for item in value]
    return value


def _normalized_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).strip().replace("\\", "/")
    text = re.sub("\\s+", " ", text).casefold()
    for anchor in ("geotoolagent-dataset/", "benchmark/", "assets/", "outputs/geotool_runtime/"):
        index = text.find(anchor)
        if index >= 0:
            return text[index:]
    return text


def value_similarity(expected: Any, actual: Any) -> float:
    expected = _clean_arguments(expected)
    actual = _clean_arguments(actual)
    if expected is None or actual is None:
        return 1.0 if expected is actual else 0.0
    if isinstance(expected, bool) or isinstance(actual, bool):
        return 1.0 if expected is actual else 0.0
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return (
            1.0
            if math.isclose(float(expected), float(actual), rel_tol=1e-05, abs_tol=1e-08)
            else 0.0
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = set(expected) | set(actual)
        if not keys:
            return 1.0
        matched = sum(
            (value_similarity(expected[key], actual[key]) for key in set(expected) & set(actual))
        )
        return 2.0 * matched / (len(expected) + len(actual))
    if isinstance(expected, list) and isinstance(actual, list):
        if not expected and (not actual):
            return 1.0
        remaining = list(actual)
        matched = 0.0
        for expected_item in expected:
            if not remaining:
                break
            similarities = [value_similarity(expected_item, item) for item in remaining]
            best_index = max(range(len(similarities)), key=similarities.__getitem__)
            matched += similarities[best_index]
            remaining.pop(best_index)
        return 2.0 * matched / (len(expected) + len(actual))
    expected_text = _normalized_text(str(expected))
    actual_text = _normalized_text(str(actual))
    if expected_text == actual_text:
        return 1.0
    if "/" in expected_text and "/" in actual_text:
        return 1.0 if expected_text.rsplit("/", 1)[-1] == actual_text.rsplit("/", 1)[-1] else 0.0
    return 0.0


def _lcs_alignment(expected: list[str], actual: list[str]) -> list[tuple[int, int]]:
    rows = len(expected) + 1
    cols = len(actual) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if expected[i - 1] == actual[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    alignment: list[tuple[int, int]] = []
    i, j = (len(expected), len(actual))
    while i > 0 and j > 0:
        if expected[i - 1] == actual[j - 1]:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(alignment))


def _parse_output_dict(raw_output: Any) -> dict[str, Any]:
    if isinstance(raw_output, dict):
        parsed = raw_output
    else:
        try:
            parsed = json.loads(str(raw_output or ""))
        except json.JSONDecodeError:
            return {}
    if not isinstance(parsed, dict):
        return {}
    # Explicit TaskInput.calls return a serialized ToolResult; model-driven
    # execution returns its raw domain payload. Both represent the same evidence.
    if {"call_id", "tool", "succeeded", "data"} <= parsed.keys():
        return parsed["data"] if isinstance(parsed["data"], dict) else {}
    return parsed


def _effective_expected_arguments(
    gold_step: dict[str, Any], actual_arguments: Any, prior_calls: list[dict[str, Any]]
) -> tuple[Any, Any]:
    expected = _clean_arguments(gold_step.get("arguments") or {})
    actual = _clean_arguments(actual_arguments)
    if not isinstance(actual, dict):
        if isinstance(expected, dict) and len(expected) == 1:
            only_key = next(iter(expected))
            actual = {only_key: actual}
        else:
            return (expected, actual)
    prior_outputs: dict[str, Any] = {}
    prior_argument_keys: set[str] = set()
    for call in prior_calls:
        prior_outputs.update(_parse_output_dict(call.get("tool_output")))
        prior_arguments = parse_tool_input(call.get("tool_input"))
        if isinstance(prior_arguments, dict):
            prior_argument_keys.update((str(key) for key in prior_arguments))
    dynamic_keys: list[str] = []
    for consumed in gold_step.get("consumes") or []:
        key = str(consumed)
        if key in actual and key in prior_outputs:
            expected[key] = prior_outputs[key]
            dynamic_keys.append(key)
    if dynamic_keys:
        for key in list(expected):
            if key not in actual and key in prior_argument_keys:
                expected.pop(key)
    return (expected, actual)


def _tool_selection_scores(
    gold_tools: list[str], actual_tools: list[str]
) -> tuple[float, float, float, int]:
    gold_counts = Counter(gold_tools)
    actual_counts = Counter(actual_tools)
    matched = sum((min(count, actual_counts[name]) for name, count in gold_counts.items()))
    precision = matched / len(actual_tools) if actual_tools else 0.0
    recall = matched / len(gold_tools) if gold_tools else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (precision, recall, f1, matched)


def score_task(
    *,
    gold_steps: list[dict[str, Any]],
    messages: Conversation | dict[str, Any] | list[Message | dict[str, Any]],
    answer_score: float,
    token_usage: dict[str, Any] | list[dict[str, Any]] | None = None,
    token_budget_per_gold_step: int = DEFAULT_TOKEN_BUDGET_PER_GOLD_STEP,
) -> dict[str, Any]:
    """Score one task on the 0-100 accuracy, token-cost and final scales."""
    gold_steps = list(gold_steps or [])
    gold_tools = [str(step.get("tool_name") or "") for step in gold_steps]
    conversation = as_conversation(messages)
    attempts = tool_executions(conversation)
    successful = [event for event in attempts if event.get("status") == "success"]
    actual_tools = [str(event.get("tool_name") or "") for event in successful]
    precision, recall, tsf1, matched_success = _tool_selection_scores(gold_tools, actual_tools)
    alignment = _lcs_alignment(gold_tools, actual_tools)
    tos = len(alignment) / len(gold_tools) if gold_tools else 0.0
    parameter_scores = [0.0] * len(gold_steps)
    for gold_index, actual_index in alignment:
        actual_arguments = parse_tool_input(successful[actual_index].get("tool_input"))
        expected, actual = _effective_expected_arguments(
            gold_steps[gold_index], actual_arguments, successful[:actual_index]
        )
        parameter_scores[gold_index] = value_similarity(expected, actual)
    pas = sum(parameter_scores) / len(gold_steps) if gold_steps else 0.0
    ecr = matched_success / len(attempts) if attempts else 0.0
    fas = min(1.0, max(0.0, float(answer_score)))
    accuracy_score = 100.0 * (0.4 * fas + 0.2 * tsf1 + 0.1 * tos + 0.2 * pas + 0.1 * ecr)
    task_success = (
        math.isclose(fas, 1.0)
        and math.isclose(tsf1, 1.0)
        and math.isclose(tos, 1.0)
        and math.isclose(pas, 1.0)
        and (matched_success == len(gold_steps))
        and not conversation.pending_tool_calls()
    )
    token_usage = conversation_usage(conversation, token_usage)
    usage_complete = token_usage.get("usage_complete") is True
    total_tokens = int(token_usage.get("total_tokens") or 0)
    token_budget = max(1, int(token_budget_per_gold_step)) * max(1, len(gold_steps))
    mte: float | None
    cost_score: float | None
    final_score: float | None
    if not usage_complete:
        mte = None
        cost_score = None
        final_score = None
    else:
        mte = 0.0 if not task_success else min(1.0, token_budget / max(total_tokens, 1))
        tce_for_cost = (
            0.0 if not task_success else min(1.0, len(gold_steps) / max(len(attempts), 1))
        )
        cost_score = 100.0 * (0.6 * mte + 0.4 * tce_for_cost)
        final_score = 0.85 * accuracy_score + 0.15 * cost_score
    tce = 0.0 if not task_success else min(1.0, len(gold_steps) / max(len(attempts), 1))
    return {
        "protocol": SCORING_PROTOCOL,
        "fas": round(fas, 6),
        "tool_precision": round(precision, 6),
        "tool_recall": round(recall, 6),
        "tsf1": round(tsf1, 6),
        "lcs_length": len(alignment),
        "tos": round(tos, 6),
        "parameter_step_scores": [round(value, 6) for value in parameter_scores],
        "pas": round(pas, 6),
        "matched_successful_calls": matched_success,
        "physical_tool_call_attempts": len(attempts),
        "pending_function_calls": len(conversation.pending_tool_calls()),
        "ecr": round(ecr, 6),
        "accuracy_score": round(accuracy_score, 4),
        "task_success_gate": task_success,
        "model_total_tokens": total_tokens,
        "token_budget": token_budget,
        "token_usage_complete": usage_complete,
        "mte": None if mte is None else round(mte, 6),
        "gold_tool_cost_units": len(gold_steps),
        "actual_tool_cost_units": len(attempts),
        "tce": round(tce, 6),
        "cost_score": None if cost_score is None else round(cost_score, 4),
        "final_score": None if final_score is None else round(final_score, 4),
        "score_ready": final_score is not None,
    }


def _domain_macro_average(samples: list[dict[str, Any]], field: str) -> float | None:
    by_domain: dict[Any, list[float]] = {}
    for sample in samples:
        value = (sample.get("scores") or {}).get(field)
        if value is None:
            return None
        by_domain.setdefault(sample["category_id"], []).append(float(value))
    if not by_domain:
        return None
    domain_means = [sum(values) / len(values) for values in by_domain.values()]
    return sum(domain_means) / len(domain_means)


def aggregate_scores(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "fas",
        "tsf1",
        "tos",
        "pas",
        "ecr",
        "accuracy_score",
        "mte",
        "tce",
        "cost_score",
        "final_score",
    )
    macro = {
        field: None if (value := _domain_macro_average(samples, field)) is None else round(value, 4)
        for field in fields
    }
    return {
        "protocol": SCORING_PROTOCOL,
        "aggregation": "macro average over represented domains",
        "domain_count": len({sample["category_id"] for sample in samples}),
        "sample_count": len(samples),
        "token_budget_per_gold_step": (samples[0].get("scores") or {}).get("token_budget", 0)
        // max(1, len(samples[0].get("gold_steps") or []))
        if samples
        else DEFAULT_TOKEN_BUDGET_PER_GOLD_STEP,
        "total_model_tokens": sum(
            (int((sample.get("scores") or {}).get("model_total_tokens") or 0) for sample in samples)
        ),
        "score_ready": bool(samples)
        and all((bool((sample.get("scores") or {}).get("score_ready")) for sample in samples)),
        **macro,
    }
