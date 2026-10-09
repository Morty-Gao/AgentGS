from __future__ import annotations

import json
import math
import re
from typing import Any

from agentgs.agent.answering.evidence.core import _decode_output, _numbers, _walk_dicts
from agentgs.agent.answering.evidence.options import _option_blocks, _reported_close


def _observed_comparison_pair(
    successful_outputs: list[Any] | tuple[Any, ...],
) -> tuple[float, float] | None:
    """Recover two ordered cohort observations from executed output schemas."""
    decoded = [_decode_output(output) for output in successful_outputs]
    scalar_values = [
        float(value["value"])
        for value in decoded
        if isinstance(value, dict) and isinstance(value.get("value"), (int, float))
    ]
    if len(scalar_values) >= 2:
        return (scalar_values[0], scalar_values[1])
    for semantic_key in ("percentage", "proportion", "rate", "coefficient_of_variation", "cv"):
        semantic_values: list[float] = []
        for value in decoded:
            observed = next(
                (
                    candidate.get(semantic_key)
                    for candidate in _walk_dicts(value)
                    if isinstance(candidate.get(semantic_key), (int, float))
                    and math.isfinite(float(candidate[semantic_key]))
                ),
                None,
            )
            if observed is not None:
                semantic_values.append(float(observed))
        if len(semantic_values) >= 2:
            return (semantic_values[0], semantic_values[1])
    cohort_values: list[float] = []
    for value in decoded:
        if not isinstance(value, dict) or value.get("statistic") != "mean":
            continue
        rows = value.get("values")
        if not isinstance(rows, list):
            continue
        numbers = [
            float(item["value"])
            for item in rows
            if isinstance(item, dict) and isinstance(item.get("value"), (int, float))
        ]
        if numbers:
            cohort_values.append(sum(numbers) / len(numbers))
    if len(cohort_values) >= 2:
        return (cohort_values[-2], cohort_values[-1])
    return None


def _ordered_comparison_option(
    question: str, successful_outputs: list[Any] | tuple[Any, ...]
) -> tuple[str, str] | None:
    """Match choices that assign measurements to ordered years or months."""
    options = _option_blocks(question)
    observed = _observed_comparison_pair(successful_outputs)
    if not options or observed is None:
        return None
    stem = re.split("Options\\s*:", question or "", maxsplit=1, flags=re.I)[0]
    periods: list[str] = []
    for year in re.findall("\\b(?:19|20)\\d{2}\\b", stem):
        if year not in periods:
            periods.append(year)
    if len(periods) < 2:
        periods = []
        for month in re.findall(
            "\\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\\b",
            stem,
            re.I,
        ):
            normalized = month.casefold()
            if normalized not in periods:
                periods.append(normalized)
    if len(periods) < 2:
        return None
    periods = periods[:2]
    number_pattern = re.compile("(?<![A-Za-z])[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?")
    scored: list[tuple[float, str, str]] = []
    for label, body in options:
        number_matches = list(number_pattern.finditer(body))
        if not number_matches:
            continue
        assigned: list[float] = []
        used: set[int] = set()
        for period in periods:
            mentions = list(re.finditer(f"\\b{re.escape(period)}\\b", body, re.I))
            if not mentions:
                assigned = []
                break
            mention = mentions[0]
            center = (mention.start() + mention.end()) / 2
            candidates: list[tuple[float, int, float]] = []
            for index, number_match in enumerate(number_matches):
                if index in used:
                    continue
                raw = number_match.group(0)
                if period.isdigit() and raw.lstrip("+-") == period:
                    continue
                number_center = (number_match.start() + number_match.end()) / 2
                candidates.append((abs(number_center - center), index, float(raw)))
            if not candidates:
                assigned = []
                break
            _distance, index, value = min(candidates)
            used.add(index)
            assigned.append(value)
        if len(assigned) != 2:
            continue
        pair_error = (
            sum(
                (
                    abs(expected - actual) / max(1.0, abs(actual))
                    for expected, actual in zip(assigned, observed)
                )
            )
            / 2
        )
        remaining = [
            float(match.group(0))
            for index, match in enumerate(number_matches)
            if index not in used
            and (not re.fullmatch("(?:19|20)\\d{2}", match.group(0).lstrip("+-")))
        ]
        change = abs(observed[1] - observed[0])
        change_error = min(
            (abs(abs(value) - change) / max(1.0, change) for value in remaining), default=0.0
        )
        scored.append((pair_error + change_error, label, body))
    if not scored:
        return None
    distance, label, body = min(scored, key=lambda item: item[0])
    return (label, body) if distance <= 0.12 else None


def _change_summary_option(
    question: str, successful_outputs: list[Any] | tuple[Any, ...]
) -> tuple[str, str] | None:
    """Match choices that report a signed delta and percentage change."""
    text = str(question or "").casefold()
    if not any((token in text for token in ("delta", "difference", "change", "δ", "Δ".casefold()))):
        return None
    observed = _observed_comparison_pair(successful_outputs)
    options = _option_blocks(question)
    if observed is None or not options or observed[0] == 0:
        return None
    delta = observed[1] - observed[0]
    percentage = delta / observed[0] * 100.0
    scored: list[tuple[float, str, str]] = []
    for label, body in options:
        cleaned = re.sub("\\b(?:19|20)\\d{2}(?:[-/]\\d{1,2}[-/]\\d{1,2})?\\b", " ", body)
        values = _numbers(cleaned)
        if len(values) < 2:
            continue
        option_delta, option_percentage = (values[0], values[-1])
        delta_error = abs(option_delta - delta) / max(0.01, abs(option_delta), abs(delta))
        percentage_error = abs(option_percentage - percentage) / max(
            1.0, abs(option_percentage), abs(percentage)
        )
        scored.append(((delta_error + percentage_error) / 2.0, label, body))
    if not scored:
        return None
    distance, label, body = min(scored, key=lambda item: item[0])
    return (label, body) if distance <= 0.5 else None


def _directional_comparison_option(
    question: str, successful_outputs: list[Any] | tuple[Any, ...]
) -> tuple[str, str] | None:
    """Use ordered Tool evidence when choices disagree on change direction.

    This is intentionally a fallback after numeric comparison.  It handles
    reporting-scale mismatches while still requiring a unique option whose
    qualitative direction agrees with the executed cohort observations.
    """
    options = _option_blocks(question)
    observed = _observed_comparison_pair(successful_outputs)
    text = str(question or "").casefold()
    if (
        not options
        or observed is None
        or (
            not any(
                (
                    token in text
                    for token in (
                        "difference",
                        "change",
                        "compare",
                        "increased",
                        "decreased",
                        "trend",
                    )
                )
            )
        )
    ):
        return None
    first, second = observed
    tolerance = max(1e-12, 0.005 * max(abs(first), abs(second), 1e-12))
    direction = (
        "stable"
        if abs(second - first) <= tolerance
        else "increase"
        if second > first
        else "decrease"
    )
    direction_tokens = {
        "increase": (
            "increase",
            "increased",
            "increasing",
            "intensified",
            "intensification",
            "grew",
            "higher",
        ),
        "decrease": (
            "decrease",
            "decreased",
            "decreasing",
            "alleviated",
            "reduced",
            "declined",
            "lower",
        ),
        "stable": ("unchanged", "stable", "remained the same", "no change"),
    }
    incompatible_tokens = tuple(
        (
            token
            for name, tokens in direction_tokens.items()
            if name != direction
            for token in tokens
        )
    )
    compatible: list[tuple[str, str]] = []
    for label, body in options:
        body_text = body.casefold()
        if not any((token in body_text for token in direction_tokens[direction])):
            continue
        if any((token in body_text for token in incompatible_tokens)):
            continue
        compatible.append((label, body))
    return compatible[0] if len(compatible) == 1 else None


def _semantic_option(
    question: str, successful_outputs: list[Any] | tuple[Any, ...]
) -> tuple[str, str] | None:
    """Select categorical statistical choices from executed Tool semantics."""
    options = _option_blocks(question)
    text = str(question or "").casefold()
    if not options:
        return None
    candidates: list[dict[str, Any]] = []
    for output in successful_outputs:
        try:
            value = json.loads(output) if isinstance(output, str) else output
        except (TypeError, json.JSONDecodeError):
            continue
        candidates.extend(_walk_dicts(value))
    if re.search("\\bsen(?:['’]s)?\\s+slope\\b", text):
        result = next(
            (
                candidate
                for candidate in reversed(candidates)
                if isinstance(candidate.get("slope"), (int, float))
            ),
            None,
        )
        if result is None:
            return None
        slope = float(result["slope"])
        direction_tokens = (
            ("upward", "increase", "increasing", "positive", "recovery")
            if slope > 0
            else ("downward", "decrease", "decreasing", "negative", "intensifying")
            if slope < 0
            else ("stable", "no trend", "unchanged")
        )
        compatible: list[tuple[str, str]] = []
        for label, body in options:
            if not any((token in body.casefold() for token in direction_tokens)):
                continue
            cleaned = re.sub("\\b(?:19|20)\\d{2}\\b", " ", body)
            raw_numbers = re.findall(
                "(?<![A-Za-z])[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?", cleaned
            )
            if raw_numbers and (not all((_reported_close(raw, slope) for raw in raw_numbers))):
                continue
            compatible.append((label, body))
        return compatible[0] if len(compatible) == 1 else None
    if "mann-kendall" not in text:
        return None
    result = next(
        (
            candidate
            for candidate in reversed(candidates)
            if isinstance(candidate.get("p_value"), (int, float))
        ),
        None,
    )
    if result is None:
        return None
    significant = float(result["p_value"]) < 0.05
    trend = str(result.get("trend") or "").casefold()
    scored: list[tuple[int, str, str]] = []
    for label, body in options:
        body_text = body.casefold()
        score = 0
        says_no_significance = any(
            (
                phrase in body_text
                for phrase in (
                    "no significant",
                    "not significant",
                    "non-significant",
                    "insignificant",
                )
            )
        )
        if says_no_significance == (not significant):
            score += 4
        if significant and "significant" in body_text and (not says_no_significance):
            score += 2
        if trend in {"increasing", "upward"} and any(
            (token in body_text for token in ("increasing", "upward"))
        ):
            score += 1
        if trend in {"decreasing", "downward"} and any(
            (token in body_text for token in ("decreasing", "downward"))
        ):
            score += 1
        scored.append((score, label, body))
    score, label, body = max(scored, key=lambda item: item[0])
    return (label, body) if score >= 4 else None
