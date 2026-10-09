from __future__ import annotations

import re
from typing import Any, Iterable

from agentgs.agent.execution.bindings import split_question_resource

_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys((value for value in values if value)))


def _task_text(text: str) -> str:
    """Exclude multiple-choice distractors from capability inference."""
    return re.split("\\boptions\\s*:", text or "", maxsplit=1, flags=re.IGNORECASE)[
        0
    ].strip()


def _operation_text(text: str) -> str:
    """Remove capability names that contain algebra words.

    Spectral-index expansions such as ``Normalized Difference Snow Index``
    describe one raster capability; the word ``difference`` inside the name is
    not a request for a downstream scalar subtraction.
    """
    cleaned = _task_text(text).casefold()
    return re.sub(
        "\\bnormalized\\s+difference\\s+(?:snow|water|vegetation|turbidity|burn)\\s+(?:ratio|index)\\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )


def _requested_months(text: str) -> list[int]:
    lowered = str(text or "").casefold()
    positions: list[tuple[int, int]] = []
    for name, month in _MONTH_NUMBERS.items():
        positions.extend(
            ((match.start(), month) for match in re.finditer(f"\\b{name}\\b", lowered))
        )
    return list(dict.fromkeys((month for _, month in sorted(positions))))


def _schedule_data_parallel_reductions(plan: list[str]) -> list[str]:
    """Topologically batch independent raster reductions by dependency level."""
    producer = "RemoteSensingBatchImageMean"
    scalar_consumers = {
        "RemoteSensingMean",
        "RemoteSensingCoefficientOfVariation",
        "RemoteSensingKurtosis",
        "RemoteSensingSensSlope",
        "RemoteSensingMannKendallTest",
    }
    producer_positions = [index for index, tool in enumerate(plan) if tool == producer]
    if len(producer_positions) < 2:
        return plan
    for consumer in scalar_consumers:
        consumer_positions = [index for index, tool in enumerate(plan) if tool == consumer]
        if len(consumer_positions) != len(producer_positions):
            continue
        relevant = sorted(producer_positions + consumer_positions)
        subsequence = [plan[index] for index in relevant]
        if subsequence == [producer] * len(producer_positions) + [consumer] * len(
            consumer_positions
        ):
            continue
        if any(
            (
                consumer_positions[index] < producer_positions[index]
                for index in range(len(producer_positions))
            )
        ):
            continue
        replacement = [producer] * len(producer_positions) + [consumer] * len(consumer_positions)
        result = list(plan)
        for index, tool in zip(relevant, replacement):
            result[index] = tool
        return result
    return plan


def _canonicalize_remote_chain(plan: list[str]) -> list[str]:
    """Normalize model plans for raster cohorts without changing capabilities.

    A model sometimes repeats the directory-listing step between independent
    temporal/entity cohorts (``ListFiles → producer → reducer → ListFiles``).
    Listing is a shared dependency, so keep one instance at the front and
    batch independent producers before their scalar reducers.  The operation
    is schema-based and deliberately leaves index/raster transformations and
    terminal algebra in their original relative order.
    """
    if not any((tool.startswith("RemoteSensing") for tool in plan)):
        return plan
    result: list[str] = []
    seen_list = False
    for tool in plan:
        if tool == "RemoteSensingListFiles":
            if not seen_list:
                result.append(tool)
                seen_list = True
            continue
        result.append(tool)
    return _schedule_data_parallel_reductions(result)


def _resource_parts(question: str) -> tuple[str, Any, dict[str, Any], str]:
    clean, resource = split_question_resource(question)
    obj = resource if isinstance(resource, dict) else {}
    locator = obj.get("source_locator") if isinstance(obj.get("source_locator"), dict) else obj
    sources = " ".join((str(value) for value in obj.get("sources", [])))
    return (clean, resource, locator, sources.casefold())
