from __future__ import annotations

import json
import math
import re
from typing import Any

_EVIDENCE_KEYS = {
    "value",
    "mean",
    "average",
    "difference",
    "distance",
    "area",
    "energy",
    "magnitude",
    "slope",
    "intercept",
    "percentage",
    "proportion",
    "ratio",
    "capacity",
    "growth",
    "count",
    "total",
    "sum",
    "min",
    "max",
    "index",
    "temperature",
    "depth",
    "elevation",
    "trend",
    "dlnvs",
    "fco2",
    "tco2",
    "annual",
    "cv",
    "coefficient",
    "statistic",
    "p_value",
    "change",
}


def _walk(value: Any, key: str = ""):
    if isinstance(value, dict):
        for name, item in value.items():
            yield from _walk(item, str(name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item, key)
    elif (
        isinstance(value, (int, float))
        and (not isinstance(value, bool))
        and math.isfinite(float(value))
    ):
        yield (key.casefold(), float(value))


def _numbers(text: str) -> list[float]:
    values = []
    for raw in re.findall("(?<![A-Za-z])[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?", text or ""):
        try:
            values.append(float(raw))
        except ValueError:
            pass
    return values


def _semantic_question_text(question: str = "") -> str:
    """Return user intent without the evaluator's resource locator metadata."""
    return str(question or "").split("Evaluation input resource", 1)[0].casefold().strip()


def evidence_numbers(outputs: list[Any] | tuple[Any, ...]) -> list[float]:
    scored: list[tuple[int, float]] = []
    for output in outputs:
        try:
            value = json.loads(output) if isinstance(output, str) else output
        except (TypeError, json.JSONDecodeError):
            value = output
        for key, number in _walk(value):
            score = 2 if any((token in key for token in _EVIDENCE_KEYS)) else 0
            if score:
                scored.append((score, number))
    if not scored:
        for output in outputs:
            scored.extend(((0, number) for number in _numbers(str(output))))
    result: list[float] = []
    for _score, number in sorted(scored, key=lambda item: -item[0]):
        if not any(
            (math.isclose(number, existing, rel_tol=1e-09, abs_tol=1e-09) for existing in result)
        ):
            result.append(number)
    return result


def _find_tables(value: Any):
    if isinstance(value, dict):
        columns = value.get("columns")
        rows = value.get("rows")
        if (
            isinstance(columns, list)
            and columns
            and isinstance(rows, list)
            and all((isinstance(row, (list, tuple)) for row in rows))
        ):
            yield (columns, rows)
        for item in value.values():
            yield from _find_tables(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _find_tables(item)


def _decode_output(output: Any) -> Any:
    if isinstance(output, (dict, list, tuple)):
        return output
    try:
        return json.loads(output) if isinstance(output, str) else output
    except (TypeError, json.JSONDecodeError):
        return None


def _last_structured_output(outputs: list[Any] | tuple[Any, ...]) -> Any:
    for output in reversed(outputs):
        value = _decode_output(output)
        if isinstance(value, (dict, list, tuple)):
            return value
    return None


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_dicts(item)


def _first_dict_with_keys(value: Any, keys: set[str]) -> dict[str, Any] | None:
    for candidate in _walk_dicts(value):
        if keys <= {str(key).casefold() for key in candidate}:
            return candidate
    return None


def _casefold_value(candidate: dict[str, Any], key: str) -> Any:
    target = key.casefold()
    for name, value in candidate.items():
        if str(name).casefold() == target:
            return value
    raise KeyError(key)


def _number_text(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.{digits}f}"


def _scientific_text(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.{digits}e}"
