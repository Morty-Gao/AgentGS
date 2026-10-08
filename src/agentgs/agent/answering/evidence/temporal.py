from __future__ import annotations

import math
import re
from typing import Any

from agentgs.agent.answering.evidence.core import _decode_output, _numbers, _walk_dicts
from agentgs.agent.answering.evidence.options import _option_blocks

_MONTH_LOOKUP = {
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


def _dated_mentions(text: str) -> list[tuple[int, int, tuple[int | None, int, int]]]:
    mentions: list[tuple[int, int, tuple[int | None, int, int]]] = []
    occupied: list[tuple[int, int]] = []
    for match in re.finditer(
        "(?<!\\d)((?:19|20)\\d{2})[-_/](\\d{1,2})[-_/](\\d{1,2})(?!\\d)", text
    ):
        year, month, day = (int(value) for value in match.groups())
        mentions.append((match.start(), match.end(), (year, month, day)))
        occupied.append((match.start(), match.end()))
    month_pattern = "|".join(_MONTH_LOOKUP)
    for match in re.finditer(
        f"\\b({month_pattern})\\s+(\\d{{1,2}})(?:st|nd|rd|th)?(?:\\s*,?\\s*((?:19|20)\\d{{2}}))?\\b",
        text,
        re.IGNORECASE,
    ):
        if any((start <= match.start() < end for start, end in occupied)):
            continue
        month_name, day, year = match.groups()
        mentions.append(
            (
                match.start(),
                match.end(),
                (int(year) if year else None, _MONTH_LOOKUP[month_name.casefold()], int(day)),
            )
        )
    return sorted(mentions)


def _temporal_series_option(
    question: str, successful_outputs: list[Any] | tuple[Any, ...]
) -> tuple[str, str] | None:
    """Match dated Tool series to dated option measurements."""
    options = _option_blocks(question)
    if not options:
        return None
    observations: dict[tuple[int, int, int], list[float]] = {}
    dated_series: list[tuple[str, dict[tuple[int, int, int], float]]] = []
    for output in successful_outputs:
        value = _decode_output(output)
        for candidate in _walk_dicts(value):
            items = candidate.get("values")
            if not isinstance(items, list):
                continue
            current_series: dict[tuple[int, int, int], float] = {}
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    continue
                observed = item.get("value")
                if not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
                    continue
                path_dates = _dated_mentions(str(item["path"]))
                if not path_dates:
                    continue
                date = path_dates[-1][2]
                if date[0] is not None:
                    full_date = (date[0], date[1], date[2])
                    numeric_observed = float(observed)
                    observations.setdefault(full_date, []).append(numeric_observed)
                    current_series[full_date] = numeric_observed
            if current_series:
                dated_series.append(
                    (str(candidate.get("statistic") or "").casefold(), current_series)
                )
    if not observations:
        return None
    question_text = str(question or "").casefold()
    preferred_series: dict[tuple[int, int, int], float] | None = None
    if any(
        (token in question_text for token in ("proportion", "percentage", "coverage", "hotspot"))
    ):
        preferred_series = next(
            (
                series
                for statistic, series in reversed(dated_series)
                if statistic == "hotspot_percentage"
            ),
            None,
        )
    if preferred_series is None and any(
        (token in question_text for token in ("mean ndvi", "mean ndsi", "mean ndwi", "average"))
    ):
        preferred_series = next(
            (series for statistic, series in reversed(dated_series) if statistic == "mean"), None
        )
    if preferred_series is None and dated_series:
        preferred_series = dated_series[-1][1]

    def same_date(left: tuple[int | None, int, int], right: tuple[int, int, int]) -> bool:
        year, month, day = left
        return month == right[1] and day == right[2] and (year is None or year == right[0])

    maximum_date = max(preferred_series, key=preferred_series.get) if preferred_series else None
    minimum_date = min(preferred_series, key=preferred_series.get) if preferred_series else None
    scored: list[tuple[int, float, int, str, str]] = []
    fully_compatible: list[tuple[str, str]] = []
    for label, body in options:
        mentions = _dated_mentions(body)
        pair_distances: list[float] = []
        extrema_penalty = 0
        for index, (_start, end, option_date) in enumerate(mentions):
            segment_end = mentions[index + 1][0] if index + 1 < len(mentions) else len(body)
            measurements = _numbers(body[end:segment_end])
            if not measurements:
                continue
            year, month, day = option_date
            matched = [
                observed
                for (obs_year, obs_month, obs_day), values in observations.items()
                if obs_month == month and obs_day == day and (year is None or obs_year == year)
                for observed in values
            ]
            if not matched:
                continue
            observed_variants = list(matched)
            observed_variants.extend(
                (value / 100.0 for value in matched if 1.0 < abs(value) <= 100.0)
            )
            observed_variants.extend(
                (value * 100.0 for value in matched if 0.0 <= abs(value) <= 1.0)
            )
            measurement = measurements[0]
            pair_distances.append(
                min(
                    (
                        abs(measurement - observed) / max(0.01, abs(measurement), abs(observed))
                        for observed in observed_variants
                    )
                )
            )
        for start, _end, option_date in mentions:
            prefix = body[max(0, start - 80) : start].casefold()
            is_maximum_claim = bool(
                re.search(
                    "(?:maximum|highest|peak|largest|most severe)[^.;；]{0,50}(?:on|at)?\\s*$",
                    prefix,
                )
            )
            is_minimum_claim = bool(
                re.search(
                    "(?:minimum|lowest|smallest|least severe)[^.;；]{0,50}(?:on|at)?\\s*$", prefix
                )
            )
            if (
                is_maximum_claim
                and maximum_date is not None
                and (not same_date(option_date, maximum_date))
            ):
                extrema_penalty += 1
            if (
                is_minimum_claim
                and minimum_date is not None
                and (not same_date(option_date, minimum_date))
            ):
                extrema_penalty += 1
        if pair_distances:
            mean_distance = sum(pair_distances) / len(pair_distances)
            scored.append((extrema_penalty, mean_distance, -len(pair_distances), label, body))
            if extrema_penalty == 0 and all((distance <= 0.01 for distance in pair_distances)):
                fully_compatible.append((label, body))
    if fully_compatible:
        return fully_compatible[0]
    if not scored:
        return None
    extrema_penalty, distance, _neg_count, label, body = min(scored)
    return (label, body) if extrema_penalty == 0 and distance <= 0.08 else None


def _temporal_option(
    question: str, successful_outputs: list[Any] | tuple[Any, ...]
) -> tuple[str, str] | None:
    """Map a Tool-returned extremum index back to dated raster provenance."""
    options = _option_blocks(question)
    if not options:
        return None
    decoded = [_decode_output(output) for output in successful_outputs]
    terminal_index: int | None = None
    terminal_position = len(decoded)
    for position in range(len(decoded) - 1, -1, -1):
        candidate = next(
            (
                item
                for item in _walk_dicts(decoded[position])
                if isinstance(item.get("index"), (int, float))
            ),
            None,
        )
        if candidate is not None:
            terminal_index = int(candidate["index"])
            terminal_position = position
            break
    if terminal_index is None or terminal_index < 0:
        return None
    series: list[tuple[str, list[dict[str, Any]]]] = []
    for value in decoded[:terminal_position]:
        for candidate in _walk_dicts(value):
            items = candidate.get("values")
            if not isinstance(items, list) or terminal_index >= len(items):
                continue
            if all(
                (
                    isinstance(item, dict)
                    and isinstance(item.get("path"), str)
                    and isinstance(item.get("value"), (int, float))
                    for item in items
                )
            ):
                series.append((str(candidate.get("statistic") or ""), items))
    if not series:
        return None
    text = str(question or "").casefold()
    asks_for_proportion = any(
        (
            phrase in text
            for phrase in (
                "largest proportion",
                "highest proportion",
                "most severe water loss",
                "highest ndwi",
            )
        )
    )
    asks_for_index_value = bool(
        re.search(
            "(?:highest|maximum|peak)\\s+(?:mean\\s+)?(?:ndvi|ndsi|ndwi|ndti|nbr)\\s+value", text
        )
    )
    selected: tuple[str, list[dict[str, Any]]] | None = None
    if asks_for_proportion:
        selected = next(
            (item for item in reversed(series) if item[0] == "hotspot_percentage"), None
        )
    elif asks_for_index_value:
        selected = next((item for item in reversed(series) if item[0] == "mean"), None)
    selected = selected or series[-1]
    path = str(selected[1][terminal_index].get("path") or "")
    path_dates = [mention[2] for mention in _dated_mentions(path)]
    if not path_dates:
        return None
    scored: list[tuple[int, str, str]] = []
    for label, body in options:
        body_dates = [mention[2] for mention in _dated_mentions(body)]
        score = sum(
            (
                any(
                    (
                        path_month == body_month
                        and path_day == body_day
                        and (body_year is None or path_year == body_year)
                        for body_year, body_month, body_day in body_dates
                    )
                )
                for path_year, path_month, path_day in path_dates
            )
        )
        if score:
            scored.append((score, label, body))
    if not scored:
        return None
    score, label, body = max(scored, key=lambda item: item[0])
    return (label, body) if score == len(path_dates) else None
