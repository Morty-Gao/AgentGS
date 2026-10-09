from __future__ import annotations

import math
import re

from agentgs.agent.answering.evidence.core import _numbers


def _option_blocks(question: str) -> list[tuple[str, str]]:
    match = re.search("Options\\s*:\\s*(.*)$", question or "", re.I | re.S)
    if not match:
        return []
    text = re.split(
        "\\n\\s*Evaluation input resource\\s*\\(", match.group(1), maxsplit=1, flags=re.IGNORECASE
    )[0]
    blocks = re.findall(
        "(?:^|[\\s\\u00a0;；|])([A-H])\\s*[\\.、:：]\\s*(.*?)(?=[\\s\\u00a0;；|]+[A-H]\\s*[\\.、:：]|$)",
        text,
        re.I | re.S,
    )
    return [(label.upper(), body.strip()) for label, body in blocks]


def _reported_close(raw: str, observed: float) -> bool:
    """Allow stated decimal rounding and 0.5% numeric tolerance, not option snapping."""
    expected = float(raw)
    if not math.isfinite(expected) or not math.isfinite(observed):
        return False
    mantissa, _, exponent = raw.lower().partition("e")
    rounding = (
        0.5 * 10 ** (int(exponent or "0") - len(mantissa.split(".")[1])) if "." in mantissa else 0.0
    )
    return abs(expected - observed) <= max(1e-12, rounding, 0.005 * abs(observed))


def _best_option(question: str, evidence: list[float]) -> tuple[str, str] | None:
    options = _option_blocks(question)
    if not options or not evidence:
        return None
    text = str(question or "").casefold()
    if re.search("\\bsen(?:['’]s)?\\s+slope\\b", text):
        return None

    def measurement_values(body: str) -> list[float]:
        without_dates = re.sub("\\b(?:19|20)\\d{2}[-/]\\d{1,2}[-/]\\d{1,2}\\b", " ", body)
        without_dates = re.sub("\\b(?:19|20)\\d{2}\\b", " ", without_dates)
        return _numbers(without_dates)

    option_values = [value for _, body in options for value in measurement_values(body)]
    comparison_evidence = list(evidence)
    if any(
        (
            token in text
            for token in ("proportion", "percentage", "coverage", "water body", "hotspot")
        )
    ):
        comparison_evidence.extend(
            (value / 100.0 for value in evidence if 1.0 < abs(value) <= 100.0)
        )
        comparison_evidence.extend(
            (value * 100.0 for value in evidence if 0.0 <= abs(value) <= 1.0)
        )
    if any((token in text for token in ("temperature", "thermal", "b10", "band 10"))):
        comparison_evidence.extend(
            (value - 273.15 for value in evidence if 150.0 <= value <= 400.0)
        )
        if "kelvin" not in text:
            comparison_evidence.extend(
                (value / 10.0 for value in evidence if 150.0 <= value <= 400.0)
            )
    directional_magnitude = any(
        (
            token in text
            for token in (
                "how much higher",
                "how much lower",
                "how much warmer",
                "how much colder",
                "warmer",
                "colder",
            )
        )
    )
    if (
        option_values
        and any((value < 0 for value in evidence))
        and (
            directional_magnitude
            or any(
                (
                    token in str(question or "").casefold()
                    for token in ("difference", "change", "loss", "decrease")
                )
            )
        )
        and (all((value >= 0 for value in option_values)) or directional_magnitude)
    ):
        comparison_evidence.extend((abs(value) for value in evidence if value < 0))
    compatible = []
    for label, body in options:
        cleaned = re.sub("\\b(?:19|20)\\d{2}(?:[-/]\\d{1,2}[-/]\\d{1,2})?\\b", " ", body)
        raw_values = re.findall("(?<![A-Za-z])[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?", cleaned)
        if not raw_values:
            continue
        if all(
            (
                any((_reported_close(raw, observed) for observed in comparison_evidence))
                for raw in raw_values
            )
        ):
            compatible.append((label, body))
    return compatible[0] if len(compatible) == 1 else None
