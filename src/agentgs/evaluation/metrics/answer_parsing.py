"""Bounded, reference-independent parsing of final answer representations."""

from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlsplit, urlunsplit

NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NUMBER_RE = re.compile(r"(?<![\w.])" + NUMBER + r"(?!\d|\.\d|,\d)")
TRANSLATION = str.maketrans({"−": "-", "–": "-", "′": "'", "″": '"', "’": "'", "“": '"', "”": '"'})


def clean_text(value) -> str:
    text = str(value if value is not None else "").translate(TRANSLATION)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"^\s*```[^\n]*\n([\s\S]*?)\n?```\s*$", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", lambda m: m[1] or m[2], text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    return text.strip()


def final_text(value) -> str:
    text = clean_text(value)
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        # A named final field is an explicit answer boundary, independent of gold.
        for key in ("final_answer", "FinalAnswer", "answer"):
            if key in obj and isinstance(obj[key], (str, int, float)):
                value = clean_text(obj[key])
                if isinstance(obj.get("unit"), str):
                    value += " " + obj["unit"]
                return value
    markers = list(re.finditer(r"(?im)^\s*(?:final\s*answer|最终答案)\s*[:：]\s*", text))
    if markers:
        text = text[markers[-1].end() :]
    return clean_text(text)


def normalized(value) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).casefold().strip(" \t\n.,;:。")


def json_safe(value):
    """Nonfinite diagnostics must not prevent persisting a failed prediction."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def requested_scalar(text: str, template: str) -> str:
    """Select a named requested measurement from a returned mapping, without gold."""
    text = final_text(text)
    if not text.startswith("{") or len(text) > 65536:
        return text
    try:
        obj = json.loads(text)
    except ValueError:
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError, RecursionError):
            return text
    aliases = {
        "fco2": {"fco2", "fco2_uatm", "fco2_μatm"},
        "tco2": {"tco2", "tco2_umol_kg", "tco2_umol/kg"},
        "capacity": {"capacity", "capacity_mw"},
        "distance": {"distance", "distance_km"},
        "area": {"area", "area_km2"},
        "temperature": {"temperature", "temperature_c"},
        "death_count": {"death_count", "total_deaths"},
    }
    slots = re.findall(r"<([^<>]+)>", template.lower())
    names = set().union(*(aliases.get(slot, set()) for slot in slots))
    candidates = []

    def visit(value, depth=0):
        if depth > 20:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in names or (depth == 0 and key in {"value", "result"}):
                    if isinstance(child, (str, int, float)) and not isinstance(child, bool):
                        unit = value.get("unit", "")
                        suffix = {
                            "_uatm": "μatm",
                            "_μatm": "μatm",
                            "_mw": "MW",
                            "_km": "km",
                            "_km2": "km2",
                            "_c": "°C",
                            "_umol_kg": "μmol/kg",
                            "_umol/kg": "μmol/kg",
                        }
                        if not unit:
                            unit = next(
                                (
                                    u
                                    for ending, u in suffix.items()
                                    if str(key).lower().endswith(ending)
                                ),
                                "",
                            )
                        candidates.append(str(child) + (" " + str(unit) if unit else ""))
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)

    visit(obj)
    if not candidates:
        return text
    return candidates[0] if len(candidates) == 1 else "unknown: ambiguous requested field"


def negated(text: str) -> bool:
    return bool(re.search(r"\b(?:not|incorrect|wrong|neither)\b|不是|不正确|错误", text, re.I))


# (dimension, scale, offset) maps a reported value to its base unit.
UNITS = {
    "km": ("length", 1000.0, 0.0),
    "m": ("length", 1.0, 0.0),
    "cm": ("length", 0.01, 0.0),
    "mm": ("length", 0.001, 0.0),
    "km2": ("area", 1e6, 0.0),
    "m2": ("area", 1.0, 0.0),
    "ha": ("area", 1e4, 0.0),
    "mw": ("power", 1e6, 0.0),
    "kw": ("power", 1e3, 0.0),
    "gw": ("power", 1e9, 0.0),
    "w": ("power", 1.0, 0.0),
    "j": ("energy", 1.0, 0.0),
    "%": ("ratio", 0.01, 0.0),
    "fraction": ("ratio", 1.0, 0.0),
    "°c": ("temperature", 1.0, 0.0),
    "k": ("temperature", 1.0, -273.15),
    "μatm": ("pressure", 1.0, 0.0),
    "atm": ("pressure", 1e6, 0.0),
    "pa": ("pressure", 1e6 / 101325.0, 0.0),
    "μmol/kg": ("concentration", 1.0, 0.0),
    "mmol/kg": ("concentration", 1e3, 0.0),
    "mol/kg": ("concentration", 1e6, 0.0),
    "°": ("angle", 1.0, 0.0),
    "kyr bp": ("age", 1000.0, 0.0),
    "yr bp": ("age", 1.0, 0.0),
    "count": ("count", 1.0, 0.0),
}
UNIT_ALIASES = {
    "kilometers": "km",
    "kilometres": "km",
    "meters": "m",
    "metres": "m",
    "square kilometers": "km2",
    "square kilometres": "km2",
    "km^2": "km2",
    "m^2": "m2",
    "percent": "%",
    "percentage": "%",
    "degrees celsius": "°c",
    "celsius": "°c",
    "degc": "°c",
    "° c": "°c",
    "kelvin": "k",
    "degrees": "°",
    "degree": "°",
    "uatm": "μatm",
    "microatmospheres": "μatm",
    "umol/kg": "μmol/kg",
    "events": "count",
    "event": "count",
    "records": "count",
    "record": "count",
    "cells": "count",
    "cell": "count",
    "deaths": "count",
    "people": "count",
    "years bp": "yr bp",
    "ratio": "fraction",
}
UNIT_RE = re.compile(
    r"\s*("
    + "|".join(
        re.escape(u) for u in sorted(UNITS.keys() | UNIT_ALIASES.keys(), key=len, reverse=True)
    )
    + r")(?![\w/^])",
    re.I,
)


@dataclass(frozen=True)
class Quantity:
    value: float
    raw: str
    unit: str | None


def quantities(text: str) -> list[Quantity]:
    text = clean_text(text)
    text = re.sub(r"https?://\S+", " ", text)
    found = []
    end = 0
    for match in NUMBER_RE.finditer(text):
        if match.start() < end:
            continue
        raw = match[0].replace(",", "")
        value = float(raw)
        unit_match = UNIT_RE.match(text, match.end())
        unit = None
        if unit_match:
            name = unit_match[1].casefold()
            unit = UNIT_ALIASES.get(name, name)
            end = unit_match.end()
            if UNIT_RE.match(text, end):
                unit = "unknown:multiple_units"
        else:
            tail = text[match.end() :]
            unknown = re.match(r"\s*([a-zA-Zμ°%]+(?:[/^][\w]+)?)", tail)
            if unknown and unknown[1].lower() not in {"and", "or", "which", "is", "was"}:
                unit = "unknown:" + unknown[1]
        found.append(Quantity(value, raw, unit))
    return found


def tolerance(raw: str, question: str) -> float:
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    match = re.search(
        r"(?:round(?:ed)?(?:\s+each\s+value)?\s+to|kept\s+to|keep)\s+"
        r"(\d+|zero|one|two|three|four|five|six)\s+decimal",
        question,
        re.I,
    )
    if match:
        digits = int(match[1]) if match[1].isdigit() else words[match[1].lower()]
        return 0.5 * 10.0**-digits
    mantissa, _, exponent = raw.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return 0.5 * 10.0 ** (int(exponent or 0) - decimals) if decimals or exponent else 0.0


def convert(actual: Quantity, expected_unit: str | None, *, anomaly=False) -> float | None:
    if not math.isfinite(actual.value):
        return None
    # Omitted units inherit the requested unit; explicit incompatible units never do.
    if actual.unit is None:
        return actual.value
    if actual.unit.startswith("unknown:"):
        return None
    if expected_unit is None:
        return actual.value
    if actual.unit not in UNITS or expected_unit not in UNITS:
        return actual.value if actual.unit == expected_unit else None
    src, dst = UNITS[actual.unit], UNITS[expected_unit]
    if src[0] != dst[0]:
        return None
    offset = 0.0 if anomaly else src[2] - dst[2]
    return (actual.value * src[1] + offset) / dst[1]


def quantity_match(expected: Quantity, actual: Quantity, question: str) -> bool:
    value = convert(actual, expected.unit, anomaly="anomaly" in question.lower())
    if value is None or not math.isfinite(expected.value):
        return False
    if expected.unit == "count" and not value.is_integer():
        return False
    # Precision comes from the reference or task, never a rounded prediction.
    return math.isclose(
        expected.value, value, rel_tol=1e-9, abs_tol=tolerance(expected.raw, question) + 1e-12
    )


def scalar(text: str) -> tuple[Quantity | None, str]:
    text = final_text(text)
    for group in re.findall(r"\d[\d,]*,\d[\d,]*", text):
        if not re.fullmatch(r"\d{1,3}(?:,\d{3})+", group):
            return None, "invalid_digit_grouping"
    if negated(text) or re.search(r"\b(?:unknown|nan|inf(?:inity)?)\b", text, re.I):
        return None, "negated_or_unknown"
    values = quantities(text)
    if not values:
        return None, "missing"
    first = next((value for value in values if value.unit is not None), values[0])
    if not math.isfinite(first.value):
        return None, "nonfinite"
    for value in values:
        converted = convert(value, first.unit)
        if converted is None or not math.isclose(
            first.value, converted, rel_tol=1e-12, abs_tol=1e-12
        ):
            return None, "ambiguous_values"
    return first, "parsed"


def urls(text: str) -> list[str]:
    result = []
    for match in re.finditer(r"https?://[^\s<>\"'`;,]+", text, re.I):
        value = match[0].rstrip(".")
        for left, right in [("(", ")"), ("[", "]")]:
            while value.endswith(right) and value.count(right) > value.count(left):
                value = value[:-1]
        parts = urlsplit(value)
        result.append(
            urlunsplit(
                (
                    parts.scheme.lower(),
                    parts.netloc.lower(),
                    parts.path,
                    parts.query,
                    parts.fragment,
                )
            )
        )
    return list(dict.fromkeys(result))


def dates(text: str) -> list[str]:
    text = final_text(text)
    matches = re.findall(
        r"(?<!\d)(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})(?:日)?(?!\d)", text
    )
    months = "January February March April May June July August September October November December".split()
    month_names = {name.lower(): i for i, name in enumerate(months, 1)}
    month_names.update({name[:3].lower(): i for i, name in enumerate(months, 1)})
    month_re = "(?:" + "|".join(month_names) + ")"
    for month, day, year in re.findall(
        r"\b(" + month_re + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})\b", text, re.I
    ):
        matches.append((year, str(month_names[month.lower()]), day))
    for day, month, year in re.findall(
        r"\b(\d{1,2})\s+(" + month_re + r")\.?[,]?\s+(\d{4})\b", text, re.I
    ):
        matches.append((year, str(month_names[month.lower()]), day))
    values = []
    for year, month, day in matches:
        try:
            values.append(date(int(year), int(month), int(day)).isoformat())
        except ValueError:
            values.append("invalid_date")
    return list(dict.fromkeys(values))


def dms(text: str) -> list[tuple[float, str, str]]:
    pattern = r"(?<![\w.-])(\d+)\s*°\s*(\d+)\s*'\s*(" + NUMBER + r")\s*\"\s*([NSEW])\b"
    values = []
    for degree, minute, second, hemisphere in re.findall(pattern, clean_text(text), re.I):
        deg, mins, secs = int(degree), int(minute), float(second)
        hemisphere = hemisphere.upper()
        limit = 90 if hemisphere in "NS" else 180
        if mins >= 60 or not 0 <= secs < 60 or deg + mins / 60 + secs / 3600 > limit:
            return []
        values.append((deg * 3600 + mins * 60 + secs, hemisphere, second))
    return list(dict.fromkeys(values))
