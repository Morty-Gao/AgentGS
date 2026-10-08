"""Versioned deterministic compatibility layer; never completes a missing answer.

Parsing of predictions is independent of the reference answer. Supported schemas
bind numbers to named quantities and compatible units, not an unordered number bag.
Uncovered formats retain the frozen legacy score with an explicit fallback marker.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from agentgs.evaluation.metrics.answers import score_answer as legacy_score

VERSION = "semantic_compat_v0.1"
NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ALIASES = {
    "moho": r"moho(?:\s+depth)?",
    "lid_bottom": r"lid[- ]bottom(?:\s+depth)?",
    "dlnvs": r"dlnvs(?:\s+at\s+"
    + NUMBER
    + r"\s*km(?:\s+depth)?|\s*\(\s*"
    + NUMBER
    + r"\s*km\s*\))?",
    "age": r"(?:nearest\s+age|actual\s+age|age|年代)",
    "temperature_anomaly": r"(?:temperature\s+anomaly|temp\s+anomaly|temperature\s+departure|t|温度异常)",
    "sea_level": r"(?:sea\s+level|sl|海平面)",
    "valid_count": r"(?:valid\s+count|valid\s+records(?:\s+in\s+\d{4})?(?:\s*\([^)]*\))?|record\s+count|记录数)",
    "mean_fco2": r"(?:mean\s+fco2|fco2)",
    "mean_sst": r"(?:mean\s+sst|sst)",
    "latitude": r"(?:latitude|lat|纬度)",
    "longitude": r"(?:longitude|lon|lng|经度)",
    "dep_id": r"(?:dep\s+id|deposit\s+id)",
}
# Canonical conversion is (value * multiplier + offset); incompatible dimensions fail.
UNITS = {
    "km": ("length", 1000.0, 0.0),
    "m": ("length", 1.0, 0.0),
    "cm": ("length", 0.01, 0.0),
    "c": ("temperature", 1.0, 0.0),
    "°c": ("temperature", 1.0, 0.0),
    "k": ("temperature", 1.0, -273.15),
    "%": ("ratio", 0.01, 0.0),
    "fraction": ("ratio", 1.0, 0.0),
    "ratio": ("ratio", 1.0, 0.0),
    "μatm": ("pressure", 1.0, 0.0),
    "uatm": ("pressure", 1.0, 0.0),
    "atm": ("pressure", 1e6, 0.0),
    "pa": ("pressure", 1e6 / 101325.0, 0.0),
    "kyr bp": ("age", 1000.0, 0.0),
    "yr bp": ("age", 1.0, 0.0),
    "years bp": ("age", 1.0, 0.0),
    "degree": ("angle", 1.0, 0.0),
    "°": ("angle", 1.0, 0.0),
    "count": ("count", 1.0, 0.0),
}


def normalized(text: Any) -> str:
    return (
        unicodedata.normalize("NFKC", str(text))
        .casefold()
        .replace("；", ";")
        .replace("：", ":")
        .replace("，", ",")
    )


def number_value(raw: str) -> float:
    return float(raw.replace(",", ""))


def reference_tolerance(raw: str, unit: str, question: str) -> float:
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    match = re.search(
        r"(?:round(?:ed)?(?:\s+each\s+value)?\s+to|kept\s+to|to)\s+(\d+|zero|one|two|three|four|five|six)\s+decimal",
        question,
        re.I,
    )
    if match:
        digits = words.get(match[1].lower(), int(match[1]) if match[1].isdigit() else 0)
        return 0.5 * 10**-digits
    raw = raw.replace(",", "").lower()
    parts = raw.split("e")
    mantissa = parts[0]
    exponent = int(parts[1]) if len(parts) > 1 else 0
    decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    return (
        0.0
        if unit == "count" or (decimals == 0 and exponent == 0)
        else 0.5 * 10 ** (exponent - decimals)
    )


def text_form(answer: str) -> str:
    text = str(answer or "").strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                text = "; ".join(
                    f"{k}: "
                    + (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v))
                    for k, v in obj.items()
                )
        except json.JSONDecodeError:
            pass
    return normalized(text).replace("_", " ")


@dataclass
class Measurement:
    value: float
    raw: str
    unit: str


def parse_unit(tail: str, canonical: str) -> str | None:
    # Read only the next quantity unit; never borrow a unit from a different field.
    tail = tail.lstrip()
    replacements = [
        ("microatmospheres", "μatm"),
        ("microatmosphere", "μatm"),
        ("degrees celsius", "°c"),
        ("celsius", "°c"),
        ("kelvin", "k"),
        ("kilometres", "km"),
        ("kilometers", "km"),
        ("metres", "m"),
        ("meters", "m"),
    ]
    for old, new in replacements:
        if tail.startswith(old):
            tail = new + tail[len(old) :]
            break
    for unit in sorted(UNITS, key=len, reverse=True):
        if re.match(re.escape(unit) + r"(?![a-z])", tail):
            return unit
    if canonical in ["count", "degree"]:
        if (
            not tail
            or re.match(r"^[;,.\n)\]]|^\s*$", tail)
            or re.match(r"(?:records?|条|个)\b", tail)
        ):
            return canonical
    return None


def convert(measurement: Measurement, canonical: str, field: str = "") -> float | None:
    src = UNITS.get(measurement.unit)
    dst = UNITS.get(normalized(canonical))
    if not src or not dst or src[0] != dst[0]:
        return None
    # Temperature anomalies are differences: 1 K anomaly equals 1 °C anomaly.
    offset = 0.0 if field == "temperature_anomaly" else src[2] - dst[2]
    value = (measurement.value * src[1] + offset) / dst[1]
    return value if math.isfinite(value) else None


def collect_numeric(text: str, field: str, unit: str) -> tuple[Measurement | None, str]:
    alias = ALIASES[field]
    label = re.compile(
        r"(?<![a-z0-9])" + alias + r"(?![a-z])\s*(?::|=|≈|is|为|是)?\s*(" + NUMBER + r")", re.I
    )
    candidates = []
    matches = list(label.finditer(text))
    for match in matches:
        if re.search(
            r"\b(?:not|incorrect|wrong)\s*$", text[max(0, match.start() - 24) : match.start()]
        ):
            return None, "negated"
        candidates.append((match.group(1), text[match.end() :]))
    # Legacy concise unlabelled records have unambiguous roles defined by units.
    if not matches:
        # Explicit but malformed labels must not fall back to values in another field.
        if re.search(r"\b" + alias + r"\s*[:=≈]", text):
            return None, "invalid_or_missing_value"
        suffix = {
            "valid_count": r"\s+records?\b",
            "mean_fco2": r"\s*(?:μatm|uatm|atm|pa)\b",
            "mean_sst": r"\s*(?:°c|celsius)\b",
        }.get(field)
        if suffix:
            for match in re.finditer("(" + NUMBER + ")(" + suffix + ")", text):
                start = (
                    max(text.rfind(";", 0, match.start()), text.rfind("\n", 0, match.start())) + 1
                )
                prefix = text[start : match.start()].strip()
                if ":" in prefix or "=" in prefix:
                    continue
                if re.search(r"\b(?:not|incorrect|wrong)\s*$", prefix):
                    return None, "negated"
                candidates.append((match.group(1), text[match.end(1) :]))
    converted = []
    valid = []
    for raw, tail in candidates:
        parsed_unit = parse_unit(tail, unit)
        if parsed_unit is None:
            return None, "missing_or_unknown_unit"
        measure = Measurement(number_value(raw), raw, parsed_unit)
        value = convert(measure, unit, field)
        if value is None:
            return None, "incompatible_unit"
        if unit == "count" and (value < 0 or not value.is_integer()):
            return None, "invalid_count"
        if field == "latitude" and not -90 <= value <= 90:
            return None, "invalid_latitude"
        if field == "longitude" and not -180 <= value <= 180:
            return None, "invalid_longitude"
        converted.append(value)
        valid.append(measure)
    if not valid:
        return None, "missing"
    if any(not math.isclose(converted[0], v, rel_tol=1e-12, abs_tol=1e-12) for v in converted[1:]):
        return None, "conflicting_values"
    return valid[0], "parsed"


def parse_record(answer: str, schema: dict) -> dict:
    text = text_form(answer)
    fields = {}
    issues = {}
    for field, unit in schema["units"].items():
        fields[field], issues[field] = collect_numeric(text, field, unit)
    if schema["family"] == "mineral_location":
        pair = re.search(r"[\[(]\s*(" + NUMBER + r")\s*,\s*(" + NUMBER + r")\s*[\])]", text)
        for index, field in enumerate(["latitude", "longitude"], 1):
            if pair and issues[field] == "missing":
                value = number_value(pair[index])
                bound = 90 if field == "latitude" else 180
                if -bound <= value <= bound:
                    fields[field] = Measurement(value, pair[index], "degree")
                    issues[field] = "parsed_pair"
        # Restrict text fields to their own clauses; an unlabeled leading name is accepted.
        site = re.search(r"(?:^|[;\n])\s*(?:site\s*name|site|矿点名称)\s*[:=]\s*([^;\n]+)", text)
        first = re.split(r"[;\n]", text)[0].strip()
        fields["site_name"] = (
            site[1].strip()
            if site
            else first
            if not re.search(r"[:=]|^\[|^\(|^\s*(?:latitude|longitude)", first)
            else None
        )
        dep = re.findall(r"\b(?:dep\s*id|deposit\s*id)\s*[:=]\s*([a-z0-9-]+)", text)
        fields["dep_id"] = dep[0] if dep and len(set(dep)) == 1 else None
        issues["site_name"] = "parsed" if fields["site_name"] else "missing"
        issues["dep_id"] = "parsed" if fields["dep_id"] else "missing_or_conflicting"
    context_ok = True
    if "expected_sample_depth_km" in schema:
        depths = re.findall(
            r"(?:dlnvs\s+at|dlnvs\s*\(|sample\s+depth\s*[:=])\s*(" + NUMBER + r")\s*km", text
        )
        context_ok = all(
            math.isclose(number_value(d), schema["expected_sample_depth_km"], abs_tol=1e-8)
            for d in depths
        )
    return {"fields": fields, "parse_status": issues, "context_ok": context_ok}


def choice_letter(text: str, allowed: list[str]) -> str | None:
    options = "".join(re.escape(x) for x in allowed)
    value = str(text or "").strip()
    match = re.fullmatch(r"\(?([" + options + r"])\)?[.。]?", value, re.I)
    if match:
        return match[1].upper()
    labels = re.findall(
        r"(?:answer|option|choice|答案|选项|选择)\s*(?:is|为|是)?\s*[:：]?\s*(["
        + options
        + r"])(?![a-z])",
        value,
        re.I,
    )
    return labels[0].upper() if labels and len({x.upper() for x in labels}) == 1 else None


def score_compatible(task, answer: str, format_spec: dict) -> dict:
    original = legacy_score(task, answer)
    schema = format_spec.get("scoring_schema")
    if task.answer_type == "选择题选项":
        allowed = format_spec.get("allowed_options", [])
        expected = choice_letter(task.expected_answer, allowed)
        actual = choice_letter(answer, allowed)
        return {
            "original": original,
            "normalized": {
                "score": float(expected is not None and actual == expected),
                "method": VERSION + "_choice",
                "details": {"expected": expected, "actual": actual, "allowed": allowed},
            },
            "compatibility_applied": True,
        }
    if not schema:
        return {
            "original": original,
            "normalized": original,
            "compatibility_applied": False,
            "fallback_reason": "No typed schema for this family; frozen original scoring preserved.",
        }
    expected = parse_record(task.expected_answer, schema)
    actual = parse_record(answer, schema)
    if any(expected["fields"].get(f) is None for f in schema["required_fields"]):
        return {
            "original": original,
            "normalized": None,
            "compatibility_applied": False,
            "needs_review": True,
            "reason": "Reference could not be parsed completely; no invented semantic score.",
            "reference_parse_status": expected["parse_status"],
        }
    details = []
    for field in schema["required_fields"]:
        a = actual["fields"].get(field)
        e = expected["fields"][field]
        matched = False
        value = None
        tolerance = None
        if isinstance(e, Measurement):
            canonical = schema["units"][field]
            target = convert(e, canonical, field)
            if isinstance(a, Measurement):
                value = convert(a, canonical, field)
                tolerance = (
                    reference_tolerance(e.raw, canonical, task.question)
                    * UNITS[e.unit][1]
                    / UNITS[normalized(canonical)][1]
                )
                matched = (
                    value is not None
                    and target is not None
                    and math.isclose(value, target, rel_tol=1e-7, abs_tol=tolerance + 1e-12)
                )
            serialized_expected = {"value": target, "unit": canonical}
        else:
            serialized_expected = e
            value = a
            matched = isinstance(a, str) and a.strip(" .") == e.strip(" .")
        if field == "dlnvs" and not actual["context_ok"]:
            matched = False
        details.append(
            {
                "field": field,
                "expected": serialized_expected,
                "actual": value,
                "matched": matched,
                "tolerance": tolerance,
                "parse_status": actual["parse_status"].get(field),
            }
        )
    score = sum(d["matched"] for d in details) / len(details)
    return {
        "original": original,
        "normalized": {
            "score": round(score, 6),
            "method": VERSION + "_" + schema["family"],
            "details": details,
            "context_ok": actual["context_ok"],
        },
        "compatibility_applied": True,
    }
