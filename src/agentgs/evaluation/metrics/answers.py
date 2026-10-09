"""Deterministic answer scoring with explicit parsing and ambiguity diagnostics."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, replace
from types import SimpleNamespace

from agentgs.evaluation.metrics import legacy_answers
from agentgs.evaluation.metrics.answer_parsing import (
    convert,
    dates,
    dms,
    final_text,
    json_safe,
    negated,
    normalized,
    quantities,
    quantity_match,
    requested_scalar,
    scalar,
    tolerance,
    urls,
)
from agentgs.evaluation.metrics.compatibility import score_compatible

ANSWER_PROTOCOL = "geotooleval-answer-v2"


def result(score, method, **details):
    return {
        "score": round(float(score), 6),
        "method": method,
        "protocol": ANSWER_PROTOCOL,
        "details": json_safe(details),
    }


def choice_options(question):
    tail = re.split(r"\bOptions\s*:", question, maxsplit=1, flags=re.I)[-1]
    return {
        m[1].upper(): m[2].strip()
        for m in re.finditer(
            r"(?:^|[;\n]|\s)([A-Z])[.)、]\s*(.*?)(?=(?:[;\n]|\s)[A-Z][.)、]\s|$)",
            tail,
            re.S,
        )
    }


def choice(text, options):
    text = final_text(text)
    allowed = "".join(options) or "ABCD"
    letter = "([" + re.escape(allowed) + "])"
    direct = re.fullmatch(r"\(?" + letter + r"\)?[.。]?", text, re.I)
    if direct:
        return direct[1].upper(), "letter"
    if re.search(r"\b[A-Z]\s*(?:or|/|或者|或)\s*(?:option\s+)?[A-Z]\b", text, re.I):
        return None, "ambiguous_choices"
    # A final answer assertion has precedence over discussion of rejected options.
    assertions = re.findall(
        r"(?:\b(?:final\s+answer|answer|my\s+choice)\s*(?:is\s*)?[:=]?\s*|(?:答案|选择)\s*[为是:：]?\s*)"
        r"(?:option\s+)?\(?" + letter + r"\)?(?![a-z])",
        text,
        re.I,
    )
    if assertions:
        unique = {x.upper() for x in assertions}
        if len(unique) == 1 and not negated(text):
            return unique.pop(), "answer_assertion"
        # Negation attached to an earlier option does not negate the answer clause.
        clauses = re.split(r"[.!?;\n]", text)
        positive = [c for c in clauses if re.search(r"\banswer\b|答案", c, re.I)]
        if len(unique) == 1 and positive and all(not negated(c) for c in positive):
            return unique.pop(), "answer_assertion"
        return None, "conflicting_or_negated_choices"
    labels = []
    for clause in re.split(r"[.!?;\n]", text):
        if negated(clause):
            continue
        labels.extend(
            re.findall(r"\b(?:option|choice)\s*[:=]?\s*" + letter + r"(?![a-z])", clause, re.I)
        )
        match = re.fullmatch(r"\s*\(?" + letter + r"\)?\s*", clause, re.I)
        if match:
            labels.append(match[1])
    if labels:
        unique = {x.upper() for x in labels}
        return (unique.pop(), "option_label") if len(unique) == 1 else (None, "ambiguous_choices")
    if negated(text):
        return None, "negated"
    matched = [key for key, value in options.items() if normalized(text) == normalized(value)]
    if not matched:
        actual, _ = scalar(text)
        if actual:
            # Exact numeric option equivalence only; never choose the nearest option.
            for key, value in options.items():
                expected, _ = scalar(value)
                value = convert(actual, expected.unit) if expected else None
                if (
                    expected
                    and value is not None
                    and math.isclose(expected.value, value, rel_tol=1e-9, abs_tol=1e-12)
                ):
                    matched.append(key)
    return (
        (matched[0], "unique_option_value")
        if len(matched) == 1
        else (None, "missing_or_ambiguous_choice")
    )


def scalar_score(task, expected, predicted):
    template = (getattr(task, "answer_format", None) or {}).get("template", "")
    predicted = requested_scalar(predicted, template)
    target, reference_status = scalar(expected)
    actual, status = scalar(predicted)
    if target and target.unit is None:
        units = quantities(re.sub(r"<[^<>]+>", "1", template))
        if len(units) == 1 and units[0].unit:
            target = replace(target, unit=units[0].unit)
    matched = bool(target and actual and quantity_match(target, actual, task.question))
    return result(
        matched,
        "scalar_quantity",
        reference_parse_status=reference_status,
        parse_status=status,
        expected=asdict(target) if target else None,
        actual=asdict(actual) if actual else None,
        tolerance=tolerance(target.raw, task.question) if target else None,
    )


def typed_schema(task):
    """Select supported quantity roles from the public task, never prediction values."""
    template = (getattr(task, "answer_format", None) or {}).get("template", "").lower()
    question = task.question.lower()
    if all(name in template or name in question for name in ("moho", "lid-bottom", "dlnvs")):
        schema = {
            "family": "earth_measurements",
            "required_fields": ["moho", "lid_bottom", "dlnvs"],
            "units": {"moho": "km", "lid_bottom": "km", "dlnvs": "%"},
        }
        depth = re.search(r"dlnvs\s+at\s+(\d+(?:\.\d+)?)\s*km", question)
        if depth:
            schema["expected_sample_depth_km"] = float(depth[1])
        return schema
    if all(name in template for name in ("valid_count:", "mean_fco2:", "mean_sst:")):
        return {
            "family": "ocean_summary",
            "required_fields": ["valid_count", "mean_fco2", "mean_sst"],
            "units": {"valid_count": "count", "mean_fco2": "μatm", "mean_sst": "°C"},
        }
    if re.search(r"(?:^|;\s*)(?:mean_)?temperature_anomaly:", template):
        fields = {"temperature_anomaly": "°C"}
        if "age:" in template:
            fields["age"] = "kyr BP"
        if "sea_level:" in template:
            fields["sea_level"] = "m"
        return {"family": "paleo_measurements", "required_fields": list(fields), "units": fields}
    return None


def labelled_numbers(text):
    """Parse semicolon/JSON quantity fields without merging values across fields."""
    text = final_text(text)
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        parts = [(str(key), str(value)) for key, value in obj.items()]
    else:
        parts = []
        for part in re.split(r"[;\n]", text):
            match = re.fullmatch(r"\s*([^:=]+?)\s*[:=]\s*(.+?)\s*", part)
            if not match:
                return None
            parts.append((match[1], match[2]))
    fields = {}
    for label, value in parts:
        label = re.sub(r"\bfrom point\d+ to point\d+\b", "", normalized(label))
        label = re.sub(r"[\s_-]+", " ", label).strip()
        quantity, status = scalar(value)
        if label in fields or status != "parsed":
            return None
        fields[label] = quantity
    return fields or None


def multi_number(task, expected, predicted):
    if negated(predicted):
        return result(0, "numeric_record", parse_status="negated")
    targets = labelled_numbers(expected)
    actual = labelled_numbers(predicted)
    if targets:
        actual = actual or {}
        matched = sum(
            key in actual and quantity_match(value, actual[key], task.question)
            for key, value in targets.items()
        )
        score = 2 * matched / (len(targets) + len(actual))
        return result(
            score,
            "labelled_numeric_f1",
            expected_fields=list(targets),
            actual_fields=list(actual),
            matched_fields=matched,
        )
    # Unlabelled tuples retain their order; extra numbers cannot be ignored.
    targets, actual = quantities(expected), quantities(predicted)
    matched = sum(quantity_match(a, b, task.question) for a, b in zip(targets, actual))
    score = 2 * matched / (len(targets) + len(actual)) if targets or actual else 0
    return result(
        score,
        "ordered_numeric_f1",
        expected=[asdict(x) for x in targets],
        actual=[asdict(x) for x in actual],
        matched=matched,
    )


def text_items(text, *, allow_commas=False):
    text = final_text(text)
    try:
        value = json.loads(text)
    except ValueError:
        value = None

    def item(value):
        value = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s+", "", str(value))
        value = re.sub(r"^site[_ ]name\s*:\s*", "", value, flags=re.I)
        return re.sub(r"\s*[:=]\s*", ":", normalized(value))

    if isinstance(value, list):
        return [item(x) for x in value]
    parts = [x for x in re.split(r"[;\n]+", text) if x.strip()]
    if allow_commas and len(parts) == 1:
        parts = [x for x in text.split(",") if x.strip()]
    return [item(x) for x in parts]


def score_answer(task, final_answer):
    expected, predicted = final_text(task.expected_answer), final_text(final_answer)
    kind = task.answer_type
    if not predicted:
        return result(0, "empty_answer", parse_status="empty")
    if kind == "choice":
        options = choice_options(task.question)
        target, _ = choice(expected, options)
        actual, status = choice(predicted, options)
        return result(
            target is not None and target == actual,
            "choice",
            expected_choice=target,
            observed_choice=actual,
            parse_status=status,
        )
    if kind == "date":
        target, actual = dates(expected), dates(predicted)
        return result(
            len(target) == 1
            and target == actual
            and "invalid_date" not in actual
            and not negated(predicted),
            "calendar_date",
            expected_dates=target,
            actual_dates=actual,
        )
    if kind == "identifier":
        target, actual = urls(expected), urls(predicted)
        identifiers = []
        if target:
            matched = len(target) == 1 and target == actual and not negated(predicted)
        else:
            # Identifier text is case-sensitive and must be a bounded token.
            tokens = re.findall(r"[\w]+(?:[-./][\w]+)*", predicted)
            identifiers = [x for x in tokens if any(c.isdigit() for c in x)]
            matched = identifiers == [expected] if identifiers else predicted == expected
            matched = matched and not negated(predicted)
        return result(
            matched,
            "identifier",
            expected_identifier=target or [expected],
            actual_identifiers=actual if target else identifiers,
        )
    schema = typed_schema(task) if kind in {"structured_record", "multi_number"} else None
    if schema:
        if negated(predicted):
            return result(0, "typed_record", schema=schema, parse_status="negated")
        reference = expected
        if schema["required_fields"] == ["temperature_anomaly"]:

            def temperature_record(text):
                text = re.sub(r"\(\s*n\s*=\s*\d+\s*\)", "", text, flags=re.I).strip()
                if re.fullmatch(r"[-+\d.eE\s°CcKk]+", text):
                    return "temperature_anomaly: " + text
                return text

            reference, predicted = temperature_record(reference), temperature_record(predicted)
        typed_task = SimpleNamespace(
            expected_answer=reference, question=task.question, answer_type=task.answer_type
        )
        scored = score_compatible(typed_task, predicted, {"scoring_schema": schema})
        normalized_score = scored["normalized"]
        if normalized_score is None:
            return result(
                0,
                "typed_record",
                needs_review=True,
                reference_parse_status=scored["reference_parse_status"],
            )
        return result(
            normalized_score["score"],
            "typed_record",
            schema=schema,
            fields=normalized_score["details"],
            context_ok=normalized_score["context_ok"],
        )
    if kind == "single_number":
        return scalar_score(task, expected, predicted)
    if kind == "multi_number":
        return multi_number(task, expected, predicted)
    if kind == "single_text":
        target_dms = dms(expected)
        if target_dms:
            actual = dms(predicted)
            matched = (
                len(target_dms) == len(actual) == 1
                and not negated(predicted)
                and target_dms[0][1] == actual[0][1]
                and math.isclose(
                    target_dms[0][0],
                    actual[0][0],
                    rel_tol=0.0,
                    abs_tol=tolerance(target_dms[0][2], "") + 1e-9,
                )
            )
            return result(matched, "dms_coordinate", expected=target_dms, actual=actual)
        target, _ = scalar(expected)
        if target and target.unit is not None:
            return scalar_score(task, expected, predicted)
        target, actual = normalized(expected), normalized(predicted)
        matched = not negated(predicted) and bool(
            re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", actual)
        )
        return result(
            matched,
            "bounded_text",
            expected=target,
            actual=actual,
            scope="Lexical match; paraphrases and general semantic correctness are not inferred",
        )
    if kind == "multi_text":
        targets = text_items(expected)
        actual = text_items(predicted, allow_commas=len(targets) > 1 and "," not in expected)
        remaining = list(actual)
        matched = 0
        for target in targets:
            if target in remaining:
                matched += 1
                remaining.remove(target)
        score = 2 * matched / (len(targets) + len(actual)) if targets or actual else 0
        return result(
            score,
            "text_item_f1",
            expected_items=targets,
            actual_items=actual,
            matched_items=matched,
            scope="Lexical item equality, preserving duplicates",
        )
    if kind == "structured_record":
        scored = legacy_answers.score_answer(task, predicted)
        return result(
            scored["score"],
            "lexical_record_fallback",
            **scored["details"],
            needs_review=True,
            scope="No typed schema; lexical field overlap is not semantic verification",
        )
    return result(normalized(expected) == normalized(predicted), "normalized_exact")
