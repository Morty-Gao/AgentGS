"""Deterministic lexical answer metrics for historical report comparisons."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any

from agentgs.evaluation.datasets.base import Example

NUMBER_RE = re.compile("(?<![A-Za-z0-9.])[-+]?\\d+(?:\\.\\d+)?(?:[Ee][-+]?\\d+)?")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.casefold().strip()
    text = re.sub("\\s+", " ", text)
    return text.strip(" \t\r\n.,;:")


def extract_numbers(value: Any) -> list[tuple[float, str]]:
    text = str(value or "").replace(",", "")
    text = re.sub("\\bM(?=\\d)", "M ", text, flags=re.IGNORECASE)
    result: list[tuple[float, str]] = []
    for match in NUMBER_RE.finditer(text):
        raw = match.group(0)
        try:
            result.append((float(raw), raw))
        except ValueError:
            continue
    return result


def explicit_rounding_digits(question: str) -> int | None:
    text = normalize_text(question)
    match = re.search("round(?:ed)? to (\\d+) decimal", text)
    if match:
        return int(match.group(1))
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    for word, count in words.items():
        if re.search(f"round(?:ed)? to {word} decimal", text):
            return count
    return None


def numeric_tolerance(raw_expected: str, question: str) -> float:
    explicit = explicit_rounding_digits(question)
    if explicit is not None:
        return 0.5 * 10.0 ** (-explicit)
    mantissa = re.split("[Ee]", raw_expected.lstrip("+-"), maxsplit=1)[0]
    decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    return 0.0 if decimals == 0 else 0.5 * 10.0 ** (-decimals)


def numbers_match(expected: tuple[float, str], actual: tuple[float, str], question: str) -> bool:
    tolerance = numeric_tolerance(expected[1], question)
    return math.isclose(expected[0], actual[0], rel_tol=1e-07, abs_tol=tolerance + 1e-12)


def item_f1(expected_items: list[str], actual_items: list[str]) -> tuple[float, dict[str, Any]]:
    expected = [normalize_text(item) for item in expected_items if normalize_text(item)]
    actual = [normalize_text(item) for item in actual_items if normalize_text(item)]
    remaining = list(actual)
    matched = 0
    for item in expected:
        candidate = next(
            (
                index
                for index, observed in enumerate(remaining)
                if item == observed or item in observed or observed in item
            ),
            None,
        )
        if candidate is not None:
            matched += 1
            remaining.pop(candidate)
    precision = matched / len(actual) if actual else 0.0
    recall = matched / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        f1,
        {
            "expected_items": expected,
            "actual_items": actual,
            "matched_items": matched,
            "precision": precision,
            "recall": recall,
        },
    )


def split_items(value: str, allow_commas: bool = False) -> list[str]:
    text = str(value or "").strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    pattern = "[;；\\n]+"
    items = [
        re.sub("^\\s*(?:[-*]|\\d+[.)、])\\s*", "", item).strip() for item in re.split(pattern, text)
    ]
    items = [item for item in items if item]
    if allow_commas and len(items) <= 1:
        comma_items = [item.strip() for item in re.split("[,，]+", text) if item.strip()]
        if len(comma_items) > 1:
            return comma_items
    return items


def token_f1(expected: str, actual: str) -> tuple[float, dict[str, Any]]:
    token_pattern = re.compile("https?://[^\\s;,]+|[a-z0-9_./%°μ≈+-]+", re.IGNORECASE)
    expected_tokens = token_pattern.findall(normalize_text(expected))
    actual_tokens = token_pattern.findall(normalize_text(actual))
    expected_counter = Counter(expected_tokens)
    actual_counter = Counter(actual_tokens)
    matched = sum((min(count, actual_counter[token]) for token, count in expected_counter.items()))
    precision = matched / sum(actual_counter.values()) if actual_counter else 0.0
    recall = matched / sum(expected_counter.values()) if expected_counter else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        f1,
        {
            "expected_tokens": expected_tokens,
            "actual_tokens": actual_tokens,
            "matched_tokens": matched,
            "precision": precision,
            "recall": recall,
        },
    )


def structured_field_score(
    expected: str, actual: str, question: str
) -> tuple[float, dict[str, Any]]:
    fields = [item.strip() for item in re.split("[;；,，]+", expected) if item.strip()]
    actual_norm = normalize_text(actual)
    actual_numbers = extract_numbers(actual)
    word_pattern = re.compile(
        "https?://[^\\s;,]+|[a-z][a-z0-9_-]*|[\\u4e00-\\u9fff]+", re.IGNORECASE
    )
    ignored_words = {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
    field_details: list[dict[str, Any]] = []
    for field in fields:
        expected_numbers = extract_numbers(field)
        number_matches = sum(
            (
                1
                for expected_number in expected_numbers
                if any(
                    (
                        numbers_match(expected_number, observed, question)
                        for observed in actual_numbers
                    )
                )
            )
        )
        field_without_numbers = NUMBER_RE.sub(" ", normalize_text(field))
        expected_words = [
            word
            for word in word_pattern.findall(field_without_numbers)
            if normalize_text(word) not in ignored_words
        ]
        word_matches = sum((1 for word in expected_words if normalize_text(word) in actual_norm))
        atom_count = len(expected_numbers) + len(expected_words)
        matched_atoms = number_matches + word_matches
        field_score = matched_atoms / atom_count if atom_count else 0.0
        field_details.append(
            {
                "field": field,
                "expected_numbers": expected_numbers,
                "number_matches": number_matches,
                "expected_words": expected_words,
                "word_matches": word_matches,
                "score": field_score,
            }
        )
    score = (
        sum((item["score"] for item in field_details)) / len(field_details)
        if field_details
        else 0.0
    )
    return (
        score,
        {
            "expected_field_count": len(field_details),
            "actual_numbers": actual_numbers,
            "fields": field_details,
        },
    )


def score_answer(task: Example, final_answer: str) -> dict[str, Any]:
    expected = task.expected_answer
    answer_type = task.answer_type
    predicted = str(final_answer or "")
    expected_norm = normalize_text(expected)
    predicted_norm = normalize_text(predicted)
    if not predicted_norm:
        return {"score": 0.0, "method": "empty_answer", "details": {}}
    if answer_type == "choice":
        expected_match = re.search("(?<![A-Za-z])([A-D])(?![A-Za-z])", expected, re.IGNORECASE)
        expected_choice = (
            expected_match.group(1).upper() if expected_match else expected.strip().upper()
        )
        explicit_patterns = (
            "(?:answer|option|choice)\\s*(?:is)?\\s*:?\\s*([A-D])\\b",
            "^\\s*([A-D])(?:\\b|[.:)])",
        )
        observed = None
        for pattern in explicit_patterns:
            match = re.search(pattern, predicted, re.IGNORECASE)
            if match:
                observed = match.group(1).upper()
                break
        if observed is None:
            matches = re.findall("(?<![A-Za-z])([A-D])(?![A-Za-z])", predicted, re.IGNORECASE)
            observed = matches[-1].upper() if matches else None
        score = 1.0 if observed == expected_choice else 0.0
        return {
            "score": score,
            "method": "normalized_choice",
            "details": {"expected_choice": expected_choice, "observed_choice": observed},
        }
    if answer_type == "date":

        def dates(text: str) -> list[str]:
            values = []
            for year, month, day in re.findall(
                "(?<!\\d)(\\d{4})\\s*[-/.]\\s*(\\d{1,2})\\s*[-/.]\\s*(\\d{1,2})(?!\\d)",
                text,
            ):
                values.append(f"{int(year):04d}-{int(month):02d}-{int(day):02d}")
            return values

        expected_dates = dates(expected)
        actual_dates = dates(predicted)
        score = 1.0 if expected_dates and expected_dates[0] in actual_dates else 0.0
        return {
            "score": score,
            "method": "normalized_date",
            "details": {"expected_dates": expected_dates, "actual_dates": actual_dates},
        }
    if answer_type == "single_number":
        expected_numbers = extract_numbers(expected)
        actual_numbers = extract_numbers(predicted)
        matched = bool(
            expected_numbers
            and any(
                (
                    numbers_match(expected_numbers[0], actual, task.question)
                    for actual in actual_numbers
                )
            )
        )
        return {
            "score": 1.0 if matched else 0.0,
            "method": "single_numeric_tolerance",
            "details": {
                "expected_numbers": expected_numbers,
                "actual_numbers": actual_numbers,
                "tolerance": numeric_tolerance(expected_numbers[0][1], task.question)
                if expected_numbers
                else None,
            },
        }
    if answer_type == "multi_number":
        expected_numbers = extract_numbers(expected)
        actual_numbers = extract_numbers(predicted)
        remaining = list(actual_numbers)
        matched = 0
        for expected_number in expected_numbers:
            candidate = next(
                (
                    index
                    for index, actual_number in enumerate(remaining)
                    if numbers_match(expected_number, actual_number, task.question)
                ),
                None,
            )
            if candidate is not None:
                matched += 1
                remaining.pop(candidate)
        precision = matched / len(actual_numbers) if actual_numbers else 0.0
        recall = matched / len(expected_numbers) if expected_numbers else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "score": round(f1, 6),
            "method": "multi_numeric_f1",
            "details": {
                "expected_numbers": expected_numbers,
                "actual_numbers": actual_numbers,
                "matched_numbers": matched,
                "precision": precision,
                "recall": recall,
            },
        }
    if answer_type == "identifier":
        expected_urls = re.findall("https?://[^\\s;,]+", expected, re.IGNORECASE)
        actual_urls = re.findall("https?://[^\\s;,]+", predicted, re.IGNORECASE)
        observed_values = [normalize_text(value) for value in actual_urls]
        target = normalize_text(expected_urls[0] if expected_urls else expected)
        score = 1.0 if target in observed_values or target == predicted_norm else 0.0
        return {
            "score": score,
            "method": "normalized_identifier",
            "details": {"expected_identifier": target, "actual_identifiers": observed_values},
        }
    if answer_type == "multi_text":
        expected_items = split_items(expected)
        actual_items = split_items(
            predicted, allow_commas=not any(("," in item for item in expected_items))
        )
        score, details = item_f1(expected_items, actual_items)
        return {"score": round(score, 6), "method": "multi_text_item_f1", "details": details}
    if answer_type == "structured_record":
        if expected_norm == predicted_norm:
            score, details = (1.0, {"exact_normalized_match": True})
        else:
            score, details = structured_field_score(expected, predicted, task.question)
            details["exact_normalized_match"] = False
        return {
            "score": round(score, 6),
            "method": "structured_record_field_score",
            "details": details,
        }
    if answer_type == "single_text":
        score = 1.0 if expected_norm == predicted_norm or expected_norm in predicted_norm else 0.0
        return {
            "score": score,
            "method": "normalized_single_text",
            "details": {"expected_normalized": expected_norm, "actual_normalized": predicted_norm},
        }
    return {
        "score": 1.0 if expected_norm == predicted_norm else 0.0,
        "method": "normalized_exact_fallback",
        "details": {"expected_normalized": expected_norm, "actual_normalized": predicted_norm},
    }
