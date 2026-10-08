"""Public answer contracts. No gold values, scoring imports, or answer generation.

Exact question matching excludes only the evaluator's appended resource locator.
The final model call remains in tool_plan_executor.generate_output.
"""

import hashlib
import json
import math
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

RESOURCE_MARKER = "\n\nEvaluation input resource (authoritative benchmark locator; use it exactly when relevant, but retrieve and select the registered tools yourself): "
SLOT = re.compile("<([^<>\\n]+)>")
NUMBER = "[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[Ee][-+]?\\d+)?"


def canonical_question(question):
    return str(question).replace("\r\n", "\n").split(RESOURCE_MARKER, 1)[0].strip()


def validate_contract_rows(rows):
    ids = set()
    index = {}
    for row in rows:
        qid = row["question_id"]
        if qid in ids:
            raise ValueError("Duplicate answer-format question ID")
        ids.add(qid)
        question = row["question"]
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != row["question_sha256"]:
            raise ValueError("Answer-format question hash mismatch")
        spec = row["answer_format"]
        if spec.get("version") == "1.2":
            names = SLOT.findall(spec.get("header_template", "") + spec["template"])
            if len(names) != len(set(names)) or set(names) != set(spec["slots"]):
                raise ValueError("Ambiguous or incomplete placeholder declaration")
            if list(dict.fromkeys(names)) != spec["placeholder_names"]:
                raise ValueError("Placeholder order mismatch")
        key = canonical_question(question)
        if key in index and index[key]["answer_format"] != spec:
            raise ValueError("Conflicting answer formats for the same question")
        index.setdefault(key, row)
    return index


@lru_cache(maxsize=1)
def _decode_contracts(contents):
    return validate_contract_rows(
        [json.loads(line) for line in contents.decode("utf-8").splitlines() if line.strip()]
    )


def fixed_answer_prompt(question, evidence, contract):
    return (
        "【题目】\n"
        + str(question)
        + "\n\n【本题格式】\n"
        + contract["answer_format"]["llm_instruction"]
        + "\n\n【工具结果】\n"
        + str(evidence)
    )


def render_bound_format(contract, bindings=None, *, records=None):
    """Explicit named binding utility for callers/tests; never invokes the model.

    This helper is NOT a final-answer fallback. It does not select a source,
    infer missing values, check scientific correctness, or assert completion.
    """
    spec = contract["answer_format"]
    bindings = dict(bindings or {})

    def render(template, values):
        names = SLOT.findall(template)
        if set(values) != set(names):
            raise ValueError("Bindings must exactly match named fields: " + ", ".join(names))

        def replace(match):
            name = match[1]
            value = values[name]
            slot = spec["slots"][name]
            if value is None or value == "unknown":
                return "unknown"
            if slot["kind"] == "artifact_path" and isinstance(value, (list, tuple)):
                if not value:
                    raise ValueError("Explicit nonempty artifact paths required: " + name)
                value = " | ".join(("unknown" if p is None else str(p) for p in value))
            if slot["kind"] == "number":
                if (
                    isinstance(value, bool)
                    or not re.fullmatch(NUMBER, str(value))
                    or (not math.isfinite(float(value)))
                ):
                    raise ValueError("Non-finite/non-numeric binding: " + name)
                if "decimals" in slot:
                    return format(float(value), "." + str(slot["decimals"]) + "f")
            if "choices" in slot and str(value) not in slot["choices"]:
                raise ValueError("Invalid choice binding: " + name)
            text = str(value)
            if SLOT.search(text) or not text.strip():
                raise ValueError("Empty/unfilled binding: " + name)
            return text

        return SLOT.sub(replace, template)

    if spec.get("repeat"):
        if records is None or not records:
            raise ValueError("Explicit nonempty records required; no guessed/empty result")
        head = render(spec["header_template"], bindings) if spec.get("header_template") else ""
        if not spec.get("header_template") and bindings:
            raise ValueError("Unexpected global bindings")
        body = spec["record_separator"].join((render(spec["template"], r) for r in records))
        return head + "; " + body if head else body
    if records is not None:
        raise ValueError("This contract is not a repeated record")
    return render(spec["template"], bindings)


def _match_template(template, answer):
    chunks = []
    names = []
    last = 0
    for match in SLOT.finditer(template):
        chunks.append(re.escape(template[last : match.start()]))
        chunks.append("(.+?)")
        names.append(match[1])
        last = match.end()
    chunks.append(re.escape(template[last:]))
    matched = re.fullmatch("".join(chunks), answer, re.DOTALL)
    return dict(zip(names, matched.groups())) if matched else None


def check_fixed_answer(answer, contract):
    """Non-mutating format/existence checks; never a scientific correctness score.

    File existence is necessary, not proof of classification/labels/interactivity.
    No network requests, files written, model calls, retries, or score changes.
    """
    answer = str(answer or "").strip()
    spec = contract["answer_format"]
    issues = []
    missing_token = False
    if not answer:
        issues.append("empty_answer")
    if "```" in answer or re.search("(?m)^\\s*(?:FinalAnswer|IsCompleted):", answer):
        issues.append("unexpected_wrapper")
    if SLOT.search(answer):
        issues.append("unfilled_placeholder")
    if spec.get("allowed_options") and answer not in spec["allowed_options"]:
        issues.append("invalid_option_format")
    if spec.get("version") == "1.2":
        matched_fields = []
        remaining = answer
        if spec.get("header_template"):
            header, sep, remaining = answer.partition("; ")
            found = _match_template(spec["header_template"], header)
            if not sep or found is None:
                issues.append("missing_or_invalid_header")
            else:
                matched_fields.append(found)
        records = remaining.split(spec["record_separator"]) if spec.get("repeat") else [remaining]
        for record in records:
            found = _match_template(spec["template"], record)
            if found is None:
                issues.append("missing_field_or_literal_mismatch")
                continue
            matched_fields.append(found)
        for fields in matched_fields:
            for name, value in fields.items():
                slot = spec["slots"][name]
                value = value.strip()
                if value == "unknown":
                    continue
                if value in {"null", "NaN", "nan", "None"}:
                    missing_token = True
                    issues.append("missing_value_token:" + name)
                    continue
                if slot["kind"] == "number":
                    if not re.fullmatch(NUMBER, value) or not math.isfinite(float(value)):
                        issues.append("invalid_number:" + name)
                    elif "decimals" in slot and (
                        not re.fullmatch("[-+]?\\d+\\.\\d{" + str(slot["decimals"]) + "}", value)
                    ):
                        issues.append("invalid_precision:" + name)
                if "choices" in slot and value not in slot["choices"]:
                    issues.append("invalid_choice:" + name)
                if slot["kind"] == "date":
                    try:
                        if date.fromisoformat(value).isoformat() != value:
                            raise ValueError()
                    except ValueError:
                        issues.append("invalid_date:" + name)
                if slot["kind"] == "year_month" and (
                    not re.fullmatch("\\d{4}-(?:0[1-9]|1[0-2])", value)
                ):
                    issues.append("invalid_year_month:" + name)
                if slot["kind"] == "artifact_path":
                    for resource in value.split(" | "):
                        if resource == "unknown":
                            continue
                        try:
                            path = Path(resource)
                            if not path.is_file():
                                issues.append("missing_artifact:" + name)
                        except (OSError, ValueError):
                            issues.append("invalid_artifact_path:" + name)
    else:
        for label in re.findall("(?:^|[;\\n])\\s*([A-Za-z_][A-Za-z_ ]*):", spec["template"]):
            if not re.search("(?:^|[;\\n])\\s*" + re.escape(label) + "\\s*:", answer):
                issues.append("missing_field:" + label)
    return {
        "type": "final_answer_format_validation",
        "version": spec["version"],
        "valid": not issues,
        "issues": list(dict.fromkeys(issues)),
        "has_unknown": missing_token or bool(re.search("\\bunknown\\b", answer)),
        "model_answer_preserved": True,
        "validation_scope": "Public structure/types/units/precision and local file existence only; no factual or artifact-functionality verification",
    }
