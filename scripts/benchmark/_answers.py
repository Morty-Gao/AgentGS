"""Extract declared answers without consulting reference values or scoring rules."""

from __future__ import annotations

import json
import re

_MARKER = re.compile(
    r"^[ \t]*(?:\#{1,6}[ \t]+)?(?:\*\*|__)?[ \t]*"
    r"(?:final[ _-]*answer|final|answer|最终答案|答案)[ \t]*(?:\*\*|__)?"
    r"[ \t]*[:：=][ \t]*(?:\*\*|__)?[ \t]*",
    re.I | re.M,
)


def _unwrap(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    fence = re.fullmatch(r"```(?:text|plaintext|json)?[ \t]*\n([\s\S]*?)\n?```", text, re.I)
    return fence[1].strip() if fence else text


def extract_answer(text, *, allow_plain=False):
    """Plain text requires a caller-established final response boundary.

    Marked answers keep their full multiline body. Missing or conflicting JSON
    answer fields do not fall back to the surrounding diagnostic text.
    """
    text = _unwrap(str(text or ""))
    method, answer = "missing", ""
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        fields = [
            value
            for key, value in obj.items()
            if re.sub(r"[ _-]", "", key).lower() in {"answer", "finalanswer", "final"}
        ]
        if fields:
            values = [
                _unwrap(value)
                if isinstance(value, str)
                else ""
                if value is None
                else json.dumps(value, ensure_ascii=False)
                for value in fields
            ]
            if len(set(values)) != 1:
                return {"answer": "", "method": "conflicting_json_answers"}
            answer, method = values[0], "json_field"
            if answer and "unit" in obj:
                answer = json.dumps({"answer": fields[0], "unit": obj["unit"]}, ensure_ascii=False)
            return {"answer": answer, "method": method if answer else "empty_json_answer"}
    markers = list(_MARKER.finditer(text))
    if markers:
        answer, method = _unwrap(text[markers[-1].end() :]), "explicit_marker"
        # Wrappers can be nested, for example FINAL: followed by a JSON answer.
        nested = extract_answer(answer)
        if nested["method"] != "missing":
            answer = nested["answer"]
            if not answer:
                return nested
    elif allow_plain:
        answer, method = text, "plain_final_response"
    return {"answer": answer, "method": method if answer else "empty" if not text else "missing"}


def extract_program_answer(stdout, *, terminal=False, prefix_stdout=""):
    """Only output newly emitted by a completed plan is eligible for plain fallback."""
    extracted = extract_answer(stdout)
    if extracted["method"] != "missing":
        return extracted
    if not terminal:
        return extracted
    if not stdout.startswith(prefix_stdout):
        return {"answer": "", "method": "ambiguous_prefix_output"}
    extracted = extract_answer(stdout[len(prefix_stdout) :], allow_plain=True)
    if extracted["method"] == "plain_final_response":
        extracted["method"] = "terminal_stdout"
    return extracted
