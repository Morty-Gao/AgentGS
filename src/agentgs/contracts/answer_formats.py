"""Public answer contract decoding shared by agents and runtime adapters."""

import hashlib
import json
import re
from functools import lru_cache

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
