"""Serialize every retrieved contract without changing candidate selection."""

import json

OPTIONAL_RELATIONS = frozenset({"documentation url", "homepage url", "citation url", "paper url"})


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def relation_parts(value):
    if isinstance(value, (list, tuple)) and len(value) == 3:
        subject, predicate, target = value
        if isinstance(subject, str) and isinstance(predicate, str):
            return subject, predicate, target
    if isinstance(value, dict) and set(value) == {"subject", "predicate", "object"}:
        if isinstance(value["subject"], str) and isinstance(value["predicate"], str):
            return value["subject"], value["predicate"], value["object"]
    return None


def build_candidate_context(references, max_chars=48000):
    budget = max(0, int(max_chars))
    groups, unknown, seen = {}, [], set()
    for value in references or []:
        parts = relation_parts(value)
        if parts is None:
            unknown.append(value)
            continue
        subject, predicate, target = parts
        signature = encode([subject, predicate, target])
        if signature not in seen:
            groups.setdefault(subject, {}).setdefault(predicate, []).append(target)
            seen.add(signature)

    def payload(records, omitted=0, exceeded=False):
        result = {"retrieved_subjects": records}
        if unknown:
            result["other_reference"] = unknown
        if omitted or exceeded:
            result["context_budget"] = {
                "soft_limit_chars": budget,
                "omitted_documentation_relations": omitted,
                "mandatory_context_exceeds_budget": exceeded,
                "all_subjects_and_contracts_preserved": True,
            }
        return encode(result)

    records = [
        {"subject": subject, "relations": relations} for subject, relations in groups.items()
    ]
    complete = payload(records)
    if len(complete) <= budget:
        return complete

    records, optional_groups = [], []
    for subject, relations in groups.items():
        required, optional = {}, []
        for predicate, values in relations.items():
            if predicate.strip().casefold() in OPTIONAL_RELATIONS:
                optional.extend((predicate, value) for value in values)
            else:
                required[predicate] = list(values)
        records.append({"subject": subject, "relations": required})
        optional_groups.append(optional)
    omitted = sum(len(values) for values in optional_groups)
    if len(payload(records, omitted)) > budget:
        return payload(records, omitted, True)
    for index in range(max((len(values) for values in optional_groups), default=0)):
        for record, optional in zip(records, optional_groups):
            if index >= len(optional):
                continue
            predicate, target = optional[index]
            values = record["relations"].setdefault(predicate, [])
            values.append(target)
            if len(payload(records, omitted - 1)) <= budget:
                omitted -= 1
            else:
                values.pop()
                if not values:
                    del record["relations"][predicate]
    return payload(records, omitted)
