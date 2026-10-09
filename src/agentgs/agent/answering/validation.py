"""Versioned non-mutating evidence validation. No model or tool execution.

Checks returned facts, not general semantic correctness. In particular, an empty
unsupported-number list cannot prove the requested scientific task was solved.
"""

import json
import math
import re

VERSION = "returned-evidence-v3.3"
NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
PATH_KEY = re.compile(
    r"(?:^|_)(?:paths?|files?|url|uri|directory|source|artifact|image|raster|id)$", re.I
)


def normalize(value):
    return (
        str(value)
        .replace("−", "-")
        .replace("–", "-")
        .replace("’", "'")
        .replace("′", "'")
        .replace("″", '"')
        .casefold()
    )


def decode(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return value


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, path + (str(key),))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from leaves(child, path + (str(index),))
    else:
        yield path, value


def tokens(text):
    # Filenames and identifiers are labels, not independent measurement claims.
    text = re.sub(r"https?://\S+|[A-Za-z]:[\\/][^\s`<>]+", " ", normalize(text))
    text = re.sub(r"\b[\w.-]+\.(?:csv|tif|tiff|json|png|nc|xlsx)\b", " ", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", " ", text)
    text = re.sub(
        r"(?<![\w.])([-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?![\d,])",
        lambda match: match.group().replace(",", ""),
        text,
    )
    date_parts = [
        part for match in re.findall(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text) for part in match
    ]
    text = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " ", text)
    return date_parts + [m.group() for m in NUMBER.finditer(text)]


def close_reported(token, actual):
    """Permit the precision actually reported, but integer zero is not near-zero."""
    try:
        value = float(token)
        if not math.isfinite(value) or not math.isfinite(actual):
            return False
        literal = str(token).casefold()
        if "." in literal:
            mantissa, _, exponent = literal.partition("e")
            precision = len(mantissa.split(".")[1])
            tolerance = 0.500001 * 10 ** (int(exponent or 0) - precision)
        else:
            tolerance = 1e-9
        return math.isclose(value, actual, rel_tol=1e-9, abs_tol=max(1e-12, tolerance))
    except (ValueError, TypeError, OverflowError):
        return False


def terminal_measure(outputs):
    """Use the terminal returned scalar, never an option or earlier input count."""
    if not outputs:
        return None
    data = decode(outputs[-1])
    candidates = [
        (path, value)
        for path, value in leaves(data)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and not (path and PATH_KEY.search(path[-1]))
    ]
    for preferred in ("slope", "sen_slope", "sens_slope", "value", "result", "distance_km", "mean"):
        matches = [
            (path, value) for path, value in candidates if path and path[-1].casefold() == preferred
        ]
        if len(matches) == 1:
            path, value = matches[0]
            return {"field": ".".join(path), "value": float(value)}
    if len(candidates) == 1:
        path, value = candidates[0]
        return {"field": ".".join(path), "value": float(value)}
    return None


def option_contract(question, answer, outputs):
    split = re.split(r"(?:Options|选项)\s*[:：]", question, maxsplit=1, flags=re.I)
    if len(split) == 1:
        return {"present": False, "valid": True}
    options = {
        label: body.strip()
        for label, body in re.findall(
            r"(?:^|[；;\n]|\s)([A-H])[.、)]\s*(.*?)(?=(?:[；;\n]|\s)[A-H][.、)]\s|$)",
            split[1],
            re.S,
        )
    }
    terminal = terminal_measure(outputs)
    supported = []
    option_values = {}
    for label, body in options.items():
        if terminal and "slope" in terminal["field"].casefold():
            match = re.search(
                r"(?:slope|斜率)(?:\s+(?:of|is))?\s*[:=]?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                body,
                re.I,
            )
            values = [match.group(1)] if match else []
        else:
            values = tokens(body)
        option_values[label] = values
        if terminal and len(values) == 1 and close_reported(values[0], terminal["value"]):
            supported.append(label)
    rejection = bool(
        re.search(
            r"none of (?:the )?(?:listed |provided |given )?options|no (?:listed |provided |given )?option.{0,30}(?:supported|matches|consistent)|没有.{0,12}选项.{0,12}(?:支持|符合)|均不符合",
            answer,
            re.I,
        )
    )
    selected = re.search(
        r"^\s*(?:\*\*|`)?(?:answer\s*[:：]?\s*|option\s+|答案\s*[:：]?\s*)?([A-H])(?:[.、)\s`*:：]|$)",
        answer,
        re.I,
    )
    if not selected and not rejection:
        selected = re.search(
            r"(?:choose|select|correct (?:answer|option) is|答案(?:是|为))\s*[:：]?\s*([A-H])\b",
            answer,
            re.I,
        )
    selected = selected.group(1).upper() if selected else None
    actual_reported = bool(
        terminal and any(close_reported(token, terminal["value"]) for token in tokens(answer))
    )
    if rejection:
        valid = bool(terminal and not supported and actual_reported)
        reason = (
            "honest_no_supported_option_with_terminal_value"
            if valid
            else "unsupported_option_rejection"
        )
    elif selected:
        valid = selected in supported
        reason = (
            "selected_option_matches_terminal_value" if valid else "unsupported_selected_option"
        )
    else:
        valid = bool(terminal and actual_reported)
        reason = (
            "terminal_value_reported_without_letter"
            if valid
            else "option_or_terminal_claim_unverified"
        )
    return {
        "present": True,
        "valid": valid,
        "reason": reason,
        "terminal_measure": terminal,
        "selected_option": selected,
        "supported_options": supported,
        "rejected_all_options": rejection,
        "terminal_value_present_in_answer": actual_reported,
        "option_values": option_values,
    }


def validate_returned_evidence(
    question, answer, outputs, *, completed=True, legacy=None, suggested=None
):
    outputs = list(outputs or [])
    text = normalize(answer)
    allowed = []
    strings = []
    paths = []
    identifiers = []
    for output in outputs:
        for path, value in leaves(decode(output)):
            if isinstance(value, str):
                strings.append(normalize(value))
                # Acquisition dates embedded in returned filenames are explicit
                # date evidence. Other path/version/index digits remain excluded.
                for date in re.findall(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", value):
                    try:
                        from datetime import date as date_type

                        parsed = date_type.fromisoformat(date)
                        allowed.extend([float(parsed.year), float(parsed.month), float(parsed.day)])
                    except ValueError:
                        pass
            if path and re.search(r"(?:^|_)id$", path[-1], re.I) and isinstance(value, (str, int)):
                identifiers.append((path[-1], str(value)))
            if path and PATH_KEY.search(path[-1]):
                continue
            if (
                isinstance(value, (float, int))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                allowed.append(float(value))
                paths.append(".".join(path))
            elif isinstance(value, str):
                # Dates, DMS and other scientific strings are genuine returned facts.
                allowed.extend(float(token) for token in tokens(value))
    context = re.split(r"(?:Options|选项)\s*[:：]", question, maxsplit=1, flags=re.I)[0]
    allowed.extend(float(token) for token in tokens(context))
    # Preserve exact returned representations (including dates and coordinates).
    for fragment in sorted(set(strings), key=len, reverse=True):
        if fragment and any(char.isdigit() for char in fragment):
            text = text.replace(fragment, " ")
    # Exact, explicitly labelled ID references are labels, not measurements.
    # An unlabelled copy of the same number remains unsupported.
    for label, value in identifiers:
        pattern = (
            r"\b"
            + re.escape(normalize(label))
            + r"\s*[:=]?\s*"
            + re.escape(normalize(value))
            + r"(?![\w.])"
        )
        text = re.sub(pattern, " ", text)
    option = option_contract(question, answer, outputs)
    if option.get("rejected_all_options") and option.get("valid"):
        # A rejected option is a quotation of a hypothesis, not an asserted result.
        text = re.sub(
            r"[^.!?\n]*(?:not consistent with options|options.{0,20}(?:incorrect|unsupported)|选项.{0,20}不符合)[^\n]*",
            " ",
            text,
            flags=re.I,
        )
    unsupported = [
        float(token)
        for token in tokens(text)
        if not any(close_reported(token, value) for value in allowed)
    ]
    valid = bool(
        completed and str(answer).strip() and outputs and not unsupported and option["valid"]
    )
    event = {
        "type": "final_answer_validation",
        "status": "passed" if valid else "failed",
        # Preserve the frozen component oracle's metadata. The runtime service
        # removes this historical marker and records the actual planning policy.
        "evaluation_llm_required": True,
        "validator_version": VERSION,
        "valid": valid,
        "model_answer_preserved": True,
        "unsupported_numeric_claims": unsupported,
        "observed_numeric_leaf_count": len(paths),
        "option_evidence": option,
        "validation_scope": "Non-mutating returned-fact and terminal numeric-option check; not generic semantic accuracy or scientific task completion.",
    }
    if legacy is not None:
        event.update(
            legacy_validation=legacy,
            legacy_suggested_answer=suggested,
            legacy_would_rewrite=str(suggested).strip() != str(answer).strip(),
            legacy_status_is_diagnostic_only=True,
        )
    return str(answer).strip(), event
