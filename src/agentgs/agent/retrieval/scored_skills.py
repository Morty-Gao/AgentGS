"""The selected baseline's weighted lexical Skill retrieval algorithm.

Scores, tokenization, resource features, extension admission and tie breaking
are algorithm decisions, not tuning opportunities during this refactor. Flags
are explicit inputs rather than process-global environment reads.
"""

import json
import math
import re
from collections import Counter

from agentgs.agent.retrieval.skills import SkillMatch

RESOURCE_MARKER = (
    "Evaluation input resource (authoritative benchmark locator; use it exactly when "
    "relevant, but retrieve and select the registered tools yourself):"
)
STOPWORDS = frozenset(
    "a an and are as at be been by each for from has have how in into is it of on or "
    "that the their there these this those to was were what when where which who with".split()
)
SEMANTIC_TERMS = frozenset(
    "beach canal coast county district forest island lake lagoon locality municipality "
    "park reserve river spatial stream townland village water".split()
)


def split_resource(query: str) -> tuple[str, object]:
    if RESOURCE_MARKER not in query:
        return query.strip(), None
    body, raw = query.rsplit(RESOURCE_MARKER, 1)
    try:
        return body.strip(), json.loads(raw.strip())
    except json.JSONDecodeError:
        return body.strip(), raw.strip()


def lexical_tokens(value: str) -> set[str]:
    # Preserve the baseline's normalization order and CJK cross-word bigrams.
    # Altering either changes IDF/overlap even when it looks like parser cleanup.
    text = (value or "").casefold()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    english = re.findall(r"[a-z][a-z0-9]{1,}", re.sub(r"[_-]+", " ", text))
    cjk = re.findall(r"[\u3400-\u9fff]", text)
    bigrams = ["".join(cjk[i : i + 2]) for i in range(len(cjk) - 1)]
    return {word for word in (*english, *bigrams) if len(word) > 1 and word not in STOPWORDS}


def extension_eligible(
    skill: dict, query: str, tool_names=(), *, mapping_extensions: bool = True
) -> bool:
    if not skill.get("mapping_extension"):
        return True
    if not mapping_extensions:
        return False
    gate = skill.get("retrieval_gate") or {}
    tools = set(gate.get("tool_names") or [])
    if tools.intersection(tool_names):
        return True
    normalized = " ".join(query.casefold().split())
    for value in [*tools, *gate.get("specific_terms", [])]:
        term = " ".join(str(value).casefold().split())
        if not term:
            continue
        if re.search(r"[\u3400-\u9fff]", term):
            if term in normalized:
                return True
        elif re.search(r"(?<![a-z0-9_])" + re.escape(term) + r"(?![a-z0-9_])", normalized):
            return True
    return False


def resource_features(query: str) -> dict:
    body, resource = split_resource(query)
    tokens = lexical_tokens(body)
    resource_tokens: set[str] = set()
    raster = vector = table = semantic = False
    if isinstance(resource, str):
        lowered = resource.casefold()
        resource_tokens = lexical_tokens(lowered)
        raster = bool(re.search(r"\.(?:tif|tiff|npy)(?:$|[/?])", lowered))
        vector = bool(re.search(r"\.(?:geojson|shp|gpkg)(?:$|[/?])", lowered))
        table = bool(re.search(r"\.(?:csv|xlsx|json|jsonl)(?:$|[/?])", lowered))
        semantic = bool(
            re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", lowered)
            and SEMANTIC_TERMS.intersection(tokens | resource_tokens)
        )
        table = table or semantic
    elif isinstance(resource, dict):
        locator = resource.get("source_locator")
        locator = locator if isinstance(locator, dict) else resource
        resource_tokens = lexical_tokens(" ".join(str(v) for v in locator.values()))
        assets = resource.get("input_assets") or resource.get("assets") or []
        assets = [str(v) for v in assets] if isinstance(assets, list) else [str(assets)]
        joined = " ".join(assets).casefold()
        raster = bool(re.search(r"\.(?:tif|tiff|npy)\b", joined))
        vector = bool(re.search(r"\.(?:geojson|shp|gpkg)\b", joined))
        table = bool(re.search(r"\.(?:csv|xlsx|json|jsonl)\b", joined))
    return {
        "body": body,
        "tokens": tokens | resource_tokens,
        "has_raster": raster,
        "has_vector": vector,
        "has_table": table,
        "has_semantic_locator": semantic,
    }


def searchable_tokens(skill: dict) -> set[str]:
    return lexical_tokens(
        " ".join(
            [
                skill["id"],
                skill.get("display_name", ""),
                skill.get("description", ""),
                skill.get("domain", {}).get("primary", ""),
                " ".join(skill.get("task_types", [])),
                " ".join(skill.get("keywords", [])),
                " ".join(skill.get("tool_candidates", [])),
            ]
        )
    )


class WeightedSkillRetriever:
    def __init__(self, definitions, *, mapping_extensions: bool = True):
        self.definitions = list(definitions)
        self.mapping_extensions = mapping_extensions

    def retrieve(self, query: str, limit: int = 5, tool_names=()) -> list[SkillMatch]:
        features = resource_features(query)
        requested = set(tool_names)
        eligible = [
            skill
            for skill in self.definitions
            if extension_eligible(
                skill, features["body"], requested, mapping_extensions=self.mapping_extensions
            )
        ]
        documents = {skill["id"]: searchable_tokens(skill) for skill in eligible}
        frequency = Counter(token for tokens in documents.values() for token in tokens)
        count = max(1, len(eligible))
        body = " ".join(features["body"].casefold().split())
        matches = []
        for skill in eligible:
            overlap = sorted(features["tokens"] & documents[skill["id"]])
            hinted = requested & set(skill.get("tool_candidates", []))
            phrases = {
                " ".join(str(v).casefold().replace("_", " ").replace("-", " ").split())
                for v in skill.get("keywords", [])
                if len(str(v).strip()) > 1
            }
            hits = sorted(phrase for phrase in phrases if phrase and phrase in body)
            input_types = " ".join(
                str(item.get("semantic_type", "")) for item in skill.get("inputs", [])
            ).casefold()
            type_bonus = sum(
                (
                    weight
                    for flag, name, weight in [
                        ("has_raster", "raster", 1.5),
                        ("has_vector", "vector", 1.5),
                        ("has_table", "table", 1.0),
                    ]
                    if features[flag] and name in input_types
                ),
                0.0,
            )
            if (
                features["has_semantic_locator"]
                and skill.get("domain", {}).get("primary") == "geo-big-data"
            ):
                type_bonus += 4.0
            components = {
                "token_overlap": min(
                    10.0, sum(1.0 + math.log((count + 1) / (frequency[t] + 1)) for t in overlap)
                ),
                "phrase_match": min(8.0, 2.0 * len(hits)),
                "tool_hint": min(8.0, 3.0 * len(hinted)),
                "domain": 1.5
                if lexical_tokens(str(skill.get("domain", {}).get("primary", "")))
                & features["tokens"]
                else 0.0,
                "data_type": type_bonus,
            }
            score = sum(components.values())
            if score > 0:
                terms = tuple(dict.fromkeys(overlap + hits + sorted(hinted)))
                matches.append(SkillMatch(skill["id"], score, terms, components))
        return sorted(matches, key=lambda match: (-match.score, match.skill_id))[: max(1, limit)]
