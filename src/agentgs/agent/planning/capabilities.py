"""Capability composition and candidate scoring from the selected baseline.

Declarations build the graph; tool availability is a separate input. Semantic
nodes may be shared, while concrete tool execution instances are never merged.
"""

import re
from copy import deepcopy

GENERIC_OPERATIONS = frozenset(
    "calculate compute query retrieve analyze analysis use make give get return report determine".split()
)
QUERY_STOPWORDS = frozenset(
    "a an and are as at be been by for from has have how in into is it of on or that the "
    "their this to use using with data file files input output raster rasters geotiff tif tiff "
    "band bands red green blue nir swir nodata no-data value values".split()
)


def capability_id(label) -> str:
    text = str(label or "").casefold()
    text = re.sub(r"([a-z])([A-Z])", r"\1-\2", text)
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "-", text).strip("-") or "capability"


def concept_words(value) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-z][a-z0-9-]{1,}|[\u3400-\u9fff]{2,}", str(value or "").casefold()
        )
        if token not in GENERIC_OPERATIONS and token not in QUERY_STOPWORDS
    }


def workflow_steps(skill) -> list[dict]:
    workflow = skill.get("workflow")
    steps = workflow.get("steps") if isinstance(workflow, dict) else None
    return steps if isinstance(steps, list) else []


def semantic_types(items) -> list[str]:
    return [
        str(item.get("semantic_type") or item.get("data_type") or "")
        for item in items
        if isinstance(item, dict)
    ]


class CapabilityGraph:
    def __init__(self, definitions, *, extension_capabilities=None):
        self.skills = {str(skill["id"]): deepcopy(skill) for skill in definitions}
        self.capabilities = {}
        self.edges = {}
        self.capability_tools = {}
        self.tool_capabilities = {}
        self._build_nodes()
        self._apply_extensions(extension_capabilities or {})
        self._build_tool_mapping()

    def _build_nodes(self):
        for sid, skill in self.skills.items():
            order = []
            previous = None
            for step in workflow_steps(skill):
                label = str(step.get("capability") or step.get("operation") or "capability")
                cid = capability_id(label)
                order.append(cid)
                existing = self.capabilities.get(cid, {})
                dependencies = list(existing.get("dependencies", []))
                if previous and previous != cid and previous not in dependencies:
                    dependencies.append(previous)
                description = " ".join(
                    str(value)
                    for value in [
                        step.get("operation") or label,
                        skill.get("description") or "",
                        " ".join(str(v) for v in skill.get("keywords", [])),
                    ]
                    if value
                )
                self.capabilities[cid] = {
                    "capability_id": cid,
                    "label": label,
                    "operation": str(step.get("operation") or ""),
                    "description": description,
                    "input_types": semantic_types(skill.get("inputs", [])),
                    "output_types": semantic_types(skill.get("outputs", [])),
                    "source_skills": list(dict.fromkeys([*existing.get("source_skills", []), sid])),
                    "dependencies": dependencies,
                }
                previous = cid
            if not order:
                label = str((skill.get("goal") or {}).get("capability") or sid)
                cid = capability_id(label)
                order.append(cid)
                self.capabilities.setdefault(
                    cid,
                    {
                        "capability_id": cid,
                        "label": label,
                        "operation": "",
                        "description": "",
                        "input_types": [],
                        "output_types": [],
                        "source_skills": [sid],
                        "dependencies": [],
                    },
                )
            self.edges[sid] = {
                "skill_id": sid,
                "capability_ids": list(dict.fromkeys(order)),
                "order": order,
                "compatible_skills": [
                    str(value)
                    for value in (skill.get("planning") or {}).get("compatible_skills", [])
                    if value
                ],
            }

    def _apply_extensions(self, definitions):
        dependencies = {cid: [] for cid in definitions}
        for skill in self.skills.values():
            if not skill.get("mapping_extension"):
                continue
            steps = workflow_steps(skill)
            ids = {step["id"]: step.get("capability_id") or step["capability"] for step in steps}
            for step in steps:
                cid = step.get("capability_id") or step["capability"]
                for ref in step.get("depends_on") or []:
                    dependency = ids.get(ref, ref)
                    if (
                        dependency in ids.values()
                        and dependency != cid
                        and dependency not in dependencies[cid]
                    ):
                        dependencies[cid].append(dependency)
        for cid, definition in definitions.items():
            if cid in self.capabilities:
                self.capabilities[cid].update(
                    label=cid,
                    operation=definition["description"],
                    description=definition["description"],
                    input_types=list(definition["input_types"]),
                    output_types=list(definition["output_types"]),
                    dependencies=dependencies[cid],
                )

    def _build_tool_mapping(self):
        self.capability_tools = {cid: [] for cid in self.capabilities}
        for sid, edge in self.edges.items():
            skill = self.skills[sid]
            fallback = [str(tool) for tool in skill.get("tool_candidates", []) if tool]
            for step, cid in zip(workflow_steps(skill), edge["order"]):
                selection = step.get("tool_selection")
                selection = selection if isinstance(selection, dict) else {}
                candidates = [str(v) for v in selection.get("candidates", []) if v] or fallback
                for tool in candidates:
                    if tool not in self.capability_tools[cid]:
                        self.capability_tools[cid].append(tool)
                    self.tool_capabilities.setdefault(tool, [])
                    if cid not in self.tool_capabilities[tool]:
                        self.tool_capabilities[tool].append(cid)

    def as_dict(self):
        return deepcopy(
            {
                "schema_version": "geoskill-hypergraph-0.1",
                "hyperedges": list(self.edges.values()),
                "capabilities": list(self.capabilities.values()),
            }
        )

    def tool_mapping(self):
        return deepcopy(
            {
                "schema_version": "geotool-tool-graph-0.1",
                "capability_to_tools": self.capability_tools,
                "tool_to_capabilities": self.tool_capabilities,
            }
        )

    def expand(self, skill_ids, query="", *, complete_missing=True):
        selected = [sid for sid in dict.fromkeys(str(v) for v in skill_ids) if sid in self.edges]
        ordered = list(dict.fromkeys(cid for sid in selected for cid in self.edges[sid]["order"]))
        source_edges = list(selected)
        covered = set()
        for sid in selected:
            skill = self.skills[sid]
            covered.update(
                concept_words(
                    " ".join(
                        str(skill.get(key) or "")
                        for key in ["validation", "preconditions", "recovery"]
                    )
                )
            )
            if skill.get("validation"):
                covered.add("validate")

        def node_words(cid):
            node = self.capabilities[cid]
            return concept_words(
                " ".join(node[key] for key in ["label", "operation", "description"])
            )

        for cid in ordered:
            covered.update(node_words(cid))
        requested = sorted(word for word in concept_words(query) if len(word) > 2)
        missing = [word for word in requested if word not in covered]
        completed = []
        unresolved = []
        if complete_missing and missing:
            for sid in selected:
                for compatible in self.edges[sid]["compatible_skills"]:
                    for cid in self.edges.get(compatible, {}).get("order", []):
                        words = node_words(cid)
                        if words & set(missing) and cid not in ordered:
                            ordered.append(cid)
                            source_edges.append(compatible)
                            completed.append(cid)
                            covered.update(words)
            unresolved = [word for word in missing if word not in covered]
        index = {cid: i for i, cid in enumerate(ordered)}
        # Keep the implemented historical ordering rule, including cycle behavior.
        # A new topological sort is an algorithm change, not a container refactor.
        dependencies = sorted(
            ordered,
            key=lambda cid: (
                min(
                    (index.get(dep, -1) for dep in self.capabilities[cid]["dependencies"]),
                    default=-1,
                ),
                index[cid],
            ),
        )
        return deepcopy(
            {
                "skill_ids": list(dict.fromkeys(source_edges)),
                "capabilities": [self.capabilities[cid] for cid in ordered],
                "capability_ids": ordered,
                "completed_capabilities": completed,
                "missing_capabilities": missing,
                "unresolved_capabilities": unresolved,
                "dependency_order": dependencies,
                "deduplicated_count": sum(len(self.edges[sid]["order"]) for sid in selected)
                - len(ordered),
            }
        )

    def candidates(
        self, ids, available_tools=(), inferred_tools=(), schemas=None, artifact_types=()
    ):
        available, inferred = set(map(str, available_tools)), set(map(str, inferred_tools))
        schemas = schemas or {}
        types = concept_words(" ".join(str(value) for value in artifact_types))
        result = {}
        for cid in dict.fromkeys(str(value) for value in ids):
            if cid not in self.capabilities:
                continue
            rows = []
            for tool in self.capability_tools.get(cid, []):
                schema = schemas.get(tool) or {}
                input_format = str(schema.get("input_format") or schema.get("Input") or "")
                output_format = str(schema.get("output_format") or schema.get("Output") or "")
                score = (
                    1.0 + 4.0 * (tool in available) + 3.0 * (tool in inferred) + float(bool(schema))
                )
                if types and types & concept_words(input_format + " " + output_format):
                    score += 0.5
                rows.append(
                    {
                        "tool_name": tool,
                        "capability_id": cid,
                        "score": round(score, 4),
                        "available": tool in available,
                        "input_format": input_format,
                        "output_format": output_format,
                        "source_skills": list(self.capabilities[cid]["source_skills"]),
                    }
                )
            result[cid] = sorted(rows, key=lambda row: (-row["score"], row["tool_name"]))
        return result
