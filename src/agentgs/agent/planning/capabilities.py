"""Scoped capability dependencies and candidate scoring.

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
    text = re.sub(r"([a-z])([A-Z])", r"\1-\2", str(label or "")).casefold()
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
        self.scopes = {}
        self.required = {}
        self.capability_tools = {}
        self.tool_capabilities = {}
        self._build_nodes()
        self._apply_extensions(extension_capabilities or {})
        self._build_tool_mapping()

    def _build_nodes(self):
        for sid, skill in self.skills.items():
            order = []
            steps = workflow_steps(skill)
            ids = {
                str(step.get("id", i)): capability_id(
                    step.get("capability_id") or step.get("capability") or step.get("operation")
                )
                for i, step in enumerate(steps)
            }
            scoped = {}
            previous = None
            for i, step in enumerate(steps):
                label = str(step.get("capability") or step.get("operation") or "capability")
                cid = ids[str(step.get("id", i))]
                order.append(cid)
                existing = self.capabilities.get(cid, {})
                refs = step.get("depends_on")
                dependencies = (
                    [ids.get(str(ref), capability_id(ref)) for ref in refs]
                    if refs is not None
                    else ([previous] if previous else [])
                )
                if cid in scoped:
                    raise ValueError(
                        f"Repeated capability {cid} inside Skill {sid} needs distinct capability IDs"
                    )
                scoped[cid] = dependencies
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
                    "dependencies": [],
                }
                previous = cid
            if not order:
                label = str((skill.get("goal") or {}).get("capability") or sid)
                cid = capability_id(label)
                order.append(cid)
                scoped[cid] = []
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
            self.scopes[sid] = scoped
            self.required[sid] = (
                {
                    ids[str(step.get("id", i))]
                    for i, step in enumerate(steps)
                    if step.get("required", True)
                }
                if steps
                else set(order)
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
        for cid, definition in definitions.items():
            if cid in self.capabilities:
                self.capabilities[cid].update(
                    label=cid,
                    operation=definition["description"],
                    description=definition["description"],
                    input_types=list(definition["input_types"]),
                    output_types=list(definition["output_types"]),
                )
        for cid, node in self.capabilities.items():
            node["dependency_scopes"] = {
                sid: list(deps[cid]) for sid, deps in self.scopes.items() if cid in deps
            }

    def _build_tool_mapping(self):
        self.capability_tools = {cid: [] for cid in self.capabilities}
        for sid, edge in self.edges.items():
            skill = self.skills[sid]
            fallback = [str(tool) for tool in skill.get("tool_candidates", []) if tool]
            if not workflow_steps(skill):
                for cid in edge["order"]:
                    self.capability_tools[cid] = list(
                        dict.fromkeys([*self.capability_tools[cid], *fallback])
                    )
                    for tool in fallback:
                        self.tool_capabilities.setdefault(tool, [])
                        if cid not in self.tool_capabilities[tool]:
                            self.tool_capabilities[tool].append(cid)
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

    def expand(
        self,
        skill_ids,
        query="",
        *,
        complete_missing=True,
        expand_predecessors=True,
        requirements=None,
    ):
        selected = [sid for sid in dict.fromkeys(map(str, skill_ids)) if sid in self.edges]
        active = []
        completed = []
        requested = concept_words(query)
        for sid in selected:
            order = self.edges[sid]["order"]
            checked, stack = set(), set()

            def check_cycle(cid):
                if cid in stack:
                    raise ValueError(f"Cyclic capability dependencies in {sid}: {cid}")
                if cid in checked:
                    return
                stack.add(cid)
                for dep in self.scopes[sid].get(cid, []):
                    check_cycle(dep)
                stack.remove(cid)
                checked.add(cid)

            for cid in order:
                check_cycle(cid)
            required = self.required[sid]
            prerequisites = {
                dep for cid, deps in self.scopes[sid].items() if cid in required for dep in deps
            }
            terminals = [cid for cid in order if cid in required and cid not in prerequisites]
            # A scope without terminal nodes still needs cycle diagnostics.
            roots = terminals if requirements is None else requirements.get(sid, [])
            for cid in roots:
                if cid not in self.scopes[sid]:
                    raise ValueError(f"Unknown capability selection: {sid}::{cid}")
                active.append((sid, cid))
        missing, visiting, visited, ordered = [], set(), set(), []
        initial = set(active)

        def visit(key):
            sid, cid = key
            if key in visiting:
                raise ValueError(f"Cyclic capability dependencies in {sid}: {cid}")
            if key in visited:
                return
            visiting.add(key)
            for dep in self.scopes[sid].get(cid, []):
                target = (sid, dep)
                if dep not in self.scopes[sid]:
                    missing.append(f"{sid}::{dep}")
                elif expand_predecessors or target in initial:
                    visit(target)
                else:
                    missing.append(f"{sid}::{dep}")
            visiting.remove(key)
            visited.add(key)
            ordered.append(key)

        for key in active:
            visit(key)
        if complete_missing:
            omitted_predecessors = set()
            if not expand_predecessors:

                def mark_predecessors(sid, cid):
                    for dep in self.scopes[sid].get(cid, []):
                        key = (sid, dep)
                        if key not in omitted_predecessors:
                            omitted_predecessors.add(key)
                            mark_predecessors(sid, dep)

                for sid, cid in active:
                    mark_predecessors(sid, cid)
            covered = (
                set().union(
                    *(concept_words(self.capabilities[cid]["description"]) for _, cid in ordered)
                )
                if ordered
                else set()
            )
            for sid in selected:
                for other in self.edges[sid]["compatible_skills"]:
                    if self.skills.get(other, {}).get("planning", {}).get("bridge_only"):
                        continue
                    for cid in self.edges.get(other, {}).get("order", []):
                        if (other, cid) in visited:
                            continue
                        # Support completion must not restore the predecessor closure
                        # disabled by an independent intervention.
                        if (other, cid) in omitted_predecessors:
                            continue
                        if concept_words(self.capabilities[cid]["description"]) & (
                            requested - covered
                        ):
                            initial.add((other, cid))
                            previous_count = len(ordered)
                            visit((other, cid))
                            completed.append(cid)
                            for _, added in ordered[previous_count:]:
                                covered.update(
                                    concept_words(self.capabilities[added]["description"])
                                )
        realized = {f"{sid}::{cid}" for sid, cid in ordered}
        missing = [ref for ref in missing if ref not in realized]
        capability_ids = list(dict.fromkeys(cid for _, cid in ordered))
        nodes = []
        for cid in capability_ids:
            node = deepcopy(self.capabilities[cid])
            node["dependencies"] = list(
                dict.fromkeys(
                    dep
                    for sid, item in ordered
                    if item == cid
                    for dep in self.scopes[sid].get(cid, [])
                )
            )
            nodes.append(node)
        return {
            "skill_ids": list(dict.fromkeys(sid for sid, _ in ordered)),
            "capabilities": nodes,
            "capability_ids": capability_ids,
            "instances": [
                {
                    "id": f"{sid}::{cid}",
                    "skill_id": sid,
                    "capability_id": cid,
                    "depends_on": [f"{sid}::{dep}" for dep in self.scopes[sid].get(cid, [])],
                }
                for sid, cid in ordered
            ],
            "completed_capabilities": list(dict.fromkeys(completed)),
            "missing_capabilities": list(dict.fromkeys(missing)),
            "unresolved_capabilities": list(dict.fromkeys(missing)),
            "unmatched_query_terms": sorted(
                requested - set().union(*(concept_words(n["description"]) for n in nodes))
            )
            if nodes
            else sorted(requested),
            "dependency_order": [f"{sid}::{cid}" for sid, cid in ordered],
            "deduplicated_count": len(ordered) - len(capability_ids),
        }

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
