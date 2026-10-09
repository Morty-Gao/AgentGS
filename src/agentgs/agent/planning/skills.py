"""Pure Skill planning: ranking, selection, diagnostics and capability expansion.

Tool retrieval, bridge search and execution belong to later stages. Keeping
these stages distinct preserves the baseline's decision order.
"""

import re
from copy import deepcopy

from agentgs.agent.planning.capabilities import CapabilityGraph
from agentgs.agent.planning.constraints import check_constraints
from agentgs.agent.planning.module_expansion import expand_requirements
from agentgs.agent.planning.selection import select_skill_ids
from agentgs.agent.retrieval.scored_skills import (
    WeightedSkillRetriever,
    lexical_tokens,
    resource_features,
    split_resource,
)

RS_SKILLS = frozenset(
    {
        "remote-sensing-data-discovery",
        "remote-sensing-preprocessing",
        "remote-sensing-raster-statistics",
        "remote-sensing-change-detection",
        "remote-sensing-trend-distribution",
    }
)
RS_PROFILE = {
    "version": "rs-local-v1",
    "pair_by": ["acquisition_date", "scene", "band_role"],
    "unknown_metadata": "report_unknown; never treat it as verified",
    "date_window": "use rs_date_window=[start,end] only when explicitly specified",
    "batch_cardinality": "one batch OR disjoint per-pair calls, never both",
    "rs_statistic_mode": "per_scene_statistics returns scalars; pixelwise_composite requires aligned grids; legacy retains both outputs",
}


def query_profile(query):
    clean, resource = split_resource(query)
    features = resource_features(query)
    profile = {
        "tokens": sorted(lexical_tokens(clean)),
        "phrases": sorted(
            {
                item.strip()
                for item in re.split(r"[,;|/]|\band\b|\bor\b", clean.casefold())
                if len(item.strip()) > 2
            }
        ),
        "has_raster": features["has_raster"],
        "has_vector": features["has_vector"],
        "has_table": features["has_table"],
        "has_semantic_locator": features["has_semantic_locator"],
        "has_temporal": bool(
            re.search(
                r"\b(?:19|20|21)\d{2}\b|\b(?:date|year|month|day|annual|monthly|temporal)\b",
                clean,
                re.I,
            )
        ),
        "has_spatial": bool(
            re.search(
                r"\b(?:crs|coordinate|bbox|distance|nearest|overlay|zone|map|raster|vector|spatial|extent)\b",
                clean,
                re.I,
            )
        ),
        "operations": sorted(
            set(
                re.findall(
                    r"\b(?:calculate|compute|compare|difference|mean|average|sum|rank|filter|query|retrieve|interpolate|overlay|validate|standardize|render|map|trend|ratio|threshold|nearest|buffer|sample|count|aggregate)\w*\b",
                    clean,
                    re.I,
                )
            )
        ),
        "resource_keys": [],
        "resource_assets": [],
        "resource_tokens": [],
    }
    if isinstance(resource, str):
        profile["resource_locator"] = resource
        profile["resource_tokens"] = sorted(lexical_tokens(resource.casefold()))
        if features["has_semantic_locator"]:
            profile["has_spatial"] = True
    elif isinstance(resource, dict):
        locator = resource.get("source_locator")
        locator = locator if isinstance(locator, dict) else resource
        profile["resource_keys"] = sorted(str(key) for key in locator)
        assets = resource.get("input_assets") or resource.get("assets") or []
        profile["resource_assets"] = (
            [str(v) for v in assets] if isinstance(assets, list) else [str(assets)]
        )
        profile["resource_tokens"] = sorted(
            lexical_tokens(" ".join(str(v) for v in locator.values()))
        )
    return profile


class SkillPlanner:
    def __init__(
        self,
        definitions,
        *,
        extension_capabilities=None,
        mapping_extensions=True,
        rs_local_contracts=True,
        predecessor_expansion=True,
        capability_completion=True,
    ):
        self.skills = {skill["id"]: deepcopy(skill) for skill in definitions}
        self.mapping_extensions = mapping_extensions
        self.rs_local_contracts = rs_local_contracts
        self.predecessor_expansion = predecessor_expansion
        self.capability_completion = capability_completion
        self.graph = CapabilityGraph(
            self.skills.values(), extension_capabilities=extension_capabilities
        )

    def retrieve(self, query, inferred_tools=()):
        inferred = list(dict.fromkeys(inferred_tools))
        matches = WeightedSkillRetriever(
            self.skills.values(), mapping_extensions=self.mapping_extensions
        ).retrieve(query, tool_names=inferred)
        selected = select_skill_ids(
            self.skills.values(), query, inferred, mapping_extensions=self.mapping_extensions
        )
        return {
            "candidates": [match.as_dict() for match in matches],
            "declarations": [deepcopy(self.skills[sid]) for sid in selected],
        }

    def plan(self, query, inferred_tools=(), *, requirements=None, support_requirements=None):
        inferred = list(dict.fromkeys(inferred_tools))
        retrieval = self.retrieve(query, inferred)
        selected = [skill["id"] for skill in retrieval["declarations"]]
        if requirements is not None:
            if set(requirements) - set(selected):
                raise ValueError("Selected strategy is outside the retrieved candidate scope")
            selected = list(requirements)
        proposed_expansion = expand_requirements(
            self.graph,
            selected,
            query,
            predecessor_expansion=self.predecessor_expansion,
            capability_completion=self.capability_completion,
            requirements=requirements,
        )
        support_candidates = [
            row
            for row in proposed_expansion["instances"]
            if row["id"] in proposed_expansion["support_added"]
        ]
        expansion = proposed_expansion
        if requirements is not None:
            expansion = expand_requirements(
                self.graph,
                selected,
                query,
                predecessor_expansion=self.predecessor_expansion,
                capability_completion=False,
                requirements=requirements,
            )
            approved = support_requirements or {}
            allowed_support = {
                (row["skill_id"], row["capability_id"]) for row in support_candidates
            }
            if any(
                (sid, cid) not in allowed_support for sid, ids in approved.items() for cid in ids
            ):
                raise ValueError("Support operation is outside the diagnosed candidate scope")
            if approved and not self.capability_completion:
                raise ValueError("Support insertion is disabled")
            if approved:
                combined = deepcopy(requirements)
                for sid, ids in approved.items():
                    combined[sid] = list(dict.fromkeys([*combined.get(sid, []), *ids]))
                final = self.graph.expand(
                    list(combined),
                    query,
                    expand_predecessors=self.predecessor_expansion,
                    complete_missing=False,
                    requirements=combined,
                )
                existing = {row["id"] for row in expansion["instances"]}
                added = [row for row in final["instances"] if row["id"] not in existing]
                expansion = {
                    **expansion,
                    **final,
                    "support_added": [row["id"] for row in added],
                    "completed_capabilities": list(
                        dict.fromkeys(row["capability_id"] for row in added)
                    ),
                }
                selected = list(combined)
        steps, constraints, produced = [], [], []
        for index, sid in enumerate(selected, 1):
            skill = self.skills[sid]
            constraints.append(check_constraints(skill, query))
            step_id = f"skill_step_{index:02d}"
            outputs = [
                f"{step_id}:{str(row.get('id') or f'output_{i}')}"
                for i, row in enumerate(skill.get("outputs", []), 1)
            ]
            step = {
                "step": index,
                "step_id": step_id,
                "instance_id": f"{sid}#{index}",
                "skill_id": sid,
                "goal": skill["goal"],
                "workflow": skill["workflow"],
                "inputs": skill["inputs"],
                "outputs": skill["outputs"],
                "consumes": [
                    str(row.get("id") or f"input_{i}")
                    for i, row in enumerate(skill.get("inputs", []), 1)
                ],
                "produces": outputs,
                "depends_on": [steps[-1]["step_id"]] if steps and produced else [],
                "available_upstream_artifacts": list(produced),
                "tool_candidates": skill["tool_candidates"],
                "inferred_tools": [
                    name for name in inferred if name in skill.get("tool_candidates", [])
                ],
                "geo_constraints": skill["geo_constraints"],
                "validation": skill["validation"],
                "failure_recovery": {
                    "failure_modes": skill["failure_modes"],
                    "recovery": skill["recovery"],
                },
            }
            if self.rs_local_contracts and sid in RS_SKILLS:
                step["remote_sensing_contract"] = deepcopy(RS_PROFILE)
            steps.append(step)
            produced.extend(outputs)
        errors = []
        known_ids = {row["step_id"] for row in steps}
        for row in steps:
            missing = set(row["depends_on"]) - known_ids
            if missing:
                errors.append(f"{row['step_id']} has unknown dependencies: {sorted(missing)}")
            if not row["tool_candidates"]:
                errors.append(f"{row['step_id']} has no Tool candidates")
        return deepcopy(
            {
                "query": query,
                "mode": "skill_guided",
                "declarations": [
                    deepcopy(self.skills[sid])
                    for sid in dict.fromkeys([*selected, *expansion["skill_ids"]])
                ],
                "candidates": retrieval["candidates"],
                "selected_requirements": deepcopy(requirements),
                "selection_source": "model" if requirements is not None else "retrieval",
                "support_candidates": support_candidates,
                "approved_support": deepcopy(support_requirements or {}),
                "steps": steps,
                "inferred_tools": inferred,
                "constraints": constraints,
                "query_profile": query_profile(query),
                "capabilities": expansion["capabilities"],
                "capability_expansion": expansion,
                "missing_capabilities": expansion["unresolved_capabilities"],
                "hypergraph": {
                    "schema_version": "geoskill-hypergraph-0.1",
                    "selected_hyperedges": [
                        self.graph.edges[sid] for sid in expansion["skill_ids"]
                    ],
                    "capability_expansion": expansion,
                },
                "validation": {
                    "valid": not errors,
                    "errors": errors,
                    "dependency_order": [row["step_id"] for row in steps],
                    "produced_artifacts": produced,
                    "capability_expansion_valid": not bool(expansion["unresolved_capabilities"]),
                    "missing_capabilities": expansion["missing_capabilities"],
                    "completed_capabilities": expansion["completed_capabilities"],
                },
            }
        )


def builtin_planner(
    *,
    mapping_extensions=True,
    rs_local_contracts=True,
    predecessor_expansion=True,
    capability_completion=True,
):
    """Assemble the frozen declaration profile without process-global flags."""
    from agentgs.skills.definitions import load_capability_extensions, load_skill_definitions

    definitions = load_skill_definitions()
    if not mapping_extensions:
        definitions = [skill for skill in definitions if not skill.get("mapping_extension")]
    return SkillPlanner(
        definitions,
        extension_capabilities=load_capability_extensions(),
        mapping_extensions=mapping_extensions,
        rs_local_contracts=rs_local_contracts,
        predecessor_expansion=predecessor_expansion,
        capability_completion=capability_completion,
    )
