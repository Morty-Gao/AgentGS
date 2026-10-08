"""Directed shared-capability completion adapted to the 2026-08-26 runtime.

The legacy planner and ToolGraph remain authoritative unless a new, explicitly
authored bridge passes every check. Whole-Skill IO and list order are never used
as evidence for a new bridge. This module has no model or network dependencies.
"""

from __future__ import annotations

import json
from collections import defaultdict
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

from .bridging import discover_bridges

_bridge_enabled = ContextVar("shared_bridge_enabled", default=True)


class _CapabilityIdentity:
    from agentgs.agent.planning.capabilities import capability_id

    capability_id = staticmethod(capability_id)


def shared_bridge_enabled():
    return _bridge_enabled.get()


@dataclass(frozen=True)
class BridgeNode:
    capability_id: str
    input_types: tuple = ()
    output_types: tuple = ()
    candidate_tools: tuple = ()
    source_skills: tuple = ()
    dependencies: tuple = ()
    data_dependencies: tuple = ()
    control_dependencies: tuple = ()

    def as_dict(self):
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.__dict__.items()
        }


def _identity(step):
    return str(
        step.get("capability_id")
        or _CapabilityIdentity.capability_id(step.get("capability") or step.get("operation"))
    )


def _explicit_types(step, key):
    values = step.get(key)
    if isinstance(values, str):
        values = [values]
    return tuple((str(item) for item in values or () if item))


class DeclaredBridgeGraph:
    """Read-only facade with only explicit within-step scientific contracts."""

    def __init__(self, skills):
        self.skills = {str(s["id"]): deepcopy(s) for s in skills}
        self.hyperedges, self.capabilities, self._step_scopes = ({}, {}, {})
        declarations = defaultdict(list)
        for sid, skill in self.skills.items():
            steps = skill.get("workflow", {}).get("steps", [])
            ids = {
                str(step.get("id") or f"step_{index}"): _identity(step)
                for index, step in enumerate(steps, 1)
            }
            scopes = {}
            for step in steps:
                cid = _identity(step)
                if cid in scopes:
                    scopes[cid]["candidate_tools"] = ()
                    scopes[cid]["duplicate_identity"] = True
                    continue
                deps = tuple((ids.get(str(d), str(d)) for d in step.get("depends_on", [])))
                conditional = tuple(
                    (ids.get(str(d), str(d)) for d in step.get("after_if_present", []))
                )
                selection = step.get("tool_selection", {})
                tools = tuple((str(x) for x in selection.get("candidates", []) if x))
                bindings = deepcopy(step.get("bridge_input_bindings", {}))
                for binding in bindings.values():
                    if isinstance(binding, dict):
                        source = str(binding.get("source", ""))
                        binding["source"] = ids.get(source, source)
                scopes[cid] = {
                    "dependencies": deps,
                    "after_if_present": conditional,
                    "candidate_tools": tools,
                    "required": bool(step.get("required", True)),
                    "input_types": _explicit_types(step, "input_types"),
                    "output_types": _explicit_types(step, "output_types"),
                    "input_bindings": bindings,
                    "definition": deepcopy(step),
                }
                declarations[cid].append((sid, scopes[cid]))
            self._step_scopes[sid] = scopes
            self.hyperedges[sid] = SimpleNamespace(capability_ids=tuple(scopes))
        for cid, definitions in declarations.items():
            inputs = {
                tuple(sorted(row["input_types"])) for _, row in definitions if row["input_types"]
            }
            outputs = {
                tuple(sorted(row["output_types"])) for _, row in definitions if row["output_types"]
            }
            consistent = len(inputs) == 1 and len(outputs) == 1
            self.capabilities[cid] = BridgeNode(
                cid,
                next(iter(inputs)) if consistent else (),
                next(iter(outputs)) if consistent else (),
                tuple(
                    dict.fromkeys(
                        (tool for _, row in definitions for tool in row["candidate_tools"])
                    )
                ),
                tuple((sid for sid, _ in definitions)),
            )

    def expansion(self, plan):
        selected = list(dict.fromkeys((row["skill_id"] for row in plan.steps)))
        active = [str(row["capability_id"]) for row in plan.capabilities]
        nodes = []
        for cid in active:
            original = self.capabilities.get(cid, BridgeNode(cid))
            scopes = [self._step_scopes.get(sid, {}).get(cid) for sid in selected]
            scopes = [row for row in scopes if row is not None]
            dependencies = tuple(
                dict.fromkeys(
                    (
                        dep
                        for row in scopes
                        for dep in (*row["dependencies"], *row["after_if_present"])
                        if dep in active
                    )
                )
            )
            declared = scopes and all(
                (row["input_types"] and row["output_types"] for row in scopes)
            )
            nodes.append(
                BridgeNode(
                    cid,
                    original.input_types if declared else (),
                    original.output_types if declared else (),
                    tuple(dict.fromkeys((t for row in scopes for t in row["candidate_tools"]))),
                    tuple((sid for sid in selected if cid in self._step_scopes.get(sid, {}))),
                    dependencies,
                    dependencies,
                )
            )
        return SimpleNamespace(skill_ids=selected, capabilities=nodes)


def _topological(nodes, edges):
    ordered = list(dict.fromkeys(nodes))
    result = []
    while len(result) < len(ordered):
        remaining = [node for node in ordered if node not in result]
        ready = [
            node
            for node in remaining
            if all((a in result for a, b in edges if b == node and a in ordered))
        ]
        if not ready:
            raise ValueError("cyclic_projected_bridge")
        result.extend(ready)
    return result


def _qualified_route(graph, expansion, route, available):
    result = deepcopy(route)
    active = {node.capability_id: node for node in expansion.capabilities}
    scopes = graph._step_scopes[route["skill_id"]]
    path = route["capability_path"]
    if any(
        (not active[cid].input_types or not active[cid].output_types for cid in (path[0], path[-1]))
    ):
        raise ValueError("missing_active_endpoint_scoped_io")
    contract = {}
    for cid in path:
        scope = scopes[cid]
        node = active[cid] if cid == path[0] else graph.capabilities[cid]
        options = set(node.candidate_tools if cid == path[0] else scope["candidate_tools"])
        options &= available
        if scope.get("duplicate_identity") or not options:
            raise ValueError("unavailable_or_duplicate_bridge_capability")
        if cid != path[0] and (not scope["input_types"] or not scope["output_types"]):
            raise ValueError("missing_step_scoped_io_contract")
        if len(options) != 1:
            raise ValueError("ambiguous_bridge_operation_tool")
        bindings = deepcopy(scope["input_bindings"]) if cid != path[0] else {}
        if cid != path[0]:
            incoming = {edge["source"] for edge in route["data_edges"] if edge["target"] == cid}
            bound = set()
            for name, binding in bindings.items():
                if not isinstance(name, str) or not isinstance(binding, dict):
                    raise ValueError("invalid_bridge_input_binding")
                if binding.get("source") not in incoming or not isinstance(
                    binding.get("path"), list
                ):
                    raise ValueError("invalid_bridge_input_binding")
                if any(
                    (
                        not isinstance(item, (str, int)) or isinstance(item, bool)
                        for item in binding["path"]
                    )
                ):
                    raise ValueError("invalid_bridge_output_projection")
                if binding.get("transform") not in (None, "identity", "list"):
                    raise ValueError("unsupported_bridge_binding_transform")
                bound.add(binding["source"])
            if bound != incoming:
                raise ValueError("bridge_data_dependency_without_argument_binding")
        contract[cid] = {"tool_name": next(iter(options)), "input_bindings": bindings}
    for edge in route["data_edges"]:
        cid = edge["source"]
        if cid in contract:
            continue
        node = active.get(cid)
        options = set(node.candidate_tools) & available if node else set()
        if len(options) != 1:
            raise ValueError("off_branch_bridge_input_requires_unique_available_tool")
        contract[cid] = {"tool_name": next(iter(options)), "input_bindings": {}}
    tool_names = [row["tool_name"] for row in contract.values()]
    if len(set(tool_names)) != len(tool_names):
        raise ValueError("shared_physical_tool_requires_instance_disambiguation")
    result["execution_contract"] = contract
    result["status"] = "planned_requires_tool_chain_validation"
    return result


def complete_shared_capability_bridges(plan, skills, available_tools, *, enabled=None):
    """Project only accepted C branches; on no bridge preserve plan byte content."""
    enabled = shared_bridge_enabled() if enabled is None else bool(enabled)
    report = {
        "enabled": enabled,
        "accepted": [],
        "rejected": [],
        "phase_order": [
            "legacy_semantic_and_compatible_completion",
            "declared_input_dependencies",
            "single_skill_shared_capability_bridge",
        ],
        "max_new_skill_hops": 1,
    }
    if not enabled:
        return report
    available_tools = tuple(available_tools)
    old_state = getattr(plan, "_shared_bridge_state", None)
    if old_state is not None:
        return old_state
    graph = DeclaredBridgeGraph(skills)
    expansion = graph.expansion(plan)
    report["seed_skill_ids"] = list(expansion.skill_ids)
    report["seed_capability_ids"] = [n.capability_id for n in expansion.capabilities]
    discovered = discover_bridges(graph, expansion, available_tools)
    report["rejected"] = discovered["rejected"]
    available = set(available_tools)
    for route in discovered["accepted"]:
        try:
            report["accepted"].append(_qualified_route(graph, expansion, route, available))
        except (ValueError, KeyError, TypeError) as exc:
            report["rejected"].append({**route, "reason": str(exc)})
    if not report["accepted"]:
        return report
    initial = {node.capability_id: node for node in expansion.capabilities}
    ids = list(initial)
    edges = {(dep, node.capability_id) for node in initial.values() for dep in node.dependencies}
    for route in report["accepted"]:
        path = route["capability_path"]
        ids.extend((cid for cid in path if cid not in ids))
        edges.discard((path[0], path[-1]))
        edges.update(((edge["source"], edge["target"]) for edge in route["data_edges"]))
    ordered = _topological(ids, edges)
    rows = {}
    for cid in ordered:
        row = deepcopy(
            next(
                (r for r in plan.capabilities if r["capability_id"] == cid),
                graph.capabilities[cid].as_dict(),
            )
        )
        row["dependencies"] = [a for a, b in sorted(edges) if b == cid]
        row["data_dependencies"] = list(row["dependencies"])
        rows[cid] = row
    by_skill = {row["skill_id"]: deepcopy(row) for row in plan.steps}
    skill_ids = list(by_skill)
    old_step_ids = {row.get("step_id", sid): sid for sid, row in by_skill.items()}
    old_skill_edges = {
        (old_step_ids.get(dep, dep), sid)
        for sid, row in by_skill.items()
        for dep in row.get("depends_on", [])
        if old_step_ids.get(dep, dep) in by_skill
    }
    skill_edges = set()
    for route in report["accepted"]:
        sid, path = (route["skill_id"], route["capability_path"])
        source, target = (route["source_skill_id"], route["target_skill_id"])
        skill_edges.update(((source, sid), (sid, target)))
        if sid not in by_skill:
            skill_ids.append(sid)
            definition = graph.skills[sid]
            by_skill[sid] = {
                "skill_id": sid,
                "goal": deepcopy(definition.get("goal", {})),
                "inputs": deepcopy(definition.get("inputs", [])),
                "outputs": deepcopy(definition.get("outputs", [])),
                "consumes": [],
                "produces": [],
                "available_upstream_artifacts": [],
                "inferred_tools": [],
                "geo_constraints": deepcopy(definition.get("geo_constraints", {})),
                "validation": deepcopy(definition.get("validation", {})),
                "failure_recovery": {
                    "failure_modes": deepcopy(definition.get("failure_modes", [])),
                    "recovery": deepcopy(definition.get("recovery", {})),
                },
                "completion_source": "directed_shared_capability_bridge",
            }
        prior = set(by_skill[sid].get("active_bridge_capabilities", []))
        prior.update(path)
        by_skill[sid]["active_bridge_capabilities"] = sorted(prior)
        definitions = graph._step_scopes[sid]
        selected_steps = []
        for cid in ordered:
            if cid not in prior:
                continue
            step = deepcopy(definitions[cid]["definition"])
            step["id"] = cid
            step["capability"] = cid
            step["capability_id"] = cid
            step["depends_on"] = (
                [a for a, b in edges if b == cid and a in prior] if cid != path[0] else []
            )
            step["after_if_present"] = []
            selected_steps.append(step)
        by_skill[sid]["workflow"] = {"type": "sequential", "steps": selected_steps}
        by_skill[sid]["tool_candidates"] = list(
            dict.fromkeys((row["tool_name"] for row in route["execution_contract"].values()))
        )

    def reaches(source, target):
        frontier, seen = ([source], set())
        while frontier:
            current = frontier.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            frontier.extend((b for a, b in skill_edges if a == current))
        return False

    replaced = {(a, b) for a, b in old_skill_edges if reaches(b, a)}
    report["replaced_legacy_skill_edges"] = [list(edge) for edge in sorted(replaced)]
    skill_edges.update(old_skill_edges - replaced)
    try:
        final_skills = _topological(skill_ids, skill_edges)
    except ValueError:
        report["rejected"].extend(
            (
                {**route, "reason": "bridge_conflicts_with_existing_skill_dependencies"}
                for route in report["accepted"]
            )
        )
        report["accepted"] = []
        return report
    original_plan = deepcopy(plan.__dict__)
    step_ids = {sid: f"skill_step_{index:02d}" for index, sid in enumerate(final_skills, 1)}
    for index, sid in enumerate(final_skills, 1):
        by_skill[sid].update(
            step=index,
            step_id=step_ids[sid],
            instance_id=f"{sid}#{index}",
            depends_on=[step_ids[a] for a, b in sorted(skill_edges) if b == sid],
        )
    plan.steps = [by_skill[sid] for sid in final_skills]
    plan.capabilities = [rows[cid] for cid in ordered]
    completed = [cid for cid in ordered if cid not in initial]
    plan.capability_expansion = {
        **deepcopy(plan.capability_expansion),
        "skill_ids": final_skills,
        "capabilities": deepcopy(plan.capabilities),
        "capability_ids": ordered,
        "dependency_order": ordered,
        "completed_capabilities": list(
            dict.fromkeys(
                [*plan.capability_expansion.get("completed_capabilities", []), *completed]
            )
        ),
        "shared_bridge": deepcopy(report),
    }
    plan.hypergraph = {
        **deepcopy(plan.hypergraph),
        "capability_expansion": deepcopy(plan.capability_expansion),
        "selected_hyperedges": [
            {
                "skill_id": sid,
                "capability_ids": list(
                    by_skill[sid].get(
                        "active_bridge_capabilities", graph.hyperedges[sid].capability_ids
                    )
                ),
            }
            for sid in final_skills
        ],
    }
    plan.validation = {
        **deepcopy(plan.validation),
        "dependency_order": [step_ids[sid] for sid in final_skills],
        "shared_bridge_capability_order": ordered,
        "shared_bridge_status": "planned_requires_tool_chain_validation",
    }
    plan._shared_bridge_original_plan = original_plan
    plan._shared_bridge_state = report
    return report


def ground_shared_bridge_chain(chain, report):
    """Insert bridge operations using all AND dependencies and original instances."""
    original = list(chain)
    state = {
        "run_id": f"shared-bridge-{uuid4().hex}",
        "routes": [],
        "execution_records": {},
        "prefix_artifacts": {},
        "comparisons": [],
        "tool_chain": list(original),
        "capability_indices": {},
        "step_bindings": {},
    }
    if not report or not report.get("enabled") or (not report.get("accepted")):
        return (original, None)
    contracts, middle_tools = ({}, [])
    for route in report["accepted"]:
        path, contract = (route["capability_path"], route["execution_contract"])
        source_tool, target_tool = (contract[path[0]]["tool_name"], contract[path[-1]]["tool_name"])
        if original.count(source_tool) != 1 or original.count(target_tool) != 1:
            raise ValueError("accepted_bridge_requires_unique_executed_endpoints")
        if original.index(source_tool) >= original.index(target_tool):
            raise ValueError("accepted_bridge_conflicts_with_proposed_tool_order")
        for cid, row in contract.items():
            previous = contracts.get(cid)
            if previous and previous["tool_name"] != row["tool_name"]:
                raise ValueError("conflicting_shared_capability_tool_contracts")
            if (
                previous
                and previous["input_bindings"]
                and row["input_bindings"]
                and (previous["input_bindings"] != row["input_bindings"])
            ):
                raise ValueError("conflicting_shared_capability_input_bindings")
            if not previous or row["input_bindings"]:
                contracts[cid] = row
        middle_tools.extend((contract[cid]["tool_name"] for cid in path[1:-1]))
        state["routes"].append(deepcopy(route))
    nodes = [("original", index) for index in range(len(original))]
    tool_for = {node: original[node[1]] for node in nodes}
    edges = set(zip(nodes, nodes[1:]))
    for name in dict.fromkeys(middle_tools):
        if name not in original:
            node = ("bridge", name)
            nodes.append(node)
            tool_for[node] = name
    node_for_capability = {}
    for cid, contract in contracts.items():
        matching = [node for node in nodes if tool_for[node] == contract["tool_name"]]
        if len(matching) != 1:
            raise ValueError("bridge_tool_requires_unique_execution_instance")
        node_for_capability[cid] = matching[0]
    for route in state["routes"]:
        for edge in route["data_edges"]:
            edges.add((node_for_capability[edge["source"]], node_for_capability[edge["target"]]))
    ordered = _topological(nodes, edges)
    rewritten = [tool_for[node] for node in ordered]
    for cid, contract in contracts.items():
        index = ordered.index(node_for_capability[cid]) + 1
        state["capability_indices"][cid] = index
        if contract["input_bindings"]:
            state["step_bindings"][index] = deepcopy(contract["input_bindings"])
    for index, bindings in state["step_bindings"].items():
        for binding in bindings.values():
            source = binding["source"]
            if source not in state["capability_indices"]:
                raise ValueError("off_branch_bridge_input_has_no_execution_contract")
            if state["capability_indices"][source] >= index:
                raise ValueError("bridge_input_is_not_an_executed_predecessor")
    state["tool_chain"] = rewritten
    return (rewritten, state)


def ground_or_reject_shared_bridge_chain(chain, plan, report):
    """A bridge grounding failure restores the original plan and proposed chain."""
    try:
        return (*ground_shared_bridge_chain(chain, report), None)
    except (ValueError, KeyError, TypeError) as exc:
        diagnostic = {
            "reason": "bridge_grounding_rejected",
            "detail": str(exc),
            "fallback": "original_skill_plan_and_proposed_tool_chain",
        }
        report["rejected"].extend(({**route, **diagnostic} for route in report.get("accepted", [])))
        report["accepted"] = []
        snapshot = getattr(plan, "_shared_bridge_original_plan", None)
        if snapshot is not None:
            plan.__dict__.clear()
            plan.__dict__.update(deepcopy(snapshot))
        return (list(chain), None, diagnostic)


def _project(output, binding):
    value = json.loads(output) if isinstance(output, str) else deepcopy(output)
    for part in binding["path"]:
        value = value[part]
    if binding.get("transform") == "list":
        value = [value]
    return deepcopy(value)


def bridge_execution_context(state, index):
    return (
        {"state": state, "execution_index": index}
        if state and index in state["step_bindings"]
        else None
    )


def apply_shared_bridge_inputs(tool_input, execution_context):
    context = (execution_context or {}).get("shared_bridge")
    if not context:
        return tool_input
    state, index = (context["state"], context["execution_index"])
    data = json.loads(tool_input) if isinstance(tool_input, str) else deepcopy(tool_input)
    if not isinstance(data, dict):
        raise ValueError("bridge_tool_arguments_must_be_a_json_object")
    for field_name, binding in state["step_bindings"].get(index, {}).items():
        source_index = state["capability_indices"][binding["source"]]
        record = state["execution_records"].get(source_index)
        if not record or not record["succeeded"]:
            raise ValueError("bridge_input_producer_has_not_succeeded")
        data[field_name] = _project(record["output"], binding)
    return json.dumps(data, ensure_ascii=False, allow_nan=False)


def record_shared_bridge_execution(state, index, execution, artifact_registry=None):
    if not state:
        return
    state["execution_records"][index] = {
        "execution_index": index,
        "tool_name": execution.tool_name,
        "input": execution.final_input,
        "output": execution.output,
        "succeeded": execution.succeeded,
    }
    source_indices = {
        state["capability_indices"][route["capability_path"][0]] for route in state["routes"]
    }
    if index in source_indices and artifact_registry is not None:
        state["prefix_artifacts"][index] = deepcopy(artifact_registry)


def bridge_execution_summary(state):
    if not state:
        return {}
    return {
        "run_id": state["run_id"],
        "tool_chain": list(state["tool_chain"]),
        "capability_execution_indices": dict(state["capability_indices"]),
        "executed": [
            deepcopy(state["execution_records"][index])
            for index in sorted(state["execution_records"])
        ],
        "comparisons": deepcopy(state["comparisons"]),
        "main_path": "declared_bridge",
    }
