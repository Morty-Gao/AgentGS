"""Find one declared directed Skill branch between two active Skill frontiers.

Skill membership makes a branch discoverable; it is never itself a data-flow
edge. Only ``depends_on`` and ``after_if_present`` declarations establish its
direction. This module is read-only and does not select or execute tools.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .hypergraph import CapabilityExpansion, CapabilityNode, SkillHypergraph


def _data_types(values: Iterable[str]) -> set[str]:
    return {
        token
        for value in values
        for token in re.split("[|,/ ]+", str(value).casefold())
        if token and token not in {"parameter", "parameters", "any", "unknown"}
    }


def _compatible(source: CapabilityNode, target: CapabilityNode) -> bool:
    return bool(_data_types(source.output_types) & _data_types(target.input_types))


def _adjacency(edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        result[source].add(target)
    return result


def _paths(adjacency: dict[str, set[str]], source: str, target: str) -> list[list[str]]:
    """Enumerate simple paths inside one small authored hyperedge only."""
    result: list[list[str]] = []
    stack = [(source, [source])]
    while stack:
        current, path = stack.pop()
        for successor in sorted(adjacency.get(current, ()), reverse=True):
            if successor in path:
                continue
            following = [*path, successor]
            if successor == target:
                result.append(following)
            else:
                stack.append((successor, following))
    return result


def _has_indirect_path(adjacency: dict[str, set[str]], source: str, target: str) -> bool:
    """Reachability excluding the single direct edge; linear in active graph."""
    frontier = [item for item in adjacency.get(source, ()) if item != target]
    seen = {source}
    while frontier:
        current = frontier.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(adjacency.get(current, ()))
    return False


def _cycle_nodes(edges: Iterable[tuple[str, str]]) -> set[str]:
    adjacency = _adjacency(edges)
    visiting: list[str] = []
    visited: set[str] = set()
    cyclic: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            cyclic.update(visiting[visiting.index(node) :])
            return
        if node in visited:
            return
        visiting.append(node)
        for successor in sorted(adjacency.get(node, ())):
            visit(successor)
        visiting.pop()
        visited.add(node)

    for node in sorted(set(adjacency) | {item for values in adjacency.values() for item in values}):
        visit(node)
    return cyclic


def discover_bridges(
    graph: SkillHypergraph,
    expansion: CapabilityExpansion,
    available_tools: Iterable[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Discover validated single-Skill bridges without widening the input plan.

    Returned ``data_edges`` use ``source`` / ``target``.
    Mandatory AND predecessors must already be active or lie on the proposed
    branch; absent off-branch predecessors reject the branch. Availability is
    limited to the supplied tool inventory. Scientific admission, units/CRS,
    and concrete artifact validation remain responsibilities of later layers.
    A directly compatible alternative is reported, never silently executed.
    """
    selected = [sid for sid in dict.fromkeys(expansion.skill_ids) if sid in graph.hyperedges]
    active = {node.capability_id: node for node in expansion.capabilities}
    available = None if available_tools is None else set(available_tools)
    rejected: list[dict[str, Any]] = []
    plausible: list[dict[str, Any]] = []
    if len(selected) < 2 or not active:
        return {"accepted": [], "rejected": []}

    def local_edges(skill_id: str) -> set[tuple[str, str]]:
        scopes = graph._step_scopes.get(skill_id, {})
        return {
            (dependency, target)
            for target, scope in scopes.items()
            for dependency in (*scope.get("dependencies", ()), *scope.get("after_if_present", ()))
            if dependency in scopes
        }

    initial_edges = {
        (dependency, node.capability_id)
        for node in active.values()
        for dependency in node.dependencies
        if dependency in active
    }
    initial_data_edges = {
        (dependency, node.capability_id)
        for node in active.values()
        for dependency in node.data_dependencies or node.dependencies
        if dependency in active
        and dependency not in node.control_dependencies
        and _compatible(active[dependency], node)
    }
    initial_data = _adjacency(initial_data_edges)
    frontiers: dict[str, tuple[set[str], set[str]]] = {}
    for skill_id in selected:
        members = set(graph.hyperedges[skill_id].capability_ids) & active.keys()
        edges = {
            (source, target)
            for source, target in local_edges(skill_id)
            if source in members and target in members
        }
        incoming = {target for _, target in edges}
        outgoing = {source for source, _ in edges}
        frontiers[skill_id] = (members - outgoing, members - incoming)
    for bridge_id, hyperedge in graph.hyperedges.items():
        if bridge_id in selected:
            continue
        members = set(hyperedge.capability_ids)
        declared = local_edges(bridge_id)
        adjacency = _adjacency(declared)
        scopes = graph._step_scopes.get(bridge_id, {})
        for source_skill in selected:
            for target_skill in selected:
                if source_skill == target_skill:
                    continue
                sources = sorted(frontiers[source_skill][0] & members)
                targets = sorted(frontiers[target_skill][1] & members)
                for source in sources:
                    for target in targets:
                        if source == target:
                            continue
                        base = {
                            "skill_id": bridge_id,
                            "source_skill_id": source_skill,
                            "target_skill_id": target_skill,
                            "source_capability": source,
                            "target_capability": target,
                            "capability_path": [],
                            "direct_compatible": _compatible(active[source], active[target]),
                            "data_edges": [],
                        }
                        paths = [
                            path for path in _paths(adjacency, source, target) if len(path) >= 3
                        ]
                        if not paths:
                            rejected.append({**base, "reason": "no_declared_directed_branch"})
                            continue
                        for path in paths:
                            candidate = {**base, "capability_path": path}
                            intermediate = path[1:-1]
                            if all((cid in active for cid in intermediate)):
                                rejected.append(
                                    {**candidate, "reason": "no_missing_intermediate_capability"}
                                )
                                continue
                            if _has_indirect_path(initial_data, source, target):
                                rejected.append({**candidate, "reason": "existing_data_path"})
                                continue
                            branch_nodes = set(path)
                            planned = set(active) | branch_nodes
                            edges: set[tuple[str, str]] = set(zip(path, path[1:]))
                            absent: set[str] = set()
                            for cid in path[1:]:
                                scope = scopes.get(cid, {})
                                for dependency in scope.get("dependencies", ()):
                                    if dependency not in planned:
                                        absent.add(dependency)
                                    else:
                                        edges.add((dependency, cid))
                                for dependency in scope.get("after_if_present", ()):
                                    if dependency in planned:
                                        edges.add((dependency, cid))
                            if absent:
                                rejected.append(
                                    {
                                        **candidate,
                                        "reason": "unsatisfied_and_dependency",
                                        "missing_dependencies": sorted(absent),
                                    }
                                )
                                continue
                            unavailable = []
                            for cid in sorted(branch_nodes | {a for a, _ in edges}):
                                node = graph.capabilities.get(cid)
                                tools = set(
                                    scopes.get(cid, {}).get("candidate_tools", ())
                                    if cid in branch_nodes and cid != source
                                    else active.get(cid, node).candidate_tools
                                    if node
                                    else ()
                                )
                                if not tools or (available is not None and (not tools & available)):
                                    unavailable.append(cid)
                            if unavailable:
                                rejected.append(
                                    {
                                        **candidate,
                                        "reason": "unavailable_capability_tools",
                                        "unavailable_capabilities": unavailable,
                                    }
                                )
                                continue
                            incompatible = [
                                (a, b)
                                for a, b in sorted(edges)
                                if a not in graph.capabilities
                                or b not in graph.capabilities
                                or (not _compatible(graph.capabilities[a], graph.capabilities[b]))
                            ]
                            if incompatible:
                                rejected.append(
                                    {
                                        **candidate,
                                        "reason": "incompatible_data_contract",
                                        "incompatible_edges": [list(edge) for edge in incompatible],
                                    }
                                )
                                continue
                            branch_relations = {
                                (a, b) for a, b in declared if a in planned and b in branch_nodes
                            }
                            cycles = _cycle_nodes(initial_edges | edges | branch_relations)
                            if cycles:
                                rejected.append(
                                    {
                                        **candidate,
                                        "reason": "cyclic_bridge",
                                        "cycle_capabilities": sorted(cycles),
                                    }
                                )
                                continue
                            candidate["data_edges"] = [
                                {
                                    "source": a,
                                    "target": b,
                                    "skill_id": bridge_id,
                                    "reason": "declared_bridge_dependency",
                                }
                                for a, b in sorted(edges)
                            ]
                            candidate["added_capabilities"] = [
                                cid for cid in intermediate if cid not in active
                            ]
                            candidate["comparison_required"] = candidate["direct_compatible"]
                            candidate["validation_scope"] = (
                                "declared_direction_and_semantic_io_with_available_tool_inventory"
                            )
                            candidate["artifact_contract_status"] = "requires_runtime_validation"
                            plausible.append(candidate)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in plausible:
        grouped[candidate["source_capability"], candidate["target_capability"]].append(candidate)
    accepted = []
    for candidates in grouped.values():
        routes: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            routes[candidate["skill_id"], tuple(candidate["capability_path"])].append(candidate)
        if len(routes) != 1:
            for candidates_for_route in routes.values():
                rejected.append(
                    {
                        **candidates_for_route[0],
                        "reason": "ambiguous_bridge",
                        "alternative_count": len(routes),
                    }
                )
            continue
        candidate = dict(candidates[0])
        candidate["endpoint_skill_pairs"] = [
            {"source_skill_id": source, "target_skill_id": target}
            for source, target in sorted(
                {(item["source_skill_id"], item["target_skill_id"]) for item in candidates}
            )
        ]
        accepted.append(candidate)
    combined_edges = initial_edges | {
        (edge["source"], edge["target"])
        for candidate in accepted
        for edge in candidate["data_edges"]
    }
    cycles = _cycle_nodes(combined_edges)
    if cycles:
        retained = []
        for candidate in accepted:
            if cycles & set(candidate["capability_path"]):
                rejected.append(
                    {
                        **candidate,
                        "reason": "combined_bridge_cycle",
                        "cycle_capabilities": sorted(cycles),
                    }
                )
            else:
                retained.append(candidate)
        accepted = retained
    return {"accepted": accepted, "rejected": rejected}
