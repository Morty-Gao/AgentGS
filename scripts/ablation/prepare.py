"""Select and freeze a diagnostic 60-question module study, without model calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import copy
from pathlib import Path

from _study import (
    DEFAULT_STUDY,
    ROOT,
    build_schedule,
    chain_group,
    data_inventory,
    digest,
    effective_policy,
    file_hash,
    load_design,
    source_snapshot,
    write_json,
)

from agentgs.agent.engine import MutablePlan
from agentgs.agent.planning.bridge import complete_shared_capability_bridges
from agentgs.agent.planning.scopes import required_instances, scope_tools
from agentgs.agent.planning.skills import builtin_planner
from agentgs.benchmark.inputs import public_input
from agentgs.client.config import load_llm
from agentgs.contracts.catalog import load_tool_scope
from agentgs.contracts.resources import RESOURCE_MARKER
from agentgs.evaluation.datasets.geotooleval import GeoToolEvalDataset


def requirement_signature(planner, plan):
    return {
        "capabilities": sorted(row["capability_id"] for row in plan.capabilities),
        "admitted_instances": sorted(
            (node["id"], sorted(node["depends_on"]))
            for node in plan.capability_expansion["instances"]
        ),
    }


def inspect_task(task, planner, available):
    request = public_input(task)
    question = request.question
    if request.input_resource is not None and RESOURCE_MARKER not in question:
        question += (
            "\n\n" + RESOURCE_MARKER + " " + json.dumps(request.input_resource, ensure_ascii=False)
        )

    def prepare(pred, completion, bridge):
        scoped = copy(planner)
        scoped.predecessor_expansion, scoped.capability_completion = pred, completion
        plan = MutablePlan(**scoped.plan(question))
        report = complete_shared_capability_bridges(
            plan, list(scoped.skills.values()), available, enabled=bridge
        )
        return plan, report

    full, bridge = prepare(True, True, True)
    a2, _ = prepare(False, True, True)
    a3, _ = prepare(True, False, False)
    a23, _ = prepare(False, False, False)
    expansion = full.capability_expansion
    candidate_scopes = [
        node["id"]
        for node in required_instances(planner.graph, full.as_dict())
        if len(
            set(scope_tools(planner.graph, node["skill_id"], node["capability_id"]))
            & set(available)
        )
        > 1
    ]
    reference_tools = [step["tool_name"] for step in task.gold_steps]
    features = ["answer:" + task.answer_type]
    features += ["tool:" + tool for tool in sorted(set(reference_tools))]
    features += ["capability:" + row["capability_id"] for row in full.capabilities]
    flags = {
        "predecessor_insertion": bool(expansion.get("predecessor_added")),
        "support_insertion": bool(expansion.get("support_added")),
        "bridge_insertion": bool(bridge["accepted"]),
        "multiple_candidates": bool(candidate_scopes),
        "repeated_reference_tool": len(reference_tools) > len(set(reference_tools)),
    }
    features += ["mechanism:" + key for key, value in flags.items() if value]
    return {
        "question_id": task.question_id,
        "category_id": task.category_id,
        "category": task.category,
        "chain_group": chain_group(task),
        "gold_step_count": len(task.gold_steps),
        "flags": flags,
        "features": sorted(set(features)),
        "candidate_scopes": candidate_scopes,
        "a2_final_requirements_changed": requirement_signature(planner, full)
        != requirement_signature(planner, a2),
        "a3_final_requirements_changed": requirement_signature(planner, full)
        != requirement_signature(planner, a3),
        "a2_given_a3_requirements_changed": requirement_signature(planner, a3)
        != requirement_signature(planner, a23),
        "a2_compensated_predecessors": a2.capability_expansion.get("compensated_predecessors", []),
        "bridge_rejections": dict(Counter(row["reason"] for row in bridge["rejected"])),
    }


def quotas(rows, per_domain=6, single_controls=1, long_target=3):
    counts = Counter(row["chain_group"] for row in rows)
    if len(rows) < per_domain:
        raise ValueError("A domain cannot fill its quota")
    result = {"single": 0, "short": 0, "long": 0}
    result["single"] = (
        min(single_controls, counts["single"]) if counts["short"] + counts["long"] else per_domain
    )
    result["long"] = min(long_target, counts["long"], per_domain - result["single"])
    result["short"] = min(counts["short"], per_domain - sum(result.values()))
    for group in ("long", "short", "single"):
        result[group] += min(counts[group] - result[group], per_domain - sum(result.values()))
    return result


def choose(rows, selection):
    selected, reasons, covered = [], {}, set()
    groups = defaultdict(list)
    for row in rows:
        groups[row["category_id"]].append(row)
    for domain, population in sorted(groups.items()):
        target = quotas(
            population,
            selection["per_domain"],
            selection["single_step_controls_per_mixed_domain"],
            selection["long_chains_per_domain"],
        )
        for group in ("long", "short", "single"):
            pool = [row for row in population if row["chain_group"] == group]
            for _ in range(target[group]):

                def rank(row):
                    novelty = sum(
                        8
                        if feature.startswith("mechanism:")
                        else 3
                        if feature.startswith("capability:")
                        else 2
                        if feature.startswith("tool:")
                        else 1
                        for feature in set(row["features"]) - covered
                    )
                    return novelty, sum(row["flags"].values()), -row["question_id"]

                best = max(pool, key=rank)
                novelty = sorted(set(best["features"]) - covered)
                selected.append(best["question_id"])
                reasons[str(best["question_id"])] = {
                    "domain_quota": target,
                    "stratum": group,
                    "new_features": novelty,
                    "selection": "deterministic_greedy_coverage",
                }
                covered.update(best["features"])
                pool.remove(best)
    return sorted(selected), reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/geotooleval.jsonl")
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Optional manually curated JSON list of exactly 60 IDs, six per domain",
    )
    parser.add_argument("--model", help="llm.toml model key; required when several models are configured")
    args = parser.parse_args()
    directory = args.study.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise SystemExit(
            "Study directory is not empty. Use a fresh directory; frozen designs are not overwritten."
        )
    design = load_design()
    before = source_snapshot(ROOT)
    tasks = GeoToolEvalDataset(args.source).load()
    if len({row.category_id for row in tasks}) != 10:
        raise ValueError("Selection requires all ten domains")
    planner, available = builtin_planner(), load_tool_scope()["tool_ids"]
    audit = [inspect_task(row, planner, available) for row in tasks]
    selected, reasons = choose(audit, design["selection"])
    if args.ids_file:
        selected = json.loads(args.ids_file.read_text())
        if not isinstance(selected, list) or any(type(qid) is not int for qid in selected):
            raise ValueError("Manual IDs must be a JSON list of integers")
        selected = sorted(selected)
        reasons = {
            str(qid): {"selection": "manual", "ids_file_sha256": file_hash(args.ids_file)}
            for qid in selected
        }
    by_id = {row.question_id: row for row in tasks}
    if len(selected) != 60 or len(set(selected)) != 60 or not set(selected) <= set(by_id):
        raise ValueError("Expected exactly 60 unique known IDs")
    if set(Counter(by_id[qid].category_id for qid in selected).values()) != {6}:
        raise ValueError("Expected six selected questions in each domain")
    chosen = [row for row in audit if row["question_id"] in selected]
    pilot = [
        max(
            (row for row in chosen if row["category_id"] == domain),
            key=lambda row: (
                sum(row["flags"].values()),
                row["gold_step_count"],
                -row["question_id"],
            ),
        )["question_id"]
        for domain in range(1, 11)
    ]
    config = load_llm(ROOT)
    key, model = config.configured_model(args.model)
    provider = config.providers[model.provider]
    library = list(planner.skills.values())
    eligibility = {
        "A4": {
            "status": "eligible_offline_requires_runtime_confirmation"
            if any(row["flags"]["bridge_insertion"] for row in chosen)
            else "deferred",
            "reason": "requires accepted bridges in the frozen production declarations and selected public tasks",
            "accepted_tasks": sum(row["flags"]["bridge_insertion"] for row in chosen),
            "operations_with_explicit_io": sum(
                bool(op.get("input_types")) and bool(op.get("output_types"))
                for skill in library
                for op in skill["workflow"]["steps"]
            ),
            "operations_with_bridge_bindings": sum(
                bool(op.get("bridge_input_bindings"))
                for skill in library
                for op in skill["workflow"]["steps"]
            ),
        },
        "A2": {
            "final_requirements_changed": sum(
                row["a2_final_requirements_changed"] for row in chosen
            ),
            "with_completion_off_changed": sum(
                row["a2_given_a3_requirements_changed"] for row in chosen
            ),
        },
        "A6": {
            "status": "requires_runtime_failures",
            "conditioning": "report all tasks; natural failure subset is descriptive only",
        },
    }
    cells = [
        {**cell, "effective_algorithm": effective_policy(ROOT, design, cell)}
        for cell in design["cells"]
    ]
    for cell in cells:
        if cell["id"] == "A4":
            cell["suite"] = "primary" if eligibility["A4"]["accepted_tasks"] else "deferred"
    schedule = build_schedule(design, cells)
    raw = {
        json.loads(line)["question_id"]: line
        for line in args.source.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inputs").mkdir()
    for name, ids in (("study60", selected), ("pilot10", pilot)):
        (directory / f"inputs/{name}.jsonl").write_text(
            "".join(raw[qid] + "\n" for qid in sorted(ids))
        )
    write_json(directory / "inputs/selected_ids.json", selected)
    write_json(
        directory / "selection.json",
        {
            "source": str(args.source.resolve()),
            "source_sha256": file_hash(args.source),
            "question_ids": selected,
            "pilot_ids": pilot,
            "counts": {
                str(domain): dict(
                    Counter(row["chain_group"] for row in chosen if row["category_id"] == domain)
                )
                for domain in range(1, 11)
            },
            "reasons": reasons,
            "selected_audit": chosen,
            "population_audit": audit,
        },
    )
    after = source_snapshot(ROOT)
    if before != after:
        write_json(
            directory / "preparation-conflict.json",
            {
                "changed_files": sorted(
                    path
                    for path in before.keys() | after.keys()
                    if before.get(path) != after.get(path)
                )
            },
        )
        raise RuntimeError(
            "Source changed during preparation; discard this incomplete directory and retry"
        )
    study = {
        "root": str(ROOT),
        "design": design,
        "cells": cells,
        "schedule": schedule,
        "question_ids": selected,
        "pilot_ids": sorted(pilot),
        "eligibility": eligibility,
        "model": {
            "key": key,
            **model.model_dump(mode="json"),
            "base_url_sha256": digest(provider.base_url),
        },
        "source_snapshot": before,
        "data_inventory": data_inventory(ROOT),
        "data_inventory_kind": "size_and_mtime_not_payload_checksums; raw_and_symlink_trees_excluded",
        "frozen_inputs": {
            name: file_hash(directory / name)
            for name in (
                "inputs/study60.jsonl",
                "inputs/pilot10.jsonl",
                "inputs/selected_ids.json",
                "selection.json",
            )
        },
    }
    study["study_id"] = digest(study)
    write_json(directory / "study.json", study)
    print(
        json.dumps(
            {
                "study": str(directory),
                "study_id": study["study_id"],
                "questions": len(selected),
                "chain_groups": dict(Counter(row["chain_group"] for row in chosen)),
                "pilot_episodes": len(pilot) * sum(cell["suite"] == "primary" for cell in cells),
                "primary_episodes": 60 * 3 * sum(cell["suite"] == "primary" for cell in cells),
                "interaction_episodes": 60
                * 3
                * sum(cell["suite"] == "interaction" for cell in cells),
                "eligibility": eligibility,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
