"""Paired question-level analysis; absent experiment cells are never scored as zero."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from _study import DEFAULT_STUDY, file_hash, load_study, worker_only, write_json
from run import entries

from agentgs.evaluation.analysis.conversation import as_conversation, tool_executions


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def macro(values, domains):
    grouped = defaultdict(list)
    for qid, value in values.items():
        grouped[domains[qid]].append(value)
    return mean(mean(group) for group in grouped.values())


def paired_interval(terms, domains, *, seed, samples):
    ids = sorted(terms[0][1])
    if any(set(values) != set(ids) for _, values in terms):
        raise ValueError("Paired estimates require identical question IDs")
    delta = {qid: sum(weight * values[qid] for weight, values in terms) for qid in ids}
    grouped = defaultdict(list)
    for qid in ids:
        grouped[domains[qid]].append(delta[qid])
    rng = np.random.default_rng(seed)
    estimates = np.zeros(samples)
    for domain in sorted(grouped):
        values = np.asarray(grouped[domain], dtype=float)
        estimates += values[rng.integers(0, len(values), (samples, len(values)))].mean(
            axis=1
        ) / len(grouped)
    return {
        "delta_pp": 100 * macro(delta, domains),
        "ci95_pp": (100 * np.quantile(estimates, [0.025, 0.975])).tolist(),
        "questions": len(ids),
        "domains": len(grouped),
        "method": "within_domain_paired_question_bootstrap_after_repeat_averaging",
        "interpretation": "descriptive_interval_on_purposeful_panel; no_population_or_multiplicity_adjusted_significance_claim",
    }


def trace_summary(record):
    conversation = as_conversation(record.get("conversation"))
    metrics = Counter()
    proposed = {}
    for message in conversation.messages:
        if message.role == "assistant":
            for call in message.tool_calls:
                if call.function.name == "submit_tool_plan":
                    try:
                        steps = json.loads(call.function.arguments)["steps"]
                        proposed = {step["id"]: step["tool"] for step in steps}
                    except (ValueError, KeyError, TypeError):
                        pass
        if message.role != "event":
            continue
        event = message.metadata
        kind = event.get("type")
        if kind == "module_policy":
            metrics["policy_recorded"] = 1
        elif kind == "capability_expansion":
            metrics["capability_trace_recorded"] = 1
            expansion = event.get("content", {})
            for field in ("predecessor_added", "support_added", "compensated_predecessors"):
                metrics[field] += len(expansion.get(field, []))
        elif kind == "shared_capability_bridge_search":
            metrics["accepted_bridges"] += len(event.get("content", {}).get("accepted", []))
        elif kind == "tool_selection":
            for decision in event.get("instances", []):
                metrics["selection_decisions"] += 1
                if decision["instance_id"] in proposed:
                    metrics["selection_changes"] += (
                        decision["selected_tool"] != proposed[decision["instance_id"]]
                    )
                else:
                    metrics["selection_change_unknown"] += 1
        elif kind == "plan.created":
            metrics["plans_created"] += 1
        elif kind == "plan_resume":
            metrics["reused_instances"] += 1
        elif kind == "execution_feedback" and event.get("status") == "recovery_disabled":
            metrics["recovery_disabled_events"] += 1
    metrics["failed_tool_attempt_events"] = sum(
        event.get("status") != "success" for event in tool_executions(conversation)
    )
    provenance = record.get("provenance", {})
    cost = provenance.get("execution_cost", {})
    successful = {
        event.get("call_id")
        for event in tool_executions(conversation)
        if event.get("status") == "success"
    }
    metrics["successful_bridge_bound_calls"] = sum(
        message.role == "event"
        and message.metadata.get("shared_bridge_binding") is True
        and message.metadata.get("call_id") in successful
        for message in conversation.messages
    )
    return {
        **dict(metrics),
        "physical_tool_calls": cost.get("tool_executor_requests"),
        "model_requests": cost.get("model_requests"),
        "http_attempts": provenance.get("budget", {}).get("http_attempts"),
    }


def read_cell(directory, study, cell, required_entries, ids):
    scores, traces, costs, repeats = defaultdict(list), [], [], []
    domains = {}
    missing = []
    for entry in required_entries:
        folder = directory / "runs" / entry["run_id"]
        receipt_path = folder / "receipt.json"
        if not receipt_path.is_file():
            missing.append(entry["run_id"])
            continue
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["study_id"] != study["study_id"]
            or receipt["cell"] != cell
            or receipt["repeat"] != entry["repeat"]
        ):
            raise ValueError("Run receipt belongs to a different experiment cell")
        for name in ("scores", "predictions"):
            path = folder / (name + (".json" if name == "scores" else ".jsonl"))
            if file_hash(path) != receipt[name + "_sha256"]:
                raise ValueError(f"Run artifact changed: {path}")
        report = json.loads((folder / "scores.json").read_text())
        predictions = [
            json.loads(line)
            for line in (folder / "predictions.jsonl").read_text().splitlines()
            if line.strip()
        ]
        rows = report["results"]
        if len(rows) != len(ids) or {row["question_id"] for row in rows} != set(ids):
            raise ValueError("Score report is incomplete or contains duplicate IDs")
        if len(predictions) != len(ids) or {row["question_id"] for row in predictions} != set(ids):
            raise ValueError("Prediction report is incomplete or contains duplicate IDs")
        repetition = {}
        for row in rows:
            qid = row["question_id"]
            fas = row["scores"]["fas"]
            if not isinstance(fas, (int, float)) or not np.isfinite(fas) or not 0 <= fas <= 1:
                raise ValueError("FAS must be finite and within [0, 1]")
            scores[qid].append(row["scores"])
            repetition[qid] = fas
            domains[qid] = row["category_id"]
        repeats.append(macro(repetition, domains))
        for prediction in predictions:
            traces.append(
                {
                    "question_id": prediction["question_id"],
                    "repeat": entry["repeat"],
                    "failed_run": prediction["status"] != "succeeded",
                    "execution_completed": prediction.get("tool_chain_completed") is True,
                    "reported_completed": prediction.get("completion_status") == "Completed",
                    **trace_summary(prediction),
                }
            )
        costs += [row["scores"] for row in rows]
    if missing:
        return {"cell": cell, "status": "pending", "missing_runs": missing}, None
    means = {qid: mean(row["fas"] for row in values) for qid, values in scores.items()}
    result = {
        "cell": cell,
        "status": "complete",
        "episodes": len(costs),
        "macro_fas": macro(means, domains),
        "micro_fas": mean(means.values()),
        "repeat_macro_fas": repeats,
        "repeat_macro_fas_sd": float(np.std(repeats, ddof=1)) if len(repeats) > 1 else None,
        "mean_tsf1": mean(row["tsf1"] for row in costs),
        "mean_tos": mean(row["tos"] for row in costs),
        "failed_run_rate": mean(row["failed_run"] for row in traces),
        "execution_completion_rate": mean(row["execution_completed"] for row in traces),
        "reported_completion_rate": mean(row["reported_completed"] for row in traces),
        "mechanism_trace_coverage": mean(
            bool(row.get("policy_recorded")) and bool(row.get("capability_trace_recorded"))
            for row in traces
        ),
        "known_token_usage_fraction": mean(
            row.get("token_usage_complete") is True for row in costs
        ),
        "mean_tokens_when_usage_complete": mean(
            row["model_total_tokens"] for row in costs if row.get("token_usage_complete") is True
        ),
        "trace_totals": {
            key: sum(row.get(key, 0) or 0 for row in traces)
            for key in (
                "predecessor_added",
                "support_added",
                "compensated_predecessors",
                "accepted_bridges",
                "successful_bridge_bound_calls",
                "selection_changes",
                "selection_change_unknown",
                "failed_tool_attempt_events",
                "reused_instances",
            )
        },
        "mean_model_requests_observed": mean(
            row["model_requests"] for row in traces if row["model_requests"] is not None
        ),
        "mean_physical_calls_observed": mean(
            row["physical_tool_calls"] for row in traces if row["physical_tool_calls"] is not None
        ),
        "per_episode_mechanisms": traces,
    }
    return result, means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--suite", choices=["pilot", "primary", "all"], default="primary")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    worker_only()
    directory = args.study.resolve()
    study = load_study(directory, check_source=False)
    selection = json.loads((directory / "selection.json").read_text())
    meta = {row["question_id"]: row for row in selection["selected_audit"]}
    ids = study["pilot_ids"] if args.suite == "pilot" else study["question_ids"]
    domains = {qid: meta[qid]["category_id"] for qid in ids}
    scheduled = entries(study, args.suite, None, None)
    grouped = defaultdict(list)
    for entry in scheduled:
        grouped[entry["cell"]].append(entry)
    cells, means = {}, {}
    for cell, expected in grouped.items():
        result, average = read_cell(directory, study, cell, expected, ids)
        if average is not None:
            means[cell] = average
            multi = [qid for qid in ids if meta[qid]["gold_step_count"] > 1]
            result["multistep_questions"] = len(multi)
            result["multistep_micro_fas"] = mean(average[qid] for qid in multi)
            result["domains"] = {
                str(domain): {
                    "questions": sum(domains[qid] == domain for qid in ids),
                    "fas": mean(value for qid, value in average.items() if domains[qid] == domain),
                }
                for domain in sorted(set(domains.values()))
            }
        cells[cell] = result
    contrasts, interactions = {}, {}
    params = {
        "domains": domains,
        "seed": study["design"]["bootstrap_seed"],
        "samples": study["design"]["bootstrap_samples"],
    }
    if "Full" in means and args.suite != "pilot":
        for cell in means:
            if cell != "Full":
                contrasts[cell] = paired_interval([(1, means["Full"]), (-1, means[cell])], **params)
                if cell == "A4":
                    contrasts[cell]["bridge_effect_observed_in_full"] = (
                        cells["Full"]["trace_totals"]["successful_bridge_bound_calls"] > 0
                    )
        for a, b, ab in study["design"]["interactions"]:
            if {a, b, ab} <= means.keys():
                interactions[ab] = {
                    "interaction": paired_interval(
                        [(1, means["Full"]), (-1, means[a]), (-1, means[b]), (1, means[ab])],
                        **params,
                    ),
                    "effect_of_a_when_b_disabled": paired_interval(
                        [(1, means[b]), (-1, means[ab])], **params
                    ),
                    "formula": f"Y_Full - Y_{a} - Y_{b} + Y_{ab}",
                }
    report = {
        "study_id": study["study_id"],
        "suite": args.suite,
        "primary_metric": "domain_macro_fas",
        "analysis_sha256": file_hash(__file__),
        "complete": all(row["status"] == "complete" for row in cells.values()),
        "population": "purposefully selected 60-question diagnostic panel, not a random sample of 603",
        "eligibility": study["eligibility"],
        "cells": cells,
        "full_minus_variant": contrasts,
        "interactions": interactions,
    }
    write_json(directory / f"analysis-{args.suite}.json", report)
    with (directory / f"analysis-{args.suite}.csv").open("w", newline="") as stream:
        keys = [
            "cell",
            "status",
            "macro_fas",
            "micro_fas",
            "multistep_micro_fas",
            "repeat_macro_fas_sd",
            "mean_tsf1",
            "mean_tos",
            "failed_run_rate",
        ]
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells.values())
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "completed_cells": list(means),
                "pending_cells": [cell for cell in cells if cell not in means],
                "report": str(directory / f"analysis-{args.suite}.json"),
            },
            indent=2,
        )
    )
    if args.require_complete and not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
