"""Compose answer and tool metrics for one example, then aggregate a run."""

from agentgs.agent.answering.formats import check_fixed_answer
from agentgs.evaluation.analysis.conversation import as_conversation
from agentgs.evaluation.datasets.base import Example
from agentgs.evaluation.metrics.answer_parsing import json_safe
from agentgs.evaluation.metrics.answers import ANSWER_PROTOCOL, score_answer
from agentgs.evaluation.metrics.legacy_answers import score_answer as legacy_score_answer
from agentgs.evaluation.metrics.tools import SCORING_PROTOCOL, aggregate_scores, score_task


def score_prediction(task: Example, answer, messages=(), usage=None):
    """Score one saved answer against its gold example and conversation."""
    final = score_answer(task, answer)
    scores = score_task(
        gold_steps=list(task.gold_steps),
        messages=as_conversation(messages),
        answer_score=final["score"],
        token_usage=usage,
    )
    return {
        "scoring_protocol": SCORING_PROTOCOL,
        "question_id": task.question_id,
        "category_id": task.category_id,
        "category": task.category,
        "answer_score": final,
        "legacy_answer_score": json_safe(legacy_score_answer(task, answer)),
        "answer_format": check_fixed_answer(answer, {"answer_format": task.answer_format})
        if task.answer_format
        else None,
        "scores": scores,
        "gold_steps": list(task.gold_steps),
    }


def summarize(rows, total=603, *, domain_count: int | None = 10):
    """Micro and macro final-answer scores plus the composite metric block.

    ``domain_count`` is the number of domains a complete run must cover.
    ``None`` accepts whatever domains are present in ``rows``.
    """
    grouped = {}
    for row in rows:
        grouped.setdefault(row["category_id"], []).append(row["scores"]["fas"])
    micro = sum(row["scores"]["fas"] for row in rows) / len(rows) if rows else None
    macro = (
        sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)
        if grouped
        else None
    )
    domains_complete = domain_count is None or len(grouped) == domain_count
    format_rows = [row["answer_format"] for row in rows if row.get("answer_format")]
    legacy_rows = [
        row["legacy_answer_score"]["score"] for row in rows if row.get("legacy_answer_score")
    ]
    return {
        "scoring_protocol": SCORING_PROTOCOL,
        "answer_scoring_protocol": ANSWER_PROTOCOL,
        "legacy_micro_fas": sum(legacy_rows) / len(legacy_rows) if legacy_rows else None,
        "format_checked": len(format_rows),
        "format_valid_rate": sum(row["valid"] for row in format_rows) / len(format_rows)
        if format_rows
        else None,
        "answer_review_count": sum(
            bool(row.get("answer_score", {}).get("details", {}).get("needs_review")) for row in rows
        ),
        "submitted": len(rows),
        "expected": total,
        "complete": len(rows) == total and domains_complete,
        "micro_fas": micro,
        "macro_fas": macro,
        "full_score_rate": sum(row["scores"]["fas"] == 1 for row in rows) / len(rows)
        if rows
        else None,
        "domains": {
            str(key): {"n": len(values), "fas": sum(values) / len(values)}
            for key, values in grouped.items()
        },
        "composite_metrics": aggregate_scores(rows),
        "results": rows,
    }
