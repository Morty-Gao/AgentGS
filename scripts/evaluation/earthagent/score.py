"""Score an Earth-Agent ReAct run.

The final assistant turn may be plain text. Earlier turns count only as tool
calls. Tool results are the observations that reached the scientific adapter
on that final attempt. Model usage already aggregated onto the prediction is
kept, including earlier retries.
"""

import sys
from pathlib import Path
from uuid import uuid4

import typer

_SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPTS / "benchmark"))
sys.path.insert(0, str(_SCRIPTS / "evaluation"))

from _answers import extract_answer  # noqa: E402
from _common import load_predictions, score_records  # noqa: E402
from _runtime import conversation  # noqa: E402

from agentgs.evaluation.analysis.conversation import as_conversation  # noqa: E402


def _object(value):
    if isinstance(value, dict):
        return value
    return {} if value is None else {"value": value}


def _calls(trace) -> list[dict]:
    calls = []
    for turn in trace.get("turns") or []:
        for observation in turn.get("observations") or []:
            arguments = _object(observation.get("arguments"))
            calls.append(
                {
                    "call_id": observation.get("tool_call_id") or uuid4().hex,
                    "name": observation.get("name") or "",
                    "arguments": arguments,
                    "bound_arguments": _object(observation.get("bound_arguments", arguments)),
                    "result": observation.get("result"),
                    "status": observation.get("status") or "success",
                }
            )
    return calls


def _answer(trace) -> tuple[str, dict]:
    """Only a turn with no tool calls is a final answer. Plain text is accepted."""
    for turn in reversed(trace.get("turns") or []):
        if turn.get("observations"):
            continue
        content = (turn.get("assistant") or {}).get("content") or ""
        extracted = extract_answer(content, allow_plain=True)
        return extracted["answer"], extracted
    return "", {"answer": "", "method": "missing"}


def _with_usage(prompt: str, calls: list[dict], saved):
    rebuilt = as_conversation(conversation(prompt, [], calls=calls))
    for message in as_conversation(saved).messages:
        if message.role != "event":
            continue
        if not str(message.metadata.get("type", "")).startswith("model."):
            continue
        cloned = message.model_copy(deep=True)
        cloned.id = uuid4().hex
        rebuilt.append(cloned)
    return rebuilt


def prepare(record: dict):
    trace = record.get("trace") or {}
    if str(trace.get("schema", "")).startswith("earth-agent-react") and trace.get("turns"):
        answer, extraction = _answer(trace)
        view = _with_usage(
            trace.get("prompt") or "question", _calls(trace), record.get("conversation")
        )
        return (
            answer,
            view,
            {
                "agent": "earthagent",
                "answer_extraction": extraction,
                "tool_match": "react_final_attempt",
            },
        )
    answer = record.get("final_answer", record.get("answer", "")) or ""
    return (
        answer,
        record.get("conversation", record.get("messages", [])),
        {
            "agent": "earthagent",
            "answer_extraction": {"method": "saved_answer"},
            "tool_match": "saved_conversation",
        },
    )


def finish(report: dict) -> None:
    composite = report.get("composite_metrics") or {}
    report["agent"] = "earthagent"
    report["primary_metric"] = "final_score"
    report["tool_match"] = "react_final_attempt"
    report["comparison"] = {
        "primary_metric": "final_score",
        "micro_fas": report.get("micro_fas"),
        "macro_fas": report.get("macro_fas"),
        "quality_score": composite.get("quality_score"),
        "final_score": composite.get("final_score"),
        "tsf1": composite.get("tsf1"),
        "tos": composite.get("tos"),
        "pas": composite.get("pas"),
    }


def evaluate(predictions: Path, output: Path, dataset, *, allow_partial=False, project=None):
    return score_records(
        dataset,
        load_predictions(predictions),
        output,
        prepare=prepare,
        finish=finish,
        allow_partial=allow_partial,
        project=project,
    )


def main(
    predictions: Path = typer.Argument(..., help="Earth-Agent predictions.jsonl"),
    project: Path = typer.Option(Path.cwd(), "--project"),
    output: Path = typer.Option(Path("outputs/evaluations/earthagent/scores.json"), "--output"),
    dataset: Path | None = typer.Option(None, "--dataset"),
    dataset_name: str | None = typer.Option(None, "--dataset-name"),
    allow_partial: bool = typer.Option(False, "--allow-partial/--no-allow-partial"),
) -> None:
    """Score Earth-Agent predictions. This does not call the model."""
    from agentgs.cli.parsing import select_dataset, under_project
    from agentgs.config.paths import project_root

    root = project_root(project)
    report = evaluate(
        predictions,
        under_project(root, output),
        select_dataset(root, dataset, dataset_name),
        allow_partial=allow_partial,
        project=root,
    )
    typer.echo(
        report["comparison"]["primary_metric"] + " written to " + str(under_project(root, output))
    )


if __name__ == "__main__":
    typer.run(main)
