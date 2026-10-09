"""Score a GeoAgent search run.

Domain metrics use the selected execution. A failed search uses the last
attempted execution instead. Tool calls from pruned branches are ignored.
Model usage on the prediction is kept, including those branches and retries.
An unmarked answer is accepted only as the suffix of that selected execution.
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


def _executions(trace) -> list[dict]:
    return [event for event in trace.get("events") or [] if event.get("phase") == "execute"]


def _selected(trace) -> dict | None:
    events = _executions(trace)
    selected = trace.get("selected_execution_id")
    if selected:
        match = next((event for event in events if event.get("execution_id") == selected), None)
        if match is not None:
            return match
    return events[-1] if events else None


def _calls(event: dict | None) -> list[dict]:
    calls = []
    for receipt in (event or {}).get("calls") or []:
        arguments = _object(receipt.get("arguments"))
        calls.append(
            {
                "call_id": receipt.get("call_id") or uuid4().hex,
                "name": receipt.get("name") or "",
                "arguments": arguments,
                "bound_arguments": _object(receipt.get("bound_arguments", arguments)),
                "result": receipt.get("result"),
                "status": receipt.get("status") or "success",
            }
        )
    return calls


def _answer(trace, record) -> tuple[str, dict]:
    event = _selected(trace)
    saved = str(record.get("final_answer", record.get("answer", "")) or "")
    extraction = dict(record.get("answer_extraction") or {})
    if event is None:
        return saved, extraction or {"answer": saved, "method": "saved_answer"}
    stdout = event.get("stdout") or ""
    marked = extract_answer(stdout)
    if marked.get("answer"):
        return marked["answer"], marked
    # Unmarked output counts only when this execution's own suffix is the saved answer.
    if extraction.get("method") == "terminal_stdout" and saved and stdout.endswith(saved):
        return saved, extraction
    return "", {"answer": "", "method": "missing" if not saved else "rejected_unselected_output"}


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
    if str(trace.get("schema", "")).startswith("geoagent-mcts") and trace.get("events"):
        answer, extraction = _answer(trace, record)
        view = _with_usage(
            trace.get("prompt") or "question",
            _calls(_selected(trace)),
            record.get("conversation"),
        )
        return (
            answer,
            view,
            {
                "agent": "geoagent",
                "answer_extraction": extraction,
                "tool_match": "selected_execution",
            },
        )
    answer = record.get("final_answer", record.get("answer", "")) or ""
    return (
        answer,
        record.get("conversation", record.get("messages", [])),
        {
            "agent": "geoagent",
            "answer_extraction": {"method": "saved_answer"},
            "tool_match": "saved_conversation",
        },
    )


def finish(report: dict) -> None:
    composite = report.get("composite_metrics") or {}
    report["agent"] = "geoagent"
    report["primary_metric"] = "final_score"
    report["tool_match"] = "selected_execution"
    report["comparison"] = {
        "primary_metric": "final_score",
        "micro_fas": report.get("micro_fas"),
        "macro_fas": report.get("macro_fas"),
        "accuracy_score": composite.get("accuracy_score"),
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
    predictions: Path = typer.Argument(..., help="GeoAgent predictions.jsonl"),
    project: Path = typer.Option(Path.cwd(), "--project"),
    output: Path = typer.Option(Path("outputs/evaluations/geoagent/scores.json"), "--output"),
    dataset: Path | None = typer.Option(None, "--dataset"),
    dataset_name: str | None = typer.Option(None, "--dataset-name"),
    allow_partial: bool = typer.Option(False, "--allow-partial/--no-allow-partial"),
) -> None:
    """Score GeoAgent predictions. This does not call the model."""
    from agentgs.cli.parsing import select_dataset, under_project
    from agentgs.config.paths import project_root

    root = project_root(project)
    destination = under_project(root, output)
    evaluate(
        predictions,
        destination,
        select_dataset(root, dataset, dataset_name),
        allow_partial=allow_partial,
        project=root,
    )
    typer.echo("selected_execution scores written to " + str(destination))


if __name__ == "__main__":
    typer.run(main)
