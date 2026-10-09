"""Score an AgentGS run.

The answer is the text returned by the answering stage. Marker text inside
the conversation is not scanned again. Tool calls and model usage come from
the saved conversation: the final attempt's tools, and model events from
every attempt. ``data_only`` is scored on FAS only.
"""

import sys
from pathlib import Path

import typer

_SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPTS / "evaluation"))

from _common import load_predictions, planning_mode, score_records  # noqa: E402


def prepare(record: dict):
    answer = record.get("final_answer", record.get("answer", "")) or ""
    return (
        answer,
        record.get("conversation", record.get("messages", [])),
        {
            "agent": "agentgs",
            "answer_extraction": {"method": "agent_answer"},
            "tool_match": "conversation",
        },
    )


def finish_for(mode: str | None):
    def finish(report: dict) -> None:
        composite = report.get("composite_metrics") or {}
        report["agent"] = "agentgs"
        report["planning_mode"] = mode
        if mode == "data_only":
            report["primary_metric"] = "fas"
            report["tool_match"] = "not_applicable"
            report["comparison"] = {
                "primary_metric": "fas",
                "micro_fas": report.get("micro_fas"),
                "macro_fas": report.get("macro_fas"),
            }
            return
        report["primary_metric"] = "final_score"
        report["tool_match"] = "conversation"
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

    return finish


def evaluate(predictions: Path, output: Path, dataset, *, allow_partial=False, project=None):
    mode = planning_mode(Path(predictions))
    return score_records(
        dataset,
        load_predictions(predictions),
        output,
        prepare=prepare,
        finish=finish_for(mode),
        allow_partial=allow_partial,
        project=project,
    )


def main(
    predictions: Path = typer.Argument(..., help="AgentGS predictions.jsonl"),
    project: Path = typer.Option(Path.cwd(), "--project"),
    output: Path = typer.Option(Path("outputs/evaluations/agentgs/scores.json"), "--output"),
    dataset: Path | None = typer.Option(None, "--dataset"),
    dataset_name: str | None = typer.Option(None, "--dataset-name"),
    allow_partial: bool = typer.Option(False, "--allow-partial/--no-allow-partial"),
) -> None:
    """Score AgentGS predictions. This does not call the model."""
    from agentgs.cli.parsing import select_dataset, under_project
    from agentgs.config.paths import project_root

    root = project_root(project)
    destination = under_project(root, output)
    report = evaluate(
        predictions,
        destination,
        select_dataset(root, dataset, dataset_name),
        allow_partial=allow_partial,
        project=root,
    )
    typer.echo(str(report["primary_metric"]) + " written to " + str(destination))


if __name__ == "__main__":
    typer.run(main)
