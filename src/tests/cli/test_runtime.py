import json
import sys

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from agentgs.cli.app import app
from agentgs.config.settings import RuntimeSettings
from agentgs.runtime.service import TaskService


@pytest.mark.parametrize("command", ["run", "evaluate run", "evaluate benchmark", "ui serve"])
def test_runtime_options_reach_service_and_workers(project, monkeypatch, command):
    from agentgs.benchmark import cases, runner
    from agentgs.cli import app as cli
    from agentgs.evaluation.datasets import version
    from agentgs.visualization.backend import app as backend

    services = []

    class CapturedService(TaskService):
        def __init__(self, root, settings=None):
            super().__init__(root, settings=settings)
            services.append(self)

    monkeypatch.setattr(cli, "TaskService", CapturedService)
    monkeypatch.setattr(backend, "TaskService", CapturedService)

    async def evaluate(service, cases, output):
        return output

    async def run_benchmark(service, dataset, output, **kwargs):
        return {}

    def serve(application, **kwargs):
        with TestClient(application):
            pass

    monkeypatch.setattr(cases, "evaluate", evaluate)
    monkeypatch.setattr(runner, "run_benchmark", run_benchmark)
    monkeypatch.setattr(version, "benchmark_dataset", lambda root: root / "dataset.jsonl")
    monkeypatch.setattr("uvicorn.run", serve)
    # An obsolete file must never affect startup, even if it cannot be parsed.
    (project / "runtime.toml").write_text("invalid TOML = [")
    arguments = command.split()
    if command == "run":
        arguments.extend(
            ["--tool", "DeepEarthQueryUSGSCatalog", "--arguments", '{"min_magnitude":8}']
        )
    elif command == "evaluate run":
        cases = project / "cases.jsonl"
        cases.write_text("")
        arguments.append(str(cases))
    arguments.extend(
        [
            "--project",
            str(project),
            "--concurrency",
            "3",
            "--sandbox-concurrency",
            "1",
            "--sandbox-disk-limit-mb",
            "16",
            "--timeout",
            "42",
            "--rss-limit-mb",
            "512",
            "--max-total-rss-mb",
            "1024",
            "--code-timeout",
            "7",
            "--sandbox",
            "proot",
            "--allow-proot",
            "--worker-python",
            sys.executable,
        ]
    )
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert len(services) == 1
    settings = services[0].settings
    assert settings == RuntimeSettings(
        max_concurrent_tasks=3,
        max_concurrent_sandbox_calls=1,
        sandbox_disk_limit_mb=16,
        default_timeout_seconds=42,
        default_rss_limit_mb=512,
        max_total_rss_mb=1024,
        code_timeout_seconds=7,
        sandbox="proot",
        allow_proot=True,
        worker_python=sys.executable,
    )
    assert services[0].closed
    if command == "run":
        record = json.loads(result.output)
        assert record["status"] == "succeeded"
        request = project / "outputs/tasks" / record["spec"]["id"] / "request.json"
        saved = json.loads(request.read_text())
        assert saved["runtime"] == settings.model_dump()
        assert saved["spec"]["timeout_seconds"] == 42
        assert saved["spec"]["rss_limit_mb"] == 512
        task_input = saved["spec"]["input"]
        assert task_input["planning_mode"] is None
        assert task_input["allow_rule_planning"] is None
        assert task_input["plan_attempts"] is None
        assert task_input["parallel_tools"] is False
        assert saved["runtime"]["tool_backend"] == "thread"


def test_service_defaults_ignore_project_runtime_file(project):
    (project / "runtime.toml").write_text("[runtime]\nmax_concurrent_tasks=128\n")
    (project / "config.toml").write_text("[runtime]\nmax_concurrent_tasks=64\n")
    assert TaskService(project).settings == RuntimeSettings()


@pytest.mark.parametrize(
    "options", [["--concurrency", "0"], ["--timeout", "0"], ["--sandbox", "unknown"]]
)
def test_invalid_runtime_options_are_rejected_before_task_submission(project, options):
    result = CliRunner().invoke(app, ["run", "--project", str(project), *options])
    assert result.exit_code == 2
    assert not (project / "outputs").exists()


@pytest.mark.parametrize("rules", [False, True])
def test_ablation_options_reach_benchmark(project, monkeypatch, rules):
    from agentgs.benchmark import runner
    from agentgs.evaluation.datasets import version

    received = []

    async def run_benchmark(service, dataset, output, **kwargs):
        received.append(kwargs)
        return {}

    monkeypatch.setattr(runner, "run_benchmark", run_benchmark)
    monkeypatch.setattr(version, "benchmark_dataset", lambda root: root / "dataset.jsonl")
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "benchmark",
            "--project",
            str(project),
            "--planning-mode",
            "tool_only",
            "--plan-attempts",
            "2",
            "--tool-attempts",
            "4",
            "--allow-rule-planning" if rules else "--no-rule-planning",
        ],
    )
    assert result.exit_code == 0, result.output
    assert received[0]["planning_mode"] == "tool_only"
    assert received[0]["allow_rule_planning"] is rules
    assert received[0]["plan_attempts"] == 2 and received[0]["tool_attempts"] == 4


def test_separate_invocation_limits_are_forwarded_to_real_worker(project):
    arguments = [
        "run",
        "--project",
        str(project),
        "--tool",
        "DeepEarthQueryUSGSCatalog",
        "--arguments",
        '{"min_magnitude":8}',
        "--model-concurrency",
        "3",
        "--tool-concurrency",
        "2",
        "--queue-capacity",
        "5",
        "--model-timeout",
        "4",
        "--tool-timeout",
        "3",
    ]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    request = json.loads(
        (project / "outputs/tasks" / record["spec"]["id"] / "request.json").read_text()
    )
    settings = request["runtime"]
    assert settings["max_concurrent_model_calls"] == 3
    assert settings["max_concurrent_tool_calls"] == 2
    assert settings["max_pending_tasks"] == 5
    assert settings["model_timeout_seconds"] == 4
    assert settings["tool_timeout_seconds"] == 3


def test_run_forwards_task_fields(project):
    call = {
        "id": "usgs",
        "tool": "DeepEarthQueryUSGSCatalog",
        "arguments": {"min_magnitude": 8},
        "depends_on": [],
    }
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--project",
            str(project),
            "--call",
            json.dumps(call),
            "--planning-mode",
            "tool_only",
            "--allow-rule-planning",
            "--plan-attempts",
            "2",
            "--tool-attempts",
            "4",
            "--command-attempts",
            "5",
            "--parallel-tools",
            "--skill-id",
            "demo-skill",
            "--max-model-calls",
            "3",
            "--input-domain",
            "deep_earth",
            "--file",
            "data/deep_earth/usgs_source.csv",
            "--input-resource",
            '{"kind":"demo"}',
            "--output-contract",
            '{"shape":"number"}',
            "--tool-backend",
            "process",
        ],
    )
    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    request = json.loads(
        (project / "outputs/tasks" / record["spec"]["id"] / "request.json").read_text()
    )
    task = request["spec"]["input"]
    assert task["calls"] == [call]
    assert task["planning_mode"] == "tool_only"
    assert task["allow_rule_planning"] is True
    assert task["plan_attempts"] == 2
    assert task["tool_attempts"] == 4
    assert task["command_attempts"] == 5
    assert task["parallel_tools"] is True
    assert task["skill_ids"] == ["demo-skill"]
    assert task["max_model_calls"] == 3
    assert task["input_domain"] == "deep_earth"
    assert task["file_path_list"] == ["data/deep_earth/usgs_source.csv"]
    assert task["input_resource"] == {"kind": "demo"}
    assert task["output_contract"] == {"shape": "number"}
    assert request["runtime"]["tool_backend"] == "process"


def test_task_file_keeps_unspecified_fields(project):
    task_file = project / "task.json"
    task_file.write_text(
        json.dumps(
            {
                "question": "from file",
                "calls": [
                    {
                        "tool": "DeepEarthQueryUSGSCatalog",
                        "arguments": {"min_magnitude": 8},
                    }
                ],
                "planning_mode": "tool_only",
                "skill_ids": ["keep-me"],
            }
        )
    )
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--project",
            str(project),
            "--task-file",
            str(task_file),
            "--plan-attempts",
            "2",
            "--command-attempts",
            "6",
        ],
    )
    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    task = record["spec"]["input"]
    assert task["question"] == "from file"
    assert task["planning_mode"] == "tool_only"
    assert task["skill_ids"] == ["keep-me"]
    assert task["plan_attempts"] == 2
    assert task["command_attempts"] == 6
    assert task["calls"][0]["tool"] == "DeepEarthQueryUSGSCatalog"


def test_invalid_task_usage_is_rejected(project):
    mode = CliRunner().invoke(app, ["run", "--project", str(project), "--planning-mode", "nope"])
    assert mode.exit_code == 2
    arguments = CliRunner().invoke(
        app, ["run", "--project", str(project), "--arguments", '{"a": 1}']
    )
    assert arguments.exit_code == 2
    assert not (project / "outputs").exists()


def test_benchmark_defaults_and_dataset_overrides(project, monkeypatch):
    from agentgs.benchmark import runner
    from agentgs.evaluation.datasets import version
    from agentgs.evaluation.datasets.geotooleval import GeoToolEvalDataset
    from agentgs.runtime.answer_contracts import AnswerContracts

    received = []

    async def run_benchmark(service, dataset, output, **kwargs):
        received.append((dataset, output, kwargs))
        return {}

    monkeypatch.setattr(runner, "run_benchmark", run_benchmark)
    monkeypatch.setattr(version, "benchmark_dataset", lambda root: root / "dataset.jsonl")
    default = CliRunner().invoke(app, ["evaluate", "benchmark", "--project", str(project)])
    assert default.exit_code == 0, default.output
    dataset, output, kwargs = received[0]
    assert dataset == project / "dataset.jsonl"
    assert output == project / "outputs" / "evaluations" / "geotooleval"
    assert kwargs["ids"] is None
    assert kwargs["retry_failed"] is False
    assert kwargs["planning_mode"] == "skill_guided"
    assert kwargs["allow_rule_planning"] is False
    assert kwargs["plan_attempts"] is None
    assert kwargs["tool_attempts"] is None
    assert kwargs["command_attempts"] is None
    assert isinstance(kwargs["contract_provider"], AnswerContracts)

    custom = project / "custom.jsonl"
    chosen = CliRunner().invoke(
        app,
        [
            "evaluate",
            "benchmark",
            "--project",
            str(project),
            "--dataset",
            custom.name,
            "--dataset-name",
            "geotooleval",
            "--ids",
            "61,abc",
            "--retry-failed",
            "--planning-mode",
            "data_only",
            "--command-attempts",
            "9",
            "--no-answer-contracts",
            "--output",
            "outputs/evaluations/custom",
        ],
    )
    assert chosen.exit_code == 0, chosen.output
    dataset, output, kwargs = received[1]
    assert isinstance(dataset, GeoToolEvalDataset)
    assert dataset.source == custom
    assert output == project / "outputs" / "evaluations" / "custom"
    assert kwargs["ids"] == {61, "abc"}
    assert kwargs["retry_failed"] is True
    assert kwargs["planning_mode"] == "data_only"
    assert kwargs["command_attempts"] == 9
    assert kwargs["contract_provider"] is None


def test_evaluate_run_output_stays_under_project(project, monkeypatch):
    from agentgs.benchmark import cases

    seen = []

    async def evaluate(service, rows, output):
        seen.append(output)
        return output

    monkeypatch.setattr(cases, "evaluate", evaluate)
    cases_path = project / "cases.jsonl"
    cases_path.write_text("")
    result = CliRunner().invoke(
        app, ["evaluate", "run", str(cases_path), "--project", str(project)]
    )
    assert result.exit_code == 0, result.output
    assert seen == [project / "outputs" / "evaluations"]


def test_score_forwards_dataset_and_partial_flag(project, monkeypatch):
    from agentgs.benchmark import score
    from agentgs.evaluation.datasets import version

    received = {}

    def score_predictions(dataset, predictions, output, **kwargs):
        received.update(dataset=dataset, predictions=predictions, output=output, kwargs=kwargs)
        return {"summary": True}

    monkeypatch.setattr(score, "score_predictions", score_predictions)
    monkeypatch.setattr(version, "benchmark_dataset", lambda root: root / "dataset.jsonl")
    predictions = project / "predictions.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "score",
            str(predictions),
            "--project",
            str(project),
            "--allow-partial",
            "--output",
            "outputs/evaluations/scored.json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert received["dataset"] == project / "dataset.jsonl"
    assert received["predictions"] == predictions
    assert received["output"] == project / "outputs" / "evaluations" / "scored.json"
    assert received["kwargs"] == {
        "allow_partial": True,
        "project": project,
        "benchmark_scope": "full",
    }


def test_config_report_uses_library_defaults_without_secrets(project):
    (project / "llm.toml").write_text(
        "\n".join(
            [
                "schema_version = 1",
                "[providers.primary]",
                'base_url = "http://example.test/v1"',
                'api_key = "secret-value"',
                "[models.default]",
                'provider = "primary"',
                'model = "demo-model"',
                "context_window = 1000",
                "max_output_tokens = 100",
            ]
        )
    )
    (project / "config.toml").write_text('[agent]\nplanning_mode = "data_only"\n')
    result = CliRunner().invoke(app, ["config", "validate", "--project", str(project)])
    assert result.exit_code == 0, result.output
    assert "secret-value" not in result.output
    report = json.loads(result.output)
    assert report["models"] == ["default"]
    assert report["llm"]["models"][0]["model"] == "demo-model"
    assert report["agent"]["planning_mode"] == "data_only"
    assert report["agent"]["command_attempts"] == 20
    assert report["agent"]["max_total_tokens"] == 262144
    assert report["runtime"]["max_concurrent_model_calls"] == 16
    assert report["runtime"]["max_concurrent_tool_calls"] == 64
    assert report["runtime"]["max_concurrent_sandbox_calls"] == 16
    assert report["agent"]["plan_attempts"] == 3
    assert report["runtime"]["max_concurrent_tasks"] == 2
    assert report["runtime"]["tool_backend"] == "thread"
    assert report["runtime"]["sandbox"] == "auto"


def test_dataset_and_tool_commands_follow_registries(project):
    listed = CliRunner().invoke(app, ["dataset", "list"])
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.output)[0]["name"] == "geotooleval"
    path = CliRunner().invoke(app, ["dataset", "path", "--project", str(project)])
    assert path.exit_code == 0, path.output
    assert path.output.strip().endswith("geotooleval.jsonl")
    tools = json.loads(CliRunner().invoke(app, ["tools", "list"]).output)
    ids = {row["id"] for row in tools}
    assert "DeepEarthQueryUSGSCatalog" in ids
    assert "ExecutePython" not in ids
    internal = json.loads(CliRunner().invoke(app, ["tools", "list", "--internal"]).output)
    internal_ids = {row["id"] for row in internal}
    assert {"bash", "ExecutePython"} <= internal_ids
    shown = json.loads(CliRunner().invoke(app, ["tools", "show", "ExecutePython"]).output)
    assert shown["id"] == "ExecutePython"
