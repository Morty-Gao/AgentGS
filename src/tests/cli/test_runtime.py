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
