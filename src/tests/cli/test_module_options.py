"""CLI module overrides must survive task construction and benchmark dispatch."""

import json

import pytest
from typer.testing import CliRunner

from agentgs.cli.app import app
from agentgs.contracts.models import TaskRecord, TaskSpec, TaskStatus


@pytest.fixture
def captured(monkeypatch):
    from agentgs.cli import app as cli

    tasks = []

    class Service:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def submit(self, task):
            tasks.append(task)
            self.record = TaskRecord(spec=TaskSpec(input=task), status=TaskStatus.SUCCEEDED)
            return self.record

        async def wait(self, task_id):
            return self.record

    monkeypatch.setattr(cli, "TaskService", Service)
    return tasks


def test_repeatable_presets_and_json_file_overrides(project, captured):
    module_file = project / "modules.json"
    module_file.write_text('{"execution_recovery":false}')
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--project",
            str(project),
            "--planning-mode",
            "skill_guided",
            "statistics",
            "--ablation",
            "A2",
            "--ablation",
            "A4",
            "--modules",
            "@" + str(module_file),
        ],
    )
    assert result.exit_code == 0, result.output
    task = captured[0]
    assert task.ablations == ["A2", "A4"]
    assert task.modules.execution_recovery is False


def test_task_file_keeps_modules_unless_explicitly_overridden(project, captured):
    path = project / "request.json"
    path.write_text(
        json.dumps(
            {
                "question": "statistics",
                "planning_mode": "skill_guided",
                "ablations": ["A1"],
                "modules": {"directed_bridging": False},
            }
        )
    )
    result = CliRunner().invoke(app, ["run", "--project", str(project), "--task-file", str(path)])
    assert result.exit_code == 0, result.output
    assert captured[-1].ablations == ["A1"] and captured[-1].modules.directed_bridging is False
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--project",
            str(project),
            "--task-file",
            str(path),
            "--ablation",
            "A6",
            "--modules",
            '{"directed_bridging":true}',
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured[-1].ablations == ["A6"] and captured[-1].modules.directed_bridging is True


def test_benchmark_dispatches_modules(project, captured, monkeypatch):
    from agentgs.benchmark import runner

    calls = []

    async def run_benchmark(service, dataset, output, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(runner, "run_benchmark", run_benchmark)
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "benchmark",
            "--project",
            str(project),
            "--dataset",
            "tasks.jsonl",
            "--ablation",
            "BIPARTITE",
            "--modules",
            '{"predecessor_expansion":false}',
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["ablations"] == ["BIPARTITE"]
    assert calls[0]["modules"] == {"predecessor_expansion": False}


@pytest.mark.parametrize(
    "arguments", [["--ablation", "A7"], ["--modules", "[]"], ["--modules", '{"typo":false}']]
)
def test_invalid_module_arguments_fail_before_submission(project, captured, arguments):
    result = CliRunner().invoke(app, ["run", "--project", str(project), *arguments])
    assert result.exit_code != 0
    assert not captured
