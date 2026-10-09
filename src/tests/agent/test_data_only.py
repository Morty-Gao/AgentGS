"""data_only explores with bash and does not load Skill or tool retrieval."""

import json
from uuid import uuid4

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.client.models import ChatReply
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskInput, TaskSpec
from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.tools.executor import LocalToolExecutor


def native(name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": uuid4().hex,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class Events:
    def emit(self, *args):
        pass


class Shell:
    def __init__(self, root, work, settings):
        self.root, self.work, self.settings = root, work, settings
        self.commands = []

    async def describe_shell(self, data_root):
        return {
            "isolation": "bwrap",
            "data_mount": "bwrap_ro",
            "network_isolated": True,
            "mounts": ["/data/climate"],
        }

    async def run_shell(self, command, *, work, data_root, timeout, **kwargs):
        self.commands.append(command)
        return {
            "stdout": "42\n",
            "stderr": "",
            "exit_code": 0,
            "truncated": False,
            "isolation": "bwrap",
            "data_mount": "bwrap_ro",
            "network_isolated": True,
            "mounts": ["/data/climate"],
        }


class Client:
    def __init__(self, *, always_bash=False):
        self.requests = []
        self.always_bash = always_bash

    async def chat(self, messages, **kwargs):
        self.requests.append(kwargs)
        explorations = [row for row in self.requests if row["stage"] == "data_exploration"]
        if kwargs["stage"] == "data_exploration" and (self.always_bash or len(explorations) <= 2):
            command = "ls /data" if len(explorations) == 1 else "cat /data/climate/note.txt"
            return ChatReply(message=native("bash", {"command": command}))
        return ChatReply(
            message={"role": "assistant", "content": "IsCompleted:Completed\nFinalAnswer:42"}
        )

    async def aclose(self):
        pass


async def test_worker_data_only_skips_skills_and_tool_retrieval(project, monkeypatch):
    from agentgs.runtime import worker
    from agentgs.skills import definitions

    (project / "config.toml").write_text(
        '[agent]\nplanning_mode="skill_guided"\nshared_bridge=true\nallow_rule_planning=true\n'
    )
    index = project / "data/geotoolkg"
    index.mkdir(parents=True)
    (index / "index_store.json").write_text("{}")
    shells = []
    clients = []

    def fake_shell(root, work, settings, **kwargs):
        shell = Shell(root, work, settings)
        shells.append(shell)
        return shell

    def fake_client(*args, **kwargs):
        client = Client()
        clients.append(client)
        return client

    def forbidden(*args, **kwargs):
        raise AssertionError("data_only cannot load Skill resources")

    monkeypatch.setattr(definitions, "files", forbidden)
    monkeypatch.setattr(worker, "CodeExecutor", fake_shell)
    monkeypatch.setattr(worker, "load_llm", lambda _: None)
    monkeypatch.setattr("agentgs.runtime.tools.executor.builtin_tools", forbidden)
    monkeypatch.setattr(worker, "OpenAICompatibleClient", fake_client)
    monkeypatch.setattr(
        worker,
        "LazyToolRetriever",
        lambda *_: (_ for _ in ()).throw(AssertionError("no KG")),
    )
    result = await worker.execute(
        project,
        project / "work",
        TaskSpec(
            input=TaskInput(
                question="Compute the requested statistic.",
                planning_mode="data_only",
                allow_rule_planning=True,
            )
        ),
        RuntimeSettings(tool_backend="thread"),
    )
    assert result.answer == "42"
    assert result.tool_chain_completed
    assert result.provenance["data_mount"] == "bwrap_ro"
    assert result.provenance["data_paths"] == ["/data/climate"]
    effective = result.provenance["effective_algorithm"]
    assert effective["planning_mode"] == "data_only"
    assert effective["shared_bridge"] is False
    assert effective["allow_rule_planning"] is False
    assert shells[0].commands == ["ls /data", "cat /data/climate/note.txt"]
    assert [row["stage"] for row in clients[0].requests] == [
        "data_exploration",
        "data_exploration",
        "data_exploration",
    ]
    assert {"skill_retrieval", "skill_plan", "tool_retrieval", "tool_planning"}.isdisjoint(
        {event["type"] for event in result.execution_events()}
    )
    assert any(event["type"] == "data_sandbox" for event in result.execution_events())


async def test_command_budget_ends_exploration_before_another_model_turn(project):
    root = project
    shell = Shell(root, root / "code", RuntimeSettings())
    client = Client(always_bash=True)
    tools = LocalToolExecutor(
        DatasetRegistry(root), LocalArtifactStore(root / "artifacts"), Events(), shell
    )
    context = AgentContext(
        tools=tools,
        artifacts=tools.artifacts,
        events=Events(),
        client=client,
        options=AlgorithmSettings(
            planning_mode="data_only", allow_rule_planning=True, command_attempts=1
        ),
        code_executor=shell,
    )
    result = await AgentEngine().run(
        TaskInput(question="Compute the requested statistic."), context
    )
    assert shell.commands == ["ls /data"]
    assert [row["stage"] for row in client.requests] == [
        "data_exploration",
        "final_answer_generation",
    ]
    assert result.action_budget_exhausted
    assert result.tool_chain_completed is True
    assert result.provenance["effective_algorithm"]["allow_rule_planning"] is False


async def test_worker_data_only_uses_real_shell_and_records_failed_then_successful_commands(
    project, monkeypatch
):
    import sys

    import pytest

    from agentgs.runtime import worker
    from agentgs.runtime.sandbox.base import CodeExecutor, SandboxUnavailable

    if sys.platform != "linux":
        pytest.skip("Linux integration test")
    probe = CodeExecutor(project, project / "code", RuntimeSettings())
    try:
        await probe.describe_shell(project / "data")
    except SandboxUnavailable as exc:
        pytest.skip(str(exc))

    class RealShellClient:
        def __init__(self):
            self.commands = [
                "echo diagnostic >&2; exit 2",
                "python - <<'CODE'\nimport csv\nwith open('/data/deep_earth/usgs_source.csv') as f:\n    print(sum(float(row['mag']) for row in csv.DictReader(f)))\nCODE",
            ]

        async def chat(self, messages, **kwargs):
            if kwargs["stage"] == "data_exploration" and self.commands:
                return ChatReply(message=native("bash", {"command": self.commands.pop(0)}))
            return ChatReply(
                message={"role": "assistant", "content": "IsCompleted:Completed\nFinalAnswer:23.5"}
            )

        async def aclose(self):
            pass

    monkeypatch.setattr(worker, "load_llm", lambda _: None)
    monkeypatch.setattr(worker, "OpenAICompatibleClient", lambda *args, **kwargs: RealShellClient())
    result = await worker.execute(
        project,
        project / "work",
        TaskSpec(input=TaskInput(question="Compute the sum.", planning_mode="data_only")),
        RuntimeSettings(sandbox="bwrap"),
    )
    assert result.answer == "23.5" and result.tool_chain_completed
    assert [row.succeeded for row in result.tool_results] == [False, True]
    assert result.tool_results[0].data["exit_code"] == 2
    assert result.tool_results[1].data["stdout"] == "23.5\n"
    assert result.provenance["network_isolated"]
    assert all(row.artifacts for row in result.tool_results)
