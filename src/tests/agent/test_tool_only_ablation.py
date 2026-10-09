"""Behavioral controls for a shared runtime with no Skill/Capability planning."""

import json
import re
import subprocess
import sys
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.execution.service import ExecutionService
from agentgs.agent.planning.model_stages import ModelStages
from agentgs.agent.state import RunState
from agentgs.client.models import ChatReply
from agentgs.config.algorithm import AlgorithmSettings, resolve_algorithm
from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskInput, TaskSpec, ToolError, ToolResult, ToolSpec


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


class KG:
    async def retrieve(self, query):
        return [["RemoteSensingMean", "has the functionality that", "Calculate a scalar mean"]]

    def get_tool_subgraph(self, name, **kwargs):
        return [
            [name, "has the functionality that", "Calculate a scalar mean"],
            [name, "inputs", "x: numeric array"],
            [name, "outputs", "value: scalar"],
        ]

    def get_tool_info(self, relations):
        return {"input_format": "x: numeric array", "output_format": "value: scalar"}


class Client:
    def __init__(self):
        self.requests = []

    async def chat(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        if kwargs["stage"] in {"tool_planning", "skill_tool_planning"}:
            reply = native("submit_tool_plan", {"tools": ["RemoteSensingMean"]})
        elif kwargs["stage"] == "tool_argument_generation":
            reply = native("RemoteSensingMean", {"x": [1, 3]})
        else:
            reply = {"role": "assistant", "content": "IsCompleted:Completed\nFinalAnswer:2"}
        return ChatReply(message=reply)

    async def aclose(self):
        pass


class Events:
    def emit(self, *args):
        pass


@pytest.mark.parametrize("mode", ["tool_only", "skill_guided"])
async def test_worker_resolves_mode_before_loading_declarations(project, monkeypatch, mode):
    from agentgs.runtime import worker
    from agentgs.skills import definitions

    # The opposite project default must not determine task-specific initialization.
    configured = "skill_guided" if mode == "tool_only" else "tool_only"
    (project / "config.toml").write_text(f'[agent]\nplanning_mode="{configured}"\n')
    folder = project / "data/geotoolkg"
    folder.mkdir(parents=True)
    (folder / "index_store.json").write_text("{}")
    loads = []
    if mode == "tool_only":

        def forbidden_resources(*args, **kwargs):
            raise AssertionError("The ablation cannot read Skill resource files")

        monkeypatch.setattr(definitions, "files", forbidden_resources)
    for name in ("load_skill_definitions", "load_capability_extensions", "load_library_scope"):
        original = getattr(definitions, name)

        def load(original=original, name=name):
            loads.append(name)
            if mode == "tool_only":
                raise AssertionError("The ablation cannot load scientific declarations")
            return original()

        monkeypatch.setattr(definitions, name, load)
    client = Client()
    monkeypatch.setattr(worker, "load_llm", lambda _: None)
    monkeypatch.setattr(worker, "OpenAICompatibleClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(worker, "LazyToolRetriever", lambda _: KG())
    result = await worker.execute(
        project,
        project / "work",
        TaskSpec(
            input=TaskInput(
                question="Compute the requested statistic.",
                planning_mode=mode,
            )
        ),
        RuntimeSettings(tool_backend="thread"),
    )
    assert result.answer == "2" and result.tool_chain_completed
    assert result.provenance["effective_algorithm"]["planning_mode"] == mode
    stages = [kwargs["stage"] for _, kwargs in client.requests]
    assert stages == [
        "tool_planning" if mode == "tool_only" else "skill_tool_planning",
        "tool_argument_generation",
        "final_answer_generation",
    ]
    events = {event["type"] for event in result.execution_events()}
    if mode == "tool_only":
        assert not loads
        assert result.provenance["effective_algorithm"]["shared_bridge"] is False
        assert not events & {
            "skill_retrieval",
            "skill_plan",
            "capability_expansion",
            "geo_constraint_check",
            "tool_grounding",
            "tool_graph_retrieval",
            "shared_capability_bridge_search",
        }
        for messages, _ in client.requests:
            for message in messages:
                if message["role"] == "developer":
                    assert not re.search(
                        r"\b(?:skill|capabilit(?:y|ies))\b", message["content"], re.I
                    )
    else:
        assert {"load_skill_definitions", "load_capability_extensions"} <= set(loads)
        assert {"skill_plan", "capability_expansion", "tool_grounding"} <= events


def test_worker_import_has_no_scientific_planning_dependency():
    script = """
import sys
import agentgs.runtime.worker
assert not any(name.startswith((
    'agentgs.skills', 'agentgs.agent.planning.skills', 'agentgs.agent.planning.capabilities',
    'agentgs.agent.planning.bridge', 'agentgs.agent.retrieval.scored_skills',
)) for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("mode", ["tool_only", "skill_guided"])
@pytest.mark.parametrize("budget", [1, 2])
async def test_model_planning_recovery_budget_is_independent_of_rules(mode, budget):
    class Tools:
        def __init__(self):
            self.calls = []

        def specs(self):
            return [
                ToolSpec(
                    id="RemoteSensingMean", description="Mean", input_schema={"type": "object"}
                )
            ]

        async def execute(self, call):
            self.calls.append(call)
            if len(self.calls) == 1:
                return ToolResult(
                    call_id=call.id,
                    tool=call.tool,
                    succeeded=False,
                    error=ToolError(code="domain", message="Try another input"),
                )
            return ToolResult(call_id=call.id, tool=call.tool, succeeded=True, data={"value": 2})

    tools, client = Tools(), Client()
    context = AgentContext(
        tools,
        None,
        Events(),
        client=client,
        tool_retriever=KG(),
        options=AlgorithmSettings(
            planning_mode=mode, allow_rule_planning=False, plan_attempts=budget, tool_attempts=1
        ),
    )
    result = await AgentEngine().run(
        TaskInput(question="Compute the requested statistic."), context
    )
    assert len(tools.calls) == budget
    assert result.tool_chain_completed == (budget == 2)
    planning = [kwargs for _, kwargs in client.requests if kwargs["stage"].endswith("planning")]
    assert len(planning) == budget
    assert not {"schema_route", "plan_model_fallback"} & {
        e["type"] for e in result.execution_events()
    }


async def test_tool_only_rejects_injected_skill_plan_or_bridge_context():
    class Tools:
        def specs(self):
            return [
                ToolSpec(
                    id="RemoteSensingMean", description="Mean", input_schema={"type": "object"}
                )
            ]

    state = RunState(
        AgentContext(Tools(), None, Events(), options=AlgorithmSettings()),
        TaskInput(question="Compute a statistic."),
    )
    stages = ModelStages(state, lambda _: None, "")
    with pytest.raises(ValueError, match="cannot consume a Skill plan"):
        await stages.generate_tools_plan(state.task.question, [], skill_plan={"steps": []})
    with pytest.raises(ValueError, match="must match the resolved task policy"):
        await stages.generate_tools_plan(state.task.question, [], planning_mode="skill_guided")
    with pytest.raises(ValueError, match="require enabled skill-guided"):
        await ExecutionService(state, stages).run(
            "RemoteSensingMean",
            state.task.question,
            None,
            "",
            execution_context={"shared_bridge": {"injected": True}},
        )


def test_planning_policy_and_budgets_are_explicit_and_validated():
    options = AlgorithmSettings(planning_mode="skill_guided", plan_attempts=4, shared_bridge=True)
    task = TaskInput(planning_mode="tool_only", allow_rule_planning=True, tool_attempts=2)
    effective = resolve_algorithm(options, task)
    assert effective.plan_attempts == 4 and effective.tool_attempts == 2
    assert effective.allow_rule_planning and not effective.shared_bridge
    assert options.shared_bridge and options.planning_mode == "skill_guided"
    data_only = resolve_algorithm(
        options,
        TaskInput(planning_mode="data_only", allow_rule_planning=True, command_attempts=5),
    )
    assert data_only.planning_mode == "data_only"
    assert data_only.command_attempts == 5
    assert not data_only.allow_rule_planning and not data_only.shared_bridge
    for model in (AlgorithmSettings, TaskInput):
        for field in ("plan_attempts", "tool_attempts", "command_attempts"):
            with pytest.raises(ValidationError):
                model(**{field: 0})
        with pytest.raises(ValidationError):
            model(require_llm=True)
