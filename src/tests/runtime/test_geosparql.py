import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.retrieval.skills import KeywordSkillRetriever
from agentgs.client.models import ChatReply
from agentgs.contracts.models import TaskInput, TaskStatus, ToolCall
from agentgs.datasets.geoquestions import snapshot_profile
from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.service import TaskService
from agentgs.runtime.tools.domains.geosparql import SnapshotQuery, replay_query
from agentgs.runtime.tools.executor import LocalToolExecutor
from agentgs.skills.registry import builtin_skills

FIXTURES = Path(__file__).parents[1] / "fixtures/geoquestions"
ORACLE = json.loads((FIXTURES / "replay.json").read_text())


@pytest.fixture
def snapshot_project(project):
    target = project / "data/geoquestions/queries.jsonl"
    target.parent.mkdir(parents=True)
    target.write_bytes((FIXTURES / "snapshot.json").read_bytes())
    return project


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda c: c["id"])
def test_replay_matches_historical_source_decisions(snapshot_project, case):
    datasets = DatasetRegistry(snapshot_project)
    params = SnapshotQuery.model_validate(case["params"])
    if "expected_error" in case:
        with pytest.raises(ValueError, match="not found uniquely"):
            replay_query(params, datasets)
        return
    actual = replay_query(params, datasets)
    expected = case["expected"]
    for field in ["source_id", "query", "result", "mode"]:
        assert actual[field] == expected[field]
    # ID lookup is now accurately labelled, rather than called semantic equivalence.
    if params.source_id == expected["source_id"]:
        assert actual["query_match"] == "source_id"
    else:
        assert actual["query_match"] == expected["query_match"]
    assert actual["provenance"]["answer_lookup"] is True
    assert actual["provenance"]["snapshot_sha256"] == ORACLE["snapshot_sha256"]


def test_missing_or_changed_snapshot_fails(project, snapshot_project):
    target = snapshot_project / "data/geoquestions/queries.jsonl"
    target.write_text("[]")
    with pytest.raises(ValueError, match="hash differs"):
        replay_query(SnapshotQuery(source_id=591), DatasetRegistry(project))
    target.unlink()
    with pytest.raises(FileNotFoundError):
        replay_query(SnapshotQuery(source_id=591), DatasetRegistry(project))


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"query": "  "},
        {"source_id": 0},
        {"source_id": 591, "endpoint": "http://invalid"},
        {"source_id": 591, "gold_answer": "private"},
        {"source_id": 591, "gold_sparql": "ASK {}"},
    ],
)
def test_invalid_or_foreign_protocol_inputs_fail(params):
    with pytest.raises(ValidationError):
        SnapshotQuery.model_validate(params)


def test_oracle_identity_is_pinned():
    profile = snapshot_profile()
    assert ORACLE["source_sha256"] == profile["matching_source"]["sha256"]
    assert ORACLE["snapshot_sha256"] == profile["snapshot_sha256"]
    assert (
        hashlib.sha256((FIXTURES / "snapshot.json").read_bytes()).hexdigest()
        == profile["snapshot_sha256"]
    )
    assert (
        hashlib.sha256((FIXTURES / "replay.json").read_bytes()).hexdigest()
        == profile["replay_fixture_sha256"]
    )


async def test_worker_replays_without_endpoint_and_persists_artifacts(
    snapshot_project, monkeypatch
):
    monkeypatch.setenv("GEOSPARQL_ENDPOINT", "http://127.0.0.1:1/unavailable")
    service = TaskService(snapshot_project)
    try:
        task = await service.submit(
            TaskInput(
                calls=[ToolCall(tool="GeoBigDataExecuteGeoSPARQL", arguments={"source_id": 591})]
            )
        )
        record = await service.wait(task.spec.id)
        assert record.status == TaskStatus.SUCCEEDED
        result = record.result.tool_results[0]
        assert result.source_type == "derived_lookup"
        assert result.data["result"]["rows"] == [["5.835162072998658E-5"]]
        artifact = result.artifacts[0]
        assert (
            snapshot_project / "outputs/tasks" / task.spec.id / "artifacts" / artifact.path
        ).exists()
    finally:
        await service.close()


async def test_agent_selects_snapshot_skill_and_calls_query(snapshot_project):
    class Events:
        def __init__(self):
            self.selected = []

        def emit(self, kind, payload):
            if kind == "skill_plan":
                self.selected.extend(step["skill_id"] for step in payload["content"]["steps"])

    class Client:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert kwargs["tools"][0]["function"]["name"] == "GeoBigDataExecuteGeoSPARQL"
                return ChatReply(
                    message={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "replay",
                                "type": "function",
                                "function": {
                                    "name": "GeoBigDataExecuteGeoSPARQL",
                                    "arguments": json.dumps(
                                        {"query": ORACLE["cases"][1]["params"]["query"]}
                                    ),
                                },
                            }
                        ],
                    }
                )
            from html import unescape

            feedback = next(
                message["content"] for message in reversed(messages) if message["role"] == "tool"
            )
            tool_result = json.loads(
                unescape(feedback.split("<tool-result>\n", 1)[1].split("\n</tool-result>", 1)[0])
            )
            assert tool_result["query_match"] == "exact"
            return ChatReply(
                message={
                    "role": "assistant",
                    "content": "IsCompleted:Completed\nFinalAnswer:Frozen snapshot retrieved.",
                }
            )

    events = Events()
    artifacts = LocalArtifactStore(snapshot_project / "artifacts")
    tools = LocalToolExecutor(DatasetRegistry(snapshot_project), artifacts, events)
    skills = builtin_skills()
    from agentgs.config.algorithm import AlgorithmSettings

    result = await AgentEngine().run(
        TaskInput(
            question="GeoQuestions park area in Essex", input_resource="yago2geo_original_rdf"
        ),
        AgentContext(
            tools,
            artifacts,
            events,
            skills,
            KeywordSkillRetriever(skills),
            Client(),
            options=AlgorithmSettings(
                planning_mode="skill_guided", geosparql_llm=True, allow_rule_planning=True
            ),
        ),
    )
    assert events.selected == ["semantic-geospatial-query"]
    assert result.tool_results[0].data["source_id"] == 591
