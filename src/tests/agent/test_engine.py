from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.retrieval.skills import KeywordSkillRetriever
from agentgs.client.models import ChatReply
from agentgs.contracts.models import TaskInput
from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.tools.executor import LocalToolExecutor
from agentgs.skills.registry import builtin_skills


class Events:
    def __init__(self):
        self.rows = []

    def emit(self, kind, payload):
        self.rows.append((kind, payload))


async def test_llm_tool_loop_without_live_provider(project):
    class Client:
        def __init__(self):
            self.requests = []

        async def chat(self, messages, **kwargs):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                return ChatReply(
                    message={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "DeepEarthQueryUSGSCatalog",
                                    "arguments": '{"min_magnitude":8}',
                                },
                            }
                        ],
                    }
                )
            assert messages[-1]["role"] == "tool"
            return ChatReply(message={"role": "assistant", "content": "Catalog queried."})

    events = Events()
    artifacts = LocalArtifactStore(project / "artifacts")
    tools = LocalToolExecutor(DatasetRegistry(project), artifacts, events)
    skills = builtin_skills()
    client = Client()
    result = await AgentEngine().run(
        TaskInput(question="earthquake catalog", algorithm="prototype"),
        AgentContext(tools, artifacts, events, skills, KeywordSkillRetriever(skills), client),
    )
    assert result.tool_results[0].data["count"] == 2
    assert result.usage[0]["usage"] is None
    assert any(kind == "skill.selected" for kind, _ in events.rows)
    schemas = tools.specs()
    from agentgs.skills.definitions import load_tool_scope

    assert {s.id for s in schemas} == set(load_tool_scope()["tool_ids"])
    assert not tools.internal
