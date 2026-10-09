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
            if kwargs["stage"] == "tool_planning":
                return ChatReply(
                    message={
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "plan-1",
                                "type": "function",
                                "function": {
                                    "name": "submit_tool_plan",
                                    "arguments": '{"steps":[{"id":"s1","tool":"DeepEarthQueryUSGSCatalog","purpose":"Compute requested result"}]}',
                                },
                            }
                        ],
                    }
                )
            if kwargs["stage"] == "tool_argument_generation":
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
            assert any(message["role"] == "tool" for message in messages)
            return ChatReply(
                message={"role": "assistant", "content": "IsCompleted:Completed\nFinalAnswer:2"}
            )

    class Retriever:
        async def retrieve(self, query):
            return [
                [
                    "DeepEarthQueryUSGSCatalog",
                    "has the functionality that",
                    "Query a local earthquake catalog",
                ]
            ]

        def get_tool_subgraph(self, *args, **kwargs):
            return []

        def get_tool_info(self, *args):
            return {}

    events = Events()
    artifacts = LocalArtifactStore(project / "artifacts")
    tools = LocalToolExecutor(DatasetRegistry(project), artifacts, events)
    skills = builtin_skills()
    client = Client()
    result = await AgentEngine().run(
        TaskInput(question="earthquake catalog"),
        AgentContext(
            tools,
            artifacts,
            events,
            skills,
            KeywordSkillRetriever(skills),
            client,
            tool_retriever=Retriever(),
        ),
    )
    assert result.tool_results[0].data["count"] == 2
    assert result.usage[0]["usage"] is None
    assert any(kind == "tool.started" for kind, _ in events.rows)
    schemas = tools.specs()
    from agentgs.contracts.catalog import load_tool_scope

    assert {s.id for s in schemas} == set(load_tool_scope()["tool_ids"])
    assert not tools.internal
