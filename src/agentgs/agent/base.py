from dataclasses import dataclass
from typing import Any, Callable, Mapping

from agentgs.agent.retrieval.skills import SkillRetriever
from agentgs.client.models import ModelClient
from agentgs.contracts.models import ArtifactStore, EventSink, ToolExecutor
from agentgs.contracts.resources import ResourceResolver
from agentgs.skills.base import Skill


@dataclass
class AgentContext:
    tools: ToolExecutor
    artifacts: ArtifactStore
    events: EventSink
    skills: Mapping[str, Skill]
    retriever: SkillRetriever
    client: ModelClient | None = None
    tool_retriever: Any = None
    planner: Any = None
    options: Any = None
    resource_resolver: ResourceResolver | None = None
    answer_contract_lookup: Callable | None = None
    answer_system_prompt: str = ""
    checkpoint: Callable | None = None
