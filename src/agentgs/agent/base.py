from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Mapping

from agentgs.client.models import ModelClient
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.contracts.models import AgentResult, ArtifactStore, EventSink, ToolExecutor
from agentgs.contracts.resources import ResourceResolver
from agentgs.contracts.retrieval import ToolRetriever

if TYPE_CHECKING:
    from agentgs.agent.planning.skills import SkillPlanner
    from agentgs.agent.retrieval.skills import SkillRetriever
    from agentgs.skills.base import Skill


@dataclass
class AgentContext:
    tools: ToolExecutor
    artifacts: ArtifactStore
    events: EventSink
    skills: Mapping[str, Skill] = field(default_factory=dict)
    retriever: SkillRetriever | None = None
    client: ModelClient | None = None
    tool_retriever: ToolRetriever | None = None
    planner: SkillPlanner | None = None
    options: AlgorithmSettings | None = None
    resource_resolver: ResourceResolver | None = None
    answer_contract_lookup: Callable | None = None
    answer_system_prompt: str = ""
    checkpoint: Callable[[AgentResult], Awaitable[None] | None] | None = None
