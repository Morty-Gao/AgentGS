from dataclasses import dataclass
from typing import Mapping

from agentgs.agent.retrieval.skills import SkillRetriever
from agentgs.client.models import ModelClient
from agentgs.contracts.models import ArtifactStore, EventSink, ToolExecutor
from agentgs.skills.base import Skill


@dataclass
class AgentContext:
    tools: ToolExecutor
    artifacts: ArtifactStore
    events: EventSink
    skills: Mapping[str, Skill]
    retriever: SkillRetriever
    client: ModelClient | None = None
