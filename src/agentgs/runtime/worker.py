import argparse
import asyncio
import json
from pathlib import Path

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.retrieval.skills import KeywordSkillRetriever
from agentgs.client.config import load_llm
from agentgs.client.openai_compatible import OpenAICompatibleClient
from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskSpec
from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.sandbox.base import CodeExecutor
from agentgs.runtime.tools.executor import LocalToolExecutor
from agentgs.skills.registry import builtin_skills


class PipeEvents:
    def emit(self, kind, payload):
        print(json.dumps({"type": kind, "payload": payload}, ensure_ascii=False), flush=True)


async def execute(root, work, spec, settings):
    events = PipeEvents()
    artifacts = LocalArtifactStore(work / "artifacts")
    tools = LocalToolExecutor(
        DatasetRegistry(root),
        artifacts,
        events,
        CodeExecutor(root, work / "code", settings),
    )
    skills = builtin_skills()
    client = None
    try:
        if not spec.input.calls:
            client = OpenAICompatibleClient(load_llm(root), events=events)
        context = AgentContext(
            tools, artifacts, events, skills, KeywordSkillRetriever(skills), client
        )
        return await AgentEngine().run(spec.input, context)
    finally:
        if client:
            await client.aclose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    spec = TaskSpec.model_validate(request["spec"])
    root = Path(request["root"])
    work = Path(args.request).parent
    try:
        result = asyncio.run(
            execute(root, work, spec, RuntimeSettings.model_validate(request["runtime"]))
        )
        (work / "result.json").write_text(result.model_dump_json(), encoding="utf-8")
    except Exception as exc:
        # Provider credentials/bodies are never included in diagnostics.
        message = (
            str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else type(exc).__name__
        )
        (work / "error.json").write_text(
            json.dumps({"code": "execution_failed", "message": message}), encoding="utf-8"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
