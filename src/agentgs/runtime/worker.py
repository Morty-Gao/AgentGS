import argparse
import asyncio
import json
from pathlib import Path

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.planning.skills import builtin_planner
from agentgs.agent.retrieval.skills import KeywordSkillRetriever
from agentgs.agent.retrieval.tools import KGToolRetriever, LazyToolRetriever
from agentgs.agent.state import AgentExecutionError
from agentgs.client.config import load_llm
from agentgs.client.embedding import LocalEmbedding
from agentgs.client.openai_compatible import OpenAICompatibleClient
from agentgs.config.algorithm import load_algorithm
from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskSpec
from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.answer_contracts import AnswerContracts
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.sandbox.base import CodeExecutor
from agentgs.runtime.tools.context import ToolPaths
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
        allow_internal_operations=bool(spec.input.calls),
    )
    skills = builtin_skills()
    client = None
    try:
        if not spec.input.calls:
            client = OpenAICompatibleClient(load_llm(root), events=events)
        options = load_algorithm(root)
        paths = ToolPaths.create(root, artifacts.root)
        contracts = AnswerContracts(root)
        retriever = None
        from agentgs.datasets.layout import prepared_path

        index_folder = root / options.index_path
        if options.index_path.startswith("data/"):
            index_folder = prepared_path(root, options.index_path.removeprefix("data/"))
        if (
            not spec.input.calls
            and (index_folder / "index_store.json").exists()
        ):
            retriever = LazyToolRetriever(
                lambda: KGToolRetriever.from_directory(
                    index_folder,
                    LocalEmbedding(
                        root / options.model_cache,
                        options.embedding_model,
                        options.embedding_threads,
                        options.embedding_batch_size,
                    ),
                )
            )

        def checkpoint(result):
            temporary = work / "result.json.part"
            temporary.write_text(result.model_dump_json(), encoding="utf-8")
            temporary.replace(work / "result.json")

        context = AgentContext(
            tools,
            artifacts,
            events,
            skills,
            KeywordSkillRetriever(skills),
            client,
            tool_retriever=retriever,
            planner=builtin_planner(rs_local_contracts=options.rs_local_contracts),
            options=options,
            resource_resolver=paths.resolve,
            answer_contract_lookup=contracts.find,
            answer_system_prompt=contracts.system_prompt,
            checkpoint=checkpoint,
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
            str(exc)
            if isinstance(exc, (ValueError, FileNotFoundError, AgentExecutionError))
            else type(exc).__name__
        )
        (work / "error.json").write_text(
            json.dumps({"code": "execution_failed", "message": message}), encoding="utf-8"
        )
        if isinstance(exc, AgentExecutionError):
            (work / "result.json").write_text(exc.result.model_dump_json(), encoding="utf-8")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
