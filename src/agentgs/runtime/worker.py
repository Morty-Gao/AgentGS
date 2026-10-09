"""Compose async runtime services and run one isolated Agent task."""

import argparse
import asyncio
import json
import signal
from pathlib import Path

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.retrieval.tools import KGToolRetriever, LazyToolRetriever
from agentgs.agent.state import AgentExecutionError
from agentgs.client.config import load_llm
from agentgs.client.embedding import LocalEmbedding
from agentgs.client.openai_compatible import OpenAICompatibleClient
from agentgs.config.algorithm import load_algorithm, resolve_algorithm
from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import TaskSpec
from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.answer_contracts import AnswerContracts
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.concurrency import RuntimeCoordinator
from agentgs.runtime.models import ModelExecutor
from agentgs.runtime.sandbox.base import CodeExecutor
from agentgs.runtime.tools.context import ToolPaths
from agentgs.runtime.tools.executor import LocalToolExecutor


class PipeEvents:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def emit(self, kind, payload):
        async with self._lock:
            await invoke(
                lambda: print(
                    json.dumps({"type": kind, "payload": payload}, ensure_ascii=False), flush=True
                )
            )


async def execute(root, work, spec, settings):
    events = PipeEvents()
    coordinator = RuntimeCoordinator(root)
    artifacts = await invoke(LocalArtifactStore, work / "artifacts")
    tools = await invoke(
        LocalToolExecutor,
        DatasetRegistry(root),
        artifacts,
        events,
        CodeExecutor(root, work / "code", settings),
        allow_internal_operations=bool(spec.input.calls),
        settings=settings,
        coordinator=coordinator,
    )
    options, contracts, paths = await asyncio.gather(
        invoke(load_algorithm, root),
        invoke(AnswerContracts, root),
        invoke(ToolPaths.create, root, artifacts.root),
    )
    options = resolve_algorithm(options, spec.input)
    client = None
    try:
        if not spec.input.calls:
            config = await invoke(load_llm, root)
            client = ModelExecutor(
                OpenAICompatibleClient(config, events=events), settings, coordinator
            )
        from agentgs.datasets.layout import prepared_path

        index_folder = root / options.index_path
        if options.index_path.startswith("data/"):
            index_folder = prepared_path(root, options.index_path.removeprefix("data/"))
        retriever = None
        if not spec.input.calls and await invoke((index_folder / "index_store.json").exists):
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

        async def checkpoint(result):
            def write():
                temporary = work / "result.json.part"
                temporary.write_text(result.model_dump_json(), encoding="utf-8")
                temporary.replace(work / "result.json")

            await invoke(write)

        skills, skill_retriever, planner = {}, None, None
        if not spec.input.calls and options.planning_mode == "skill_guided":
            from agentgs.agent.planning.skills import builtin_planner
            from agentgs.agent.retrieval.skills import KeywordSkillRetriever
            from agentgs.skills.registry import builtin_skills

            skills, planner = await asyncio.gather(
                invoke(builtin_skills),
                invoke(builtin_planner, rs_local_contracts=options.rs_local_contracts),
            )
            skill_retriever = KeywordSkillRetriever(skills)
        context = AgentContext(
            tools,
            artifacts,
            events,
            skills,
            skill_retriever,
            client,
            tool_retriever=retriever,
            planner=planner,
            options=options,
            resource_resolver=paths.resolve,
            answer_contract_lookup=contracts.find,
            answer_system_prompt=contracts.system_prompt,
            checkpoint=checkpoint,
        )
        return await AgentEngine().run(spec.input, context)
    finally:
        await tools.aclose()
        if client:
            await client.aclose()


async def run_request(request_path):
    request = await invoke(lambda: json.loads(request_path.read_text()))
    spec = TaskSpec.model_validate(request["spec"])
    work, root = request_path.parent, Path(request["root"])
    task = asyncio.create_task(
        execute(root, work, spec, RuntimeSettings.model_validate(request["runtime"]))
    )
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    except (NotImplementedError, RuntimeError):
        pass
    try:
        result = await task
        await invoke((work / "result.json").write_text, result.model_dump_json(), encoding="utf-8")
    except asyncio.CancelledError:
        await invoke(
            (work / "error.json").write_text,
            json.dumps({"code": "cancelled", "message": "Task cancelled"}),
            encoding="utf-8",
        )
        raise
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, (ValueError, FileNotFoundError, AgentExecutionError))
            else type(exc).__name__
        )
        await invoke(
            (work / "error.json").write_text,
            json.dumps({"code": "execution_failed", "message": message}),
            encoding="utf-8",
        )
        if isinstance(exc, AgentExecutionError):
            await invoke(
                (work / "result.json").write_text, exc.result.model_dump_json(), encoding="utf-8"
            )
        raise
    finally:
        try:
            loop.remove_signal_handler(signal.SIGTERM)
        except (NotImplementedError, RuntimeError):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    try:
        asyncio.run(run_request(Path(args.request)))
    except (Exception, asyncio.CancelledError):
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
