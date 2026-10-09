"""Run a dataset on the agent and store each attempt. Scoring is delegated to evaluation."""

import asyncio
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from agentgs.benchmark.inputs import public_input
from agentgs.benchmark.io import write_json
from agentgs.benchmark.selection import id_from_dirname, resolve_dataset, task_dirname
from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import TaskInput
from agentgs.evaluation.analysis.conversation import as_conversation, conversation_usage
from agentgs.evaluation.datasets.base import Dataset
from agentgs.evaluation.metrics.report import score_prediction, summarize
from agentgs.evaluation.metrics.tools import SCORING_PROTOCOL


def _benchmark_identity(
    service,
    dataset,
    selected,
    planning_mode,
    allow_rule_planning,
    plan_attempts,
    tool_attempts,
    command_attempts,
    contract_provider,
    modules=None,
    ablations=(),
):
    input_hash = hashlib.sha256(dataset.source.read_bytes()).hexdigest()
    identity = {
        "schema_version": 5,
        "scoring_protocol": SCORING_PROTOCOL,
        "message_protocol": "openai_tool_calls",
        "dataset_sha256": input_hash,
        "planning_mode": planning_mode,
        "allow_rule_planning": allow_rule_planning,
        "algorithm": "geotoolagent-native-function-calling-v1",
        "protocol": "actual_agent_execution",
        "conversation_schema": 2,
        "effective_runtime": service.settings.model_dump(mode="json"),
        "selected_question_ids": [task.question_id for task in selected],
    }
    code_files = sorted((service.root / "src/agentgs").rglob("*.py"))
    identity["code_sha256"] = hashlib.sha256(
        b"".join(path.read_bytes() for path in code_files)
    ).hexdigest()
    identity["public_inputs_sha256"] = hashlib.sha256(
        json.dumps(
            [public_input(task).model_dump(mode="json") for task in selected],
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    for name in ["config.toml", "llm.toml"]:
        path = service.root / name
        identity[name + "_sha256"] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        )
    from agentgs.client.config import load_llm

    configuration = load_llm(service.root)
    identity["effective_models"] = {
        name: model.model_dump(mode="json") for name, model in configuration.models.items()
    }
    from agentgs.config.algorithm import load_algorithm, resolve_algorithm

    identity["effective_algorithm"] = resolve_algorithm(
        load_algorithm(service.root),
        TaskInput(
            modules=modules,
            ablations=list(ablations),
            planning_mode=planning_mode,
            allow_rule_planning=allow_rule_planning,
            plan_attempts=plan_attempts,
            tool_attempts=tool_attempts,
            command_attempts=command_attempts,
        ),
    ).model_dump(mode="json")
    from importlib.resources import files

    resources = {
        "tool_catalog": files("agentgs.contracts").joinpath("catalog.json"),
        "tool_scope": files("agentgs.contracts").joinpath("tool_scope.json"),
    }
    if planning_mode == "skill_guided":
        declarations = files("agentgs.skills.resources")
        resources.update(
            skill_definitions=declarations.joinpath("definitions.json"),
            capability_extensions=declarations.joinpath("capability_extensions.json"),
        )
    identity["planning_resources_sha256"] = {
        name: hashlib.sha256(resource.read_bytes()).hexdigest()
        for name, resource in resources.items()
    }
    if planning_mode == "data_only":
        from agentgs.runtime.sandbox.mounts import mount_manifest

        identity["data_mounts_sha256"] = hashlib.sha256(
            json.dumps(mount_manifest(service.root / "data"), sort_keys=True).encode()
        ).hexdigest()
        identity["tool_match"] = "not_applicable"
        identity["primary_metric"] = "fas"
    if contract_provider is not None:
        identity["answer_contracts_sha256"] = hashlib.sha256(
            json.dumps(
                {
                    "contracts": {
                        task.question_id: contract_provider.find(task.question) for task in selected
                    },
                    "system_prompt": getattr(contract_provider, "system_prompt", ""),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    identity["providers"] = {
        name: {
            "protocol": provider.protocol,
            "base_url_sha256": hashlib.sha256(provider.base_url.encode()).hexdigest(),
            "timeout_seconds": provider.timeout_seconds,
            "max_attempts": provider.max_attempts,
            "output_token_parameter": provider.output_token_parameter,
            "max_concurrent_requests": provider.max_concurrent_requests,
            "requests_per_minute": provider.requests_per_minute,
            "tokens_per_minute": provider.tokens_per_minute,
        }
        for name, provider in configuration.providers.items()
    }
    return (identity, configuration)


def _task_folder(output: Path, question_id) -> Path:
    return output / "tasks" / task_dirname(question_id)


async def run_benchmark(
    service,
    dataset: Dataset | Path,
    output: Path,
    *,
    ids=None,
    retry_failed=False,
    planning_mode="skill_guided",
    allow_rule_planning=False,
    plan_attempts=None,
    tool_attempts=None,
    command_attempts=None,
    contract_provider=None,
    benchmark_scope=None,
    modules=None,
    ablations=(),
):
    if planning_mode == "data_only":
        allow_rule_planning = False
    source = resolve_dataset(dataset)
    tasks = await invoke(source.load)
    selected = [task for task in tasks if ids is None or task.question_id in ids]
    if ids is not None and (missing := set(ids) - {task.question_id for task in tasks}):
        raise ValueError(f"Unknown benchmark question IDs: {sorted(missing)}")
    from agentgs.config.algorithm import load_algorithm
    from agentgs.benchmark.scope import select_scope

    selected, access_policy = select_scope(
        selected, benchmark_scope or load_algorithm(service.root).benchmark_scope, planning_mode=planning_mode
    )
    if not selected:
        raise ValueError("The declared benchmark scope contains no selected tasks")
    identity, configuration = await invoke(
        _benchmark_identity,
        service,
        source,
        selected,
        planning_mode,
        allow_rule_planning,
        plan_attempts,
        tool_attempts,
        command_attempts,
        contract_provider,
        modules,
        ablations,
    )
    await invoke(output.mkdir, parents=True, exist_ok=True)
    identity["data_access_policy"] = access_policy
    manifest = output / "protocol.json"
    if (
        await invoke(manifest.exists)
        and await invoke(lambda: json.loads(manifest.read_text())) != identity
    ):
        raise ValueError(
            "Code, model configuration, inputs or protocol differ; use a new run directory"
        )
    for task in selected:
        current = _task_folder(output, task.question_id) / "result.json"
        if await invoke(current.exists):
            previous = await invoke(lambda: json.loads(current.read_text(encoding="utf-8")))
            if previous.get("scores", {}).get("protocol") != SCORING_PROTOCOL:
                raise ValueError(
                    "Saved scores use a different scoring protocol; use a new run directory"
                )
    await invoke(write_json, manifest, identity)
    lock = output / ".run.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    started = time.monotonic()
    stopped_for_auth = False
    peak_running = 0
    completed = {}
    pending = []
    monitor = None

    def _log_event(kind, **fields):
        event = {"time": datetime.now(timezone.utc).isoformat(), "type": kind, **fields}
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        with (output / "experiment.events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    log_lock = asyncio.Lock()

    async def log_event(kind, **fields):
        async with log_lock:
            await invoke(_log_event, kind, **fields)

    def progress():
        nonlocal peak_running
        running = len(service.processes)
        peak_running = max(peak_running, running)
        return {
            "selected": len(selected),
            "completed": len(completed),
            "succeeded": sum(row["status"] == "succeeded" for row in completed.values()),
            "failed": sum(row["status"] != "succeeded" for row in completed.values()),
            "running_workers": running,
            "peak_running_workers": peak_running,
            "max_concurrent_tasks": service.settings.max_concurrent_tasks,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stopped_for_auth": stopped_for_auth,
        }

    async def heartbeat():
        while True:
            status = progress()
            await invoke(write_json, output / "progress.json", status)
            await log_event("benchmark.progress", **status)
            await asyncio.sleep(10)

    try:
        for task in selected:
            folder = _task_folder(output, task.question_id)
            current = folder / "result.json"
            if await invoke(current.exists):
                previous = await invoke(lambda: json.loads(current.read_text(encoding="utf-8")))
                if not retry_failed or previous["status"] == "succeeded":
                    completed[task.question_id] = previous
                    continue
            pending.append(task)

        queue = asyncio.Queue()
        for task in pending:
            queue.put_nowait(task)

        async def execute_task(task):
            nonlocal stopped_for_auth
            folder = _task_folder(output, task.question_id)
            current = folder / "result.json"
            await invoke(folder.mkdir, parents=True, exist_ok=True)
            number = await invoke(lambda: len(list(folder.glob("attempt_*"))) + 1)
            attempt = folder / f"attempt_{number:03d}"
            await invoke(attempt.mkdir, exist_ok=False)
            request = public_input(
                task,
                modules=modules,
                ablations=ablations,
                planning_mode=planning_mode,
                allow_rule_planning=allow_rule_planning,
                plan_attempts=plan_attempts,
                tool_attempts=tool_attempts,
                command_attempts=command_attempts,
                output_contract=contract_provider.find(task.question)
                if contract_provider
                else None,
                dataset_instructions=getattr(contract_provider, "system_prompt", None),
            )
            await invoke(write_json, attempt / "public_input.json", request.model_dump(mode="json"))
            record = await service.submit(request)
            await invoke(
                write_json,
                attempt / "submission.json",
                {"question_id": task.question_id, "task_id": record.spec.id, "attempt": number},
            )
            await log_event(
                "benchmark.task_submitted",
                question_id=task.question_id,
                category=task.category,
                task_id=record.spec.id,
                attempt=number,
            )
            result = await service.wait(record.spec.id)
            await invoke(write_json, attempt / "task_record.json", result.model_dump(mode="json"))
            events = await service.events(record.spec.id)
            await invoke(
                lambda: (attempt / "events.jsonl").write_text(
                    "".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8"
                )
            )
            work = service.root / "outputs/tasks" / record.spec.id
            if await invoke((work / "stderr.log").exists):
                await invoke(shutil.copy2, work / "stderr.log", attempt / "stderr.log")
            answer = result.result.answer if result.result else ""
            conversation = as_conversation(result.result.messages if result.result else [])
            await invoke(conversation.save, attempt / "conversation.json")
            await invoke(
                (attempt / "messages.jsonl").write_text, conversation.to_jsonl(), encoding="utf-8"
            )
            tokens = conversation_usage(conversation, result.result.usage if result.result else [])
            scored = await invoke(score_prediction, task, answer, conversation, tokens)
            row = {
                **scored,
                "status": result.status.value,
                "task_id": record.spec.id,
                "final_answer": answer,
                "attempt": number,
                "error": result.error.model_dump() if result.error else None,
                "messages": conversation.model_dump(mode="json")["messages"],
                "completion_status": result.result.completion_status if result.result else None,
                "tool_chain_completed": result.result.tool_chain_completed
                if result.result
                else None,
                "answer_generation_status": result.result.answer_generation_status
                if result.result
                else None,
                "token_usage": tokens,
            }
            await invoke(write_json, attempt / "score.json", row)
            await invoke(write_json, current, row)
            completed[task.question_id] = row
            await log_event(
                "benchmark.task_finished",
                question_id=task.question_id,
                task_id=record.spec.id,
                status=result.status.value,
                fas=scored["scores"]["fas"],
                error=row["error"],
                completed=len(completed),
                token_usage=tokens,
            )
            await invoke(write_json, output / "progress.json", progress())
            # Authentication failures cannot be repaired by rerunning more tasks.
            if result.error and (
                "HTTP 401" in result.error.message or "HTTP 403" in result.error.message
            ):
                stopped_for_auth = True

        async def worker():
            while not stopped_for_auth:
                try:
                    task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await execute_task(task)
                finally:
                    queue.task_done()

        await log_event(
            "benchmark.started",
            selected=len(selected),
            pending=len(pending),
            resumed=len(completed),
            max_concurrent_tasks=service.settings.max_concurrent_tasks,
            model={name: model.model for name, model in configuration.models.items()},
        )
        monitor = asyncio.create_task(heartbeat())
        workers = [
            asyncio.create_task(worker())
            for _ in range(min(len(pending), service.settings.max_concurrent_tasks))
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            for job in workers:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        await log_event("benchmark.finished", **progress())
    finally:
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)

        def export():
            selected_ids = {task.question_id for task in selected}
            rows = [
                json.loads(path.read_text())
                for path in sorted((output / "tasks").glob("*/result.json"))
                if id_from_dirname(path.parent.name, selected_ids) in selected_ids
            ]
            report = summarize(rows, len(tasks), domain_count=source.domain_count)
            report["execution"] = progress()
            if planning_mode == "data_only":
                report["primary_metric"] = "fas"
                report["tool_match"] = "not_applicable"
            write_json(output / "summary.json", report)
            with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(
                        json.dumps(
                            {
                                "question_id": row["question_id"],
                                "final_answer": row["final_answer"],
                                "status": row["status"],
                                "messages": row.get("messages", []),
                                "token_usage": row.get("token_usage"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            return report

        try:
            report = await invoke(export)
        finally:
            await invoke(lock.unlink, missing_ok=True)
    return report
