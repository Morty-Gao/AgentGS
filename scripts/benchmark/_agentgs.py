"""Durable GeoToolEval batches for the project AgentGS agent.

Generation records the answer and the conversation. It does not calculate FAS.
Tool messages used for domain scores come from the final attempt. Model-usage
events from earlier attempts stay on the scoring conversation so cost includes
every try. A finished Not Completed result that still returned an answer is a
produced answer; timeouts and worker crashes are retried.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from _runtime import atomic_json, atomic_jsonl

from agentgs.benchmark.identity import model_request_controls, planning_resource_hashes
from agentgs.benchmark.inputs import public_input
from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import TaskInput
from agentgs.evaluation.analysis.conversation import as_conversation
from agentgs.messages import Conversation


def task_request(
    task,
    *,
    planning_mode,
    allow_rule_planning,
    plan_attempts,
    tool_attempts,
    command_attempts,
    modules,
    ablations,
    contract_provider,
    model=None,
):
    """Build the runtime request. Gold fields stay on the example."""
    return public_input(
        task,
        modules=modules,
        ablations=ablations or [],
        planning_mode=planning_mode,
        allow_rule_planning=allow_rule_planning,
        plan_attempts=plan_attempts,
        tool_attempts=tool_attempts,
        command_attempts=command_attempts,
        output_contract=contract_provider.find(task.question) if contract_provider else None,
        dataset_instructions=getattr(contract_provider, "system_prompt", None),
        model=model,
    )


def classify_result(result) -> dict:
    """Map one finished task onto a prediction. Scoring happens later."""
    answer = ""
    conversation = Conversation()
    completion_status = None
    tool_chain_completed = None
    answer_generation_status = None
    if result.result is not None:
        answer = result.result.answer or ""
        conversation = Conversation(messages=list(result.result.messages))
        completion_status = result.result.completion_status
        tool_chain_completed = result.result.tool_chain_completed
        answer_generation_status = result.result.answer_generation_status
    error = None
    code = result.error.code if result.error is not None else ""
    # Not Completed is the agent finishing without a validated answer. When it
    # still returned text, that text is the prediction; transport failures are not.
    if result.error is not None and not (code == "task_incomplete" and answer.strip()):
        error = f"{code}: {result.error.message}"
    elif not answer.strip():
        error = f"{result.status.value}: no answer"
    return {
        "answer": answer,
        "final_answer": answer,
        "status": "succeeded" if answer.strip() and error is None else "failed",
        "error": error,
        "conversation": conversation,
        "completion_status": completion_status,
        "tool_chain_completed": tool_chain_completed,
        "answer_generation_status": answer_generation_status,
        "task_status": result.status.value,
    }


def authentication_failure(error) -> bool:
    text = error or ""
    return "HTTP 401" in text or "HTTP 403" in text


def _model_event(message) -> bool:
    return message.role == "event" and str(message.metadata.get("type", "")).startswith("model.")


def scoring_conversation(attempts: list) -> Conversation:
    """Keep the final execution and append model events from every attempt."""
    conversations = [as_conversation(item) for item in attempts]
    final = conversations[-1].model_copy(deep=True)
    final.messages = [message for message in final.messages if not _model_event(message)]
    for source in conversations:
        for message in source.messages:
            if not _model_event(message):
                continue
            cloned = message.model_copy(deep=True)
            cloned.id = uuid4().hex
            final.append(cloned)
    return final


def _tree_hash(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    ordered = sorted(paths, key=lambda path: path.relative_to(root).as_posix())
    for path in ordered:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_digest(provider) -> str | None:
    if provider is None:
        return None
    encoded = json.dumps(
        {
            "contracts": provider.contracts,
            "system_prompt": getattr(provider, "system_prompt", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def prepared_inputs(root: Path) -> dict[str, list[int]]:
    """Size and mtime of prepared data. This is not a hash of every payload."""
    data = root / "data"
    if not data.is_dir():
        return {}
    found = {}
    for path in sorted(data.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(data)
        if "raw" in relative.parts:
            continue
        stat = path.stat()
        found[str(relative)] = [stat.st_size, stat.st_mtime_ns]
    return found


def build_identity(
    root: Path,
    dataset: Path,
    *,
    planning_mode: str,
    allow_rule_planning: bool,
    plan_attempts,
    tool_attempts,
    command_attempts,
    modules,
    ablations,
    answer_contracts: bool,
    contract_sha256: str | None,
    retries: int,
    runtime: dict,
    model_key: str,
    model_id: str,
    base_url: str,
    benchmark_scope: str,
) -> dict:
    """Identity for resume. Question subsets and --retry-failed are not part of it."""
    code = list((root / "src" / "agentgs").rglob("*.py"))
    code.extend(
        path
        for path in (
            root / "scripts/benchmark/_agentgs.py",
            root / "scripts/benchmark/agentgs/run.py",
        )
        if path.is_file()
    )
    sample = TaskInput(
        planning_mode=planning_mode,
        allow_rule_planning=False if planning_mode == "data_only" else allow_rule_planning,
        plan_attempts=plan_attempts,
        tool_attempts=tool_attempts,
        command_attempts=command_attempts,
        modules=modules,
        ablations=list(ablations or []),
    )
    from agentgs.config.algorithm import load_algorithm, resolve_algorithm

    effective = resolve_algorithm(load_algorithm(root), sample)
    return {
        "agent": "agentgs",
        "protocol": "agentgs-geotooleval-v1",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "code_sha256": _tree_hash(code, root),
        "config_sha256": _file_sha256(root / "config.toml"),
        "llm_sha256": _file_sha256(root / "llm.toml"),
        "model_key": model_key,
        "model": model_id,
        "base_url_sha256": hashlib.sha256(base_url.encode()).hexdigest(),
        "planning_mode": effective.planning_mode,
        "allow_rule_planning": effective.allow_rule_planning,
        "plan_attempts": plan_attempts,
        "tool_attempts": tool_attempts,
        "command_attempts": command_attempts,
        "modules": modules,
        "ablations": list(ablations or []),
        "answer_contracts": answer_contracts,
        "answer_contracts_sha256": contract_sha256,
        "benchmark_scope": benchmark_scope,
        "retries": retries,
        "runtime": runtime,
        "effective_algorithm": effective.model_dump(mode="json"),
        "planning_resources_sha256": planning_resource_hashes(root, effective.planning_mode),
        "model_request_controls": model_request_controls(),
        "prepared_inputs": prepared_inputs(root),
    }


def select_tasks(
    root: Path,
    dataset: Path | None,
    dataset_name: str | None,
    ids: str,
    planning_mode: str,
    benchmark_scope: str | None = None,
):
    from agentgs.benchmark.scope import select_scope
    from agentgs.benchmark.selection import resolve_dataset
    from agentgs.cli.parsing import parse_ids, select_dataset
    from agentgs.config.algorithm import load_algorithm

    source = resolve_dataset(select_dataset(root, dataset, dataset_name))
    tasks = source.load()
    chosen = parse_ids(ids)
    if chosen:
        missing = set(chosen) - {task.question_id for task in tasks}
        if missing:
            raise ValueError(f"Unknown benchmark question IDs: {sorted(missing, key=str)}")
        tasks = [task for task in tasks if task.question_id in chosen]
    scope = benchmark_scope or load_algorithm(root).benchmark_scope
    selected, _policy = select_scope(tasks, scope, planning_mode=planning_mode)
    if not selected:
        raise ValueError("The declared benchmark scope contains no selected tasks")
    return source, selected, scope


def _acquire_lock(lock: Path) -> None:
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        owner = json.loads(lock.read_text(encoding="utf-8"))
        if owner.get("hostname") != os.uname().nodename:
            raise RuntimeError(
                "The run is locked by another host; verify that worker before resuming"
            ) from None
        try:
            os.kill(owner["pid"], 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError("This output directory is already running") from None
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, json.dumps({"pid": os.getpid(), "hostname": os.uname().nodename}).encode())
    os.close(fd)


def _load_latest(*paths: Path) -> dict:
    latest = {}
    for saved in paths:
        if not saved.exists():
            continue
        with saved.open("rb+") as stream:
            while line := stream.readline():
                offset = stream.tell() - len(line)
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if stream.read(1):
                        raise ValueError("Predictions contain a corrupt non-final record") from None
                    stream.truncate(offset)
                    break
                if not line.endswith(b"\n"):
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"\n")
                latest[record["question_id"]] = record
    return latest


def _public_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "attempt_conversation"}


def _normalize(payload: dict) -> dict:
    answer = payload.get("answer") or ""
    payload["answer"] = answer
    payload["final_answer"] = payload.get("final_answer") or answer
    produced = bool(str(answer).strip())
    if payload.get("error") or not produced:
        payload["status"] = "failed"
        if not produced and not payload.get("error"):
            payload["error"] = "no answer"
    else:
        payload["status"] = "succeeded"
        payload.pop("error", None)
    return payload


def _attempt_conversation(path: str) -> Conversation:
    saved = json.loads((Path(path) / "result.json").read_text(encoding="utf-8"))
    if "attempt_conversation" not in saved:
        raise ValueError(f"Attempt is missing its conversation: {path}")
    return as_conversation(saved["attempt_conversation"])


def run_batch(
    *,
    rows,
    output: Path,
    identity: dict,
    retries: int,
    retry_failed: bool,
    concurrency: int,
    opener,
    label: str = "agentgs",
) -> int:
    """Run unfinished questions. ``opener`` yields the async per-attempt callable."""
    if not 0 <= retries <= 10:
        raise ValueError("--retries must be between 0 and 10")
    if not 1 <= concurrency <= 128:
        raise ValueError("Invalid concurrency")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root = output.parent / (output.stem + ".run")
    run_root.mkdir(parents=True, exist_ok=True)
    lock = run_root / "run.lock"
    _acquire_lock(lock)
    try:
        manifest = run_root / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("Resume identity differs; use a new output directory")
        elif output.exists() and output.stat().st_size:
            raise ValueError(
                "Existing predictions have no identity manifest; use a new output directory"
            )
        else:
            atomic_json(manifest, identity)
        history = run_root / "attempts.jsonl"
        latest = _load_latest(output, history)
        if latest:
            atomic_jsonl(output, [latest[key] for key in sorted(latest)])
        pending = [
            row
            for row in rows
            if row.question_id not in latest
            or latest[row.question_id].get("retry_pending")
            or (retry_failed and latest[row.question_id].get("status") != "succeeded")
        ]
        print(
            f"{label}: {len(rows)} selected, {len(pending)} pending, concurrency {concurrency}",
            flush=True,
        )

        async def execute_pending():
            stop = asyncio.Event()
            semaphore = asyncio.Semaphore(concurrency)
            write_lock = asyncio.Lock()
            finished = 0

            def publish(record, snapshot):
                attempt = Path(record["attempt_directory"])
                atomic_json(attempt / "result.json", record)
                atomic_json(attempt / "conversation.json", record["attempt_conversation"])
                with history.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(_public_record(record), ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                atomic_jsonl(output, snapshot)

            async def one(row):
                nonlocal finished
                async with semaphore:
                    if stop.is_set():
                        return
                    previous = latest.get(row.question_id, {})
                    prior_paths = list(
                        previous.get("attempt_directories")
                        or (
                            [previous["attempt_directory"]]
                            if previous.get("attempt_directory")
                            else []
                        )
                    )
                    offset = (
                        previous.get("automatic_attempts_used", 0)
                        if previous.get("retry_pending") and not retry_failed
                        else 0
                    )
                    if offset > retries:
                        raise RuntimeError("Automatic retry state exceeded its configured budget")
                    local = [_attempt_conversation(path) for path in prior_paths]
                    for index in range(offset, retries + 1):
                        if stop.is_set():
                            return
                        attempt = (
                            run_root
                            / "questions"
                            / str(row.question_id)
                            / ("attempt_" + uuid4().hex)
                        )
                        await invoke(attempt.mkdir, parents=True, exist_ok=False)
                        started = time.time()
                        try:
                            payload = await execute(row, attempt)
                        except Exception as exc:
                            payload = {
                                "answer": "",
                                "final_answer": "",
                                "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                                "conversation": Conversation(),
                            }
                        payload = _normalize(dict(payload))
                        conversation = as_conversation(payload.pop("conversation", None))
                        local.append(conversation)
                        aggregated = scoring_conversation(local)
                        prior_paths.append(str(attempt))
                        record = {
                            **payload,
                            "question_id": row.question_id,
                            "category_id": row.category_id,
                            "attempt_directory": str(attempt),
                            "attempt_directories": list(prior_paths),
                            "automatic_attempts_used": index + 1,
                            "retry_pending": payload["status"] != "succeeded"
                            and index < retries
                            and not authentication_failure(payload.get("error")),
                            "attempt_conversation": conversation.model_dump(mode="json"),
                            "conversation": aggregated.model_dump(mode="json"),
                            "started_at": started,
                            "ended_at": time.time(),
                        }
                        public = _public_record(record)
                        async with write_lock:
                            latest[row.question_id] = public
                            snapshot = [latest[key] for key in sorted(latest)]
                            await invoke(publish, record, snapshot)
                        if authentication_failure(payload.get("error")):
                            stop.set()
                        if not record["retry_pending"]:
                            finished += 1
                            print(
                                f"[{finished}/{len(pending)}] Q{row.question_id} {record['status']}",
                                flush=True,
                            )
                            return
                        print(
                            f"Q{row.question_id} retry {index + 1}/{retries}: {payload.get('error', 'failed')}",
                            flush=True,
                        )
                        await asyncio.sleep(min(2**index, 8))
                    raise RuntimeError("Automatic retry state exceeded its configured budget")

            async with opener() as execute:
                await asyncio.gather(*(one(row) for row in pending))
            return stop.is_set()

        stopped = asyncio.run(execute_pending()) if pending else False
        report, code = _summary(rows, latest, output, retries, stopped)
        atomic_json(run_root / "summary.json", report)
        print(json.dumps(report), flush=True)
        return code
    finally:
        lock.unlink()


def _summary(rows, latest, output, retries, stopped) -> tuple[dict, int]:
    def status(row):
        saved = latest.get(row.question_id)
        return saved.get("status") if saved else "missing"

    failed = sum(status(row) != "succeeded" for row in rows)
    report = {
        "selected": len(rows),
        "succeeded": sum(status(row) == "succeeded" for row in rows),
        "failed": failed,
        "output": str(output),
        "retries": retries,
        "attempts": sum(
            len(latest[row.question_id].get("attempt_directories", []))
            for row in rows
            if row.question_id in latest
        ),
        "stopped_for_auth": stopped,
    }
    return report, 1 if failed or stopped else 0
