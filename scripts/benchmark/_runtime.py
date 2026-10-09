"""Transport, tool receipts and durable batches for benchmark adapters."""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import httpx


def _repository_root(start: Path) -> Path:
    candidate = start.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"No pyproject.toml found above {candidate}")


ROOT = _repository_root(Path(__file__))
_output = contextvars.ContextVar("fork_tool_output", default=None)
_receipts = contextvars.ContextVar("fork_tool_receipts", default=None)
_checkpoint = contextvars.ContextVar("fork_checkpoint", default=None)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_jsonl(path, records):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checkpoint(events):
    path = _checkpoint.get()
    if path:
        atomic_json(path, events)


def project_root():
    return ROOT


@lru_cache(maxsize=1)
def _registry():
    import sys

    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from agentgs.runtime.tools.registry import builtin_tools

    return builtin_tools()


def catalog():
    rows = []
    for name, item in _registry().items():
        schema = dict(item.spec.input_schema or {})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        schema.setdefault("required", [])
        description = (item.spec.description or "").strip()
        if item.spec.input_description:
            description += "\nInput: " + item.spec.input_description
        if item.spec.output_description:
            description += "\nOutput: " + item.spec.output_description
        rows.append({"name": name, "description": description, "parameters": schema})
    return rows


def rewrite(value):
    if isinstance(value, str):
        return value.replace("benchmark/data/question", "data/earthbench/question")
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    return value


@contextmanager
def tool_workspace(path, receipts=None):
    token = _output.set(str(Path(path).resolve()))
    receipt_token = _receipts.set(str(receipts) if receipts else None)
    try:
        yield
    finally:
        _receipts.reset(receipt_token)
        _output.reset(token)


def call_tool(name, arguments=None, *, strict=False):
    registry = _registry()
    from agentgs.datasets.registry import DatasetRegistry
    from agentgs.runtime.tools.compute import compute_tool
    from agentgs.runtime.tools.context import tool_context

    call_id = uuid4().hex
    original = arguments if arguments is not None else {}
    bound = rewrite(original)
    output = Path(_output.get() or os.environ.get("AGENTGS_TOOL_OUTPUT") or ROOT / "tmp/fork-tools")
    receipt_path = _receipts.get() or os.environ.get("AGENTGS_TOOL_RECEIPTS")
    started = time.time()
    result = None
    status = "error"
    try:
        if not isinstance(bound, dict):
            raise TypeError("arguments must be a JSON object")
        if name not in registry:
            raise KeyError(f"Unknown tool {name}; use a tool from the supplied catalog")
        with tool_context(ROOT, output):
            result = compute_tool(name, bound, DatasetRegistry(ROOT), registry)
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass
        status = (
            "error" if isinstance(result, dict) and result.get("status") == "error" else "success"
        )
    except Exception as exc:
        result = {"status": "error", "tool": name, "error": f"{type(exc).__name__}: {exc}"}
    receipt = {
        "call_id": call_id,
        "name": name,
        "arguments": original,
        "bound_arguments": bound,
        "result": result,
        "status": status,
        "started_at": started,
        "ended_at": time.time(),
        "artifact_directory": str(output),
    }
    if receipt_path:
        path = Path(receipt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False, default=str) + "\n")
            stream.flush()
    if strict and status != "success":
        raise RuntimeError(json.dumps(result, ensure_ascii=False, default=str))
    return result


def load_questions(path=None):
    source = Path(path or ROOT / "data/geotooleval.jsonl")
    rows, seen = [], set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        qid = int(raw.get("question_id", raw.get("题目序号")))
        question = raw.get("question", raw.get("题目正文"))
        if not isinstance(question, str) or not question.strip() or qid in seen:
            raise ValueError(f"{source}:{line_number}: invalid or duplicate question {qid}")
        seen.add(qid)
        fmt = raw.get("answer_format") or {}
        resource = raw.get("input_resource", raw.get("输入资源", ""))
        rows.append(
            {
                "question_id": qid,
                "category_id": int(raw.get("category_id", raw.get("所属类别"))),
                "question": question,
                "input_resource": resource
                if isinstance(resource, str)
                else json.dumps(resource, ensure_ascii=False),
                "answer_type": raw.get("answer_type", ""),
                "answer_instruction": fmt.get("llm_instruction") or raw.get("format_notes") or "",
                "answer_template": raw.get("answer_template") or fmt.get("template") or "",
                "expected_answer": raw.get("expected_answer", raw.get("题目答案", "")),
            }
        )
    if len(rows) != 603 or {r["category_id"] for r in rows} != set(range(1, 11)):
        raise ValueError("GeoToolEval requires 603 unique questions in ten domains")
    return rows


def question_text(row):
    parts = [row["question"].strip(), "", "Evaluation input resource:", row["input_resource"]]
    if row["answer_template"]:
        parts.extend(["", "Answer template:", row["answer_template"]])
    if row["answer_instruction"]:
        parts.extend(["", row["answer_instruction"]])
    return "\n".join(parts)


def environment(args):
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from agentgs.config.env import environment as read_environment

    env = read_environment(ROOT)
    for attr, key in [
        ("base_url", "LLM_BASE_URL"),
        ("model", "LLM_MODEL"),
        ("api_key", "LLM_API_KEY"),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            env[key] = value
    env["max_output_tokens"] = args.max_output_tokens
    env["output_token_parameter"] = args.output_token_parameter
    env["reasoning_effort"] = getattr(args, "reasoning_effort", None)
    if not env.get("LLM_BASE_URL") or not env.get("LLM_MODEL"):
        raise ValueError("LLM_BASE_URL and LLM_MODEL are required")
    return env


def archive_message(message, usage=None):
    archived = dict(message)
    archived.setdefault("role", "assistant")
    if usage is not None:
        archived["usage"] = usage
    return archived


def chat(
    env,
    messages,
    client,
    events,
    *,
    stage,
    temperature=0.7,
    tools=None,
    logprobs=False,
    response_schema=None,
):
    payload = {
        "model": env["LLM_MODEL"],
        "messages": messages,
        "temperature": temperature,
        env.get("output_token_parameter", "max_tokens"): int(env.get("max_output_tokens", 8192)),
    }
    effort = os.environ.get("AGENTGS_REASONING_EFFORT", "").strip() or env.get("reasoning_effort")
    if effort:
        payload["reasoning_effort"] = effort
    if tools is not None:
        payload.update(tools=tools, tool_choice="auto")
    if logprobs:
        payload["logprobs"] = True
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": stage, "strict": True, "schema": response_schema},
        }
    from agentgs.client.openai_compatible import request_extras

    payload.update(request_extras())
    # A ReAct caller appends to its dialogue after each reply. Freeze the exact
    # wire input before that mutation so earlier requests remain reproducible.
    payload = deepcopy(payload)
    request_id = uuid4().hex
    event = {
        "phase": "model",
        "stage": stage,
        "request_id": request_id,
        "request": payload,
        "started_at": time.time(),
        "status": "started",
    }
    events.append(event)
    checkpoint(events)
    try:
        headers = (
            {"Authorization": "Bearer " + env["LLM_API_KEY"]} if env.get("LLM_API_KEY") else {}
        )
        response = client.post(
            env["LLM_BASE_URL"].rstrip("/") + "/chat/completions", headers=headers, json=payload
        )
        event["http_status"] = response.status_code
        response.raise_for_status()
        body = response.json()
        event["raw_response"] = body
        choice = body["choices"][0]
        event["response"] = archive_message(choice["message"], body.get("usage"))
        event["logprobs"] = choice.get("logprobs")
        event["finish_reason"] = choice.get("finish_reason")
        event["status"] = "completed"
        return event
    except Exception as exc:
        event.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        event["ended_at"] = time.time()
        checkpoint(events)


_STOP_TOKENS = {"<｜end▁of▁sentence｜>", "<|endoftext|>", "<|im_end|>"}


def _token_bytes(token):
    if token.get("bytes") is not None:
        return bytes(token["bytes"])
    return (token.get("token") or "").encode()


def _message_tokens(tokens, content):
    """Use the token suffix that is the message body.

    Thinking text and the chat stop token stay in the logprob list, but they
    are not part of message.content.
    """
    body = content.encode()
    kept = [token for token in tokens if token.get("token") not in _STOP_TOKENS]
    if not body:
        return []
    chosen = []
    total = 0
    for token in reversed(kept):
        width = len(_token_bytes(token))
        chosen.append(token)
        total += width
        if total == len(body):
            ordered = list(reversed(chosen))
            if b"".join(_token_bytes(item) for item in ordered) == body:
                return ordered
            break
        if total > len(body):
            break
    raise ValueError("API token logprobs do not align with the returned content")


def average_logprob(event, field_name=None, index=None):
    tokens = (event.get("logprobs") or {}).get("content") or []
    if not tokens:
        raise RuntimeError("GeoAgent requires API token logprobs; this model response has none")
    content = event.get("response", {}).get("content") or ""
    tokens = _message_tokens(tokens, content)
    bounds = None
    if field_name:
        import re

        match = re.search(r'(?<!\\)"' + re.escape(field_name) + r'"\s*:\s*', content)
        if match is None:
            raise ValueError(f"Model JSON has no {field_name} probability span")
        value, end = json.JSONDecoder().raw_decode(content, match.end())
        start = match.end()
        if index is not None:
            if not isinstance(value, list) or not 0 <= index < len(value):
                raise ValueError("Invalid probability array index")
            position = start + 1
            for item_index in range(index + 1):
                while content[position] in " \t\r\n,":
                    position += 1
                start = position
                _, position = json.JSONDecoder().raw_decode(content, position)
                if item_index == index:
                    end = position
        bounds = (len(content[:start].encode()), len(content[:end].encode()))
    offset, values = 0, []
    for token in tokens:
        width = len(_token_bytes(token))
        if bounds is None or offset < bounds[1] and offset + width > bounds[0]:
            values.append(token.get("logprob"))
        offset += width
    if offset != len(content.encode()):
        raise ValueError("API token logprobs do not align with the returned content")
    if not values or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
        raise RuntimeError("GeoAgent requires API token logprobs; this model response has none")
    probability = sum(math.exp(min(0.0, value)) for value in values) / len(values)
    return math.log(max(probability, 1e-300))


def conversation(prompt, events, dialogue=None, calls=()):
    from agentgs.messages import Conversation

    normalized = []
    for original in dialogue or [{"role": "user", "content": prompt}]:
        message = dict(original)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    },
                }
                for call in message["tool_calls"]
            ]
        normalized.append(message)
    result = Conversation.from_openai(normalized)
    for event in events:
        if event.get("phase") != "model":
            continue
        result.record_event(
            "model.started",
            {
                "request_id": event["request_id"],
                "stage": event["stage"],
                "request_payload": event["request"],
            },
        )
        result.record_event(
            "model.completed" if event["status"] == "completed" else "model.failed",
            {
                "request_id": event["request_id"],
                "usage": event.get("raw_response", {}).get("usage"),
                "attempts": 1,
                "error": event.get("error"),
            },
        )
    for call in calls:
        call_id = result.begin_tool(
            call["name"], json.dumps(call["arguments"], ensure_ascii=False), call_id=call["call_id"]
        )
        result.finish_tool(
            call_id,
            json.dumps(call["result"], ensure_ascii=False, default=str),
            tool_input=call["bound_arguments"],
            status=call["status"],
        )
    return result.model_dump(mode="json")


def add_arguments(parser):
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ids", type=int, nargs="*")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort")
    parser.add_argument(
        "--output-token-parameter",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_tokens",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retries", type=int, default=3)


def batch(args, agent, source, run_one):
    retries = getattr(args, "retries", 3)
    if not 0 <= retries <= 10:
        raise ValueError("--retries must be between 0 and 10")
    if not 1 <= args.concurrency <= 128 or args.request_timeout <= 0 or args.max_output_tokens < 1:
        raise ValueError("Invalid concurrency, timeout or output token budget")
    rows = load_questions(args.questions)
    if args.ids:
        selected = set(args.ids)
        missing = selected - {r["question_id"] for r in rows}
        if missing:
            raise ValueError(f"Unknown question ids: {sorted(missing)}")
        rows = [r for r in rows if r["question_id"] in selected]
    output = (args.output or ROOT / f"outputs/evaluations/{agent}/predictions.jsonl").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root = output.parent / (output.stem + ".run")
    run_root.mkdir(parents=True, exist_ok=True)
    lock = run_root / "run.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        owner = json.loads(lock.read_text())
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
    try:
        env = environment(args)
        source_paths = [
            Path(source),
            ROOT / "scripts/benchmark/_runtime.py",
            Path(source).with_name("geotool_bridge.py"),
        ]
        for directory in (
            "config",
            "contracts",
            "datasets",
            "messages",
            "runtime/tools",
            "evaluation",
            "benchmark",
        ):
            for suffix in ("*.py", "*.json"):
                source_paths.extend(sorted((ROOT / "src/agentgs" / directory).rglob(suffix)))
        source_paths.extend(
            p
            for p in [
                ROOT / "src/agentgs/cli/parsing.py",
                ROOT / "src/agentgs/client/config.py",
                ROOT / "src/agentgs/client/models.py",
                ROOT / "pyproject.toml",
                ROOT / "uv.lock",
            ]
            if p.exists()
        )
        source_paths.extend(sorted((ROOT / "scripts/benchmark").rglob("*.py")))
        dataset = Path(args.questions or ROOT / "data/geotooleval.jsonl")
        settings = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in {"api_key", "output", "retry_failed", "ids"}
        }
        settings.update(model=env["LLM_MODEL"], base_url=env["LLM_BASE_URL"])
        identity = {
            "agent": agent,
            "protocol": "fork-geotooleval-v2",
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source_paths
            },
            "settings": settings,
        }
        identity["prepared_inputs"] = {
            str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted((ROOT / "data").rglob("*"))
            if p.is_file() and "raw" not in p.relative_to(ROOT / "data").parts
        }
        manifest = run_root / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Resume identity differs; use a new output directory")
        elif output.exists() and output.stat().st_size:
            raise ValueError(
                "Existing predictions have no identity manifest; use a new output directory"
            )
        else:
            atomic_json(manifest, identity)
        latest = {}
        history = run_root / "attempts.jsonl"
        for saved in [output, history]:
            if not saved.exists():
                continue
            with saved.open("rb+") as stream:
                while line := stream.readline():
                    offset = stream.tell() - len(line)
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        if stream.read(1):
                            raise ValueError(
                                "Predictions contain a corrupt non-final record"
                            ) from None
                        stream.truncate(offset)
                        break
                    if not line.endswith(b"\n"):
                        stream.seek(0, os.SEEK_END)
                        stream.write(b"\n")
                    latest[int(record["question_id"])] = record
        if latest:
            atomic_jsonl(output, [latest[qid] for qid in sorted(latest)])
        pending = [
            r
            for r in rows
            if r["question_id"] not in latest
            or (args.retry_failed and latest[r["question_id"]].get("status") != "succeeded")
            or latest[r["question_id"]].get("retry_pending", False)
        ]
        print(
            f"{agent}: {len(rows)} selected, {len(pending)} pending, {len(catalog())} tools, concurrency {args.concurrency}",
            flush=True,
        )
        client = httpx.Client(
            timeout=httpx.Timeout(args.request_timeout, connect=min(30.0, args.request_timeout)),
            limits=httpx.Limits(
                max_connections=args.concurrency, max_keepalive_connections=args.concurrency
            ),
        )

        def run_attempt(row):
            qroot = run_root / "questions" / str(row["question_id"])
            attempt = qroot / ("attempt_" + uuid4().hex)
            attempt.mkdir(parents=True)
            token = _checkpoint.set(attempt / "model-events.json")
            events = []
            record = {
                "question_id": row["question_id"],
                "category_id": row["category_id"],
                "attempt_directory": str(attempt),
                "started_at": time.time(),
            }
            try:
                record.update(run_one(env, row, attempt, client, events))
            except Exception as exc:
                record.update(
                    answer="",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    trace={"events": events},
                )
                record["conversation"] = conversation(question_text(row), events)
            finally:
                _checkpoint.reset(token)
            record.setdefault(
                "status",
                "succeeded" if record.get("answer") and not record.get("error") else "failed",
            )
            record["ended_at"] = time.time()
            return record

        write_lock = threading.Lock()

        def run_one_tracked(row):
            from agentgs.messages import Conversation

            previous = latest.get(row["question_id"], {})
            prior_paths = list(
                previous.get("attempt_directories")
                or ([previous["attempt_directory"]] if previous.get("attempt_directory") else [])
            )
            offset = (
                previous.get("automatic_attempts_used", 0)
                if previous.get("retry_pending") and not args.retry_failed
                else 0
            )
            model_events = []
            for path in prior_paths:
                prior = json.loads((Path(path) / "result.json").read_text())
                model_events.extend(
                    e for e in prior.get("trace", {}).get("events", []) if e.get("phase") == "model"
                )
            for index in range(offset, retries + 1):
                record = run_attempt(row)
                prior_paths.append(record["attempt_directory"])
                record.update(
                    attempt_directories=list(prior_paths),
                    automatic_attempts_used=index + 1,
                    retry_pending=record["status"] != "succeeded" and index < retries,
                )
                model_events.extend(
                    e
                    for e in record.get("trace", {}).get("events", [])
                    if e.get("phase") == "model"
                )
                # Domain scoring uses the current execution. Cost includes every
                # failed/successful attempt, with unknown usage left unknown.
                combined = Conversation.model_validate(record["conversation"])
                combined.messages = [
                    m
                    for m in combined.messages
                    if not (
                        m.role == "event" and str(m.metadata.get("type", "")).startswith("model.")
                    )
                ]
                for event in model_events:
                    combined.record_event(
                        "model.started",
                        {
                            "request_id": event["request_id"],
                            "stage": event["stage"],
                            "request_payload": event["request"],
                        },
                    )
                    combined.record_event(
                        "model.completed" if event["status"] == "completed" else "model.failed",
                        {
                            "request_id": event["request_id"],
                            "usage": event.get("raw_response", {}).get("usage"),
                            "attempts": 1,
                            "error": event.get("error"),
                        },
                    )
                record["conversation"] = combined.model_dump(mode="json")
                attempt = Path(record["attempt_directory"])
                atomic_json(attempt / "result.json", record)
                atomic_json(attempt / "conversation.json", record["conversation"])
                # Publish each failed attempt before retrying so interruption
                # recovery preserves both the evidence and the remaining budget.
                with write_lock:
                    stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    latest[record["question_id"]] = record
                    atomic_jsonl(output, [latest[qid] for qid in sorted(latest)])
                if not record["retry_pending"]:
                    return record
                print(
                    f"Q{row['question_id']} retry {index + 1}/{retries}: {record.get('error', 'failed')}",
                    flush=True,
                )
                time.sleep(min(2**index, 8))
            raise RuntimeError("Automatic retry state exceeded its configured budget")

        # Give instrumentation and callers a stable question boundary.
        run_one_tracked.__name__ = "run_one"
        with (
            client,
            history.open("a", encoding="utf-8") as stream,
            ThreadPoolExecutor(max_workers=args.concurrency) as pool,
        ):
            futures = [pool.submit(run_one_tracked, row) for row in pending]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                print(
                    f"[{index}/{len(pending)}] Q{record['question_id']} {record['status']}",
                    flush=True,
                )
        report = {
            "selected": len(rows),
            "succeeded": sum(latest[r["question_id"]].get("status") == "succeeded" for r in rows),
            "failed": sum(latest[r["question_id"]].get("status") != "succeeded" for r in rows),
            "output": str(output),
            "retries": retries,
            "attempts": sum(
                len(latest[r["question_id"]].get("attempt_directories", [])) for r in rows
            ),
        }
        atomic_json(run_root / "summary.json", report)
        print(json.dumps(report), flush=True)
        return 1 if report["failed"] else 0
    finally:
        lock.unlink()
