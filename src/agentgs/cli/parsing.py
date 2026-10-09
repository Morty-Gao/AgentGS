"""Turn CLI values into the objects the runtime and benchmark already accept."""

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from agentgs.contracts.models import TaskInput, ToolCall
from agentgs.messages import Conversation

PLANNING_MODES = ("tool_only", "skill_guided", "data_only")


def passed(ctx: typer.Context, name: str) -> bool:
    # Typer vendors Click, so identity with click.core.ParameterSource does not hold.
    return ctx.get_parameter_source(name).name == "COMMANDLINE"


def parse_json(text: str, label: str):
    raw = Path(text[1:]).expanduser().read_text(encoding="utf-8") if text.startswith("@") else text
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{label} is not valid JSON: {exc.msg}") from None


def parse_object(text: str, label: str) -> dict:
    value = parse_json(text, label)
    if not isinstance(value, dict):
        raise typer.BadParameter(f"{label} must be a JSON object")
    return value


def load_messages(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return Conversation.from_jsonl(text).messages
    data = json.loads(text)
    if isinstance(data, list):
        return Conversation(messages=data).messages
    return Conversation.model_validate(data).messages


def check_planning_mode(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in PLANNING_MODES:
        allowed = ", ".join(PLANNING_MODES)
        raise typer.BadParameter(f"planning_mode must be one of: {allowed}")
    return value


def parse_ids(text: str) -> set | None:
    if not text:
        return None
    chosen = set()
    for part in text.split(","):
        item = part.strip()
        if item:
            chosen.add(int(item) if item.isdigit() else item)
    return chosen


def under_project(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _validate_call(body: dict) -> ToolCall:
    try:
        return ToolCall.model_validate(body)
    except ValidationError as exc:
        details = "; ".join(error["msg"] for error in exc.errors())
        raise typer.BadParameter(f"Invalid tool call: {details}") from None


def extra_calls(values: list[str] | None) -> list[ToolCall]:
    found = []
    for raw in values or []:
        parsed = parse_json(raw, "--call")
        rows = parsed if isinstance(parsed, list) else [parsed]
        for row in rows:
            if not isinstance(row, dict):
                raise typer.BadParameter("--call entries must be JSON objects")
            found.append(_validate_call(row))
    return found


def build_calls(
    tool: str | None,
    arguments: str,
    call: list[str] | None,
    call_id: str | None,
    depends_on: list[str] | None,
    *,
    arguments_passed: bool,
) -> list[ToolCall]:
    if tool is None and (call_id or depends_on or arguments_passed):
        raise typer.BadParameter("--call-id, --depends-on and --arguments require --tool")
    calls = []
    if tool is not None:
        body = {
            "tool": tool,
            "arguments": parse_object(arguments, "--arguments"),
            "depends_on": list(depends_on or []),
        }
        if call_id:
            body["id"] = call_id
        calls.append(_validate_call(body))
    calls.extend(extra_calls(call))
    return calls


def build_task_input(
    ctx: typer.Context,
    *,
    task_file: Path | None,
    question: str,
    tool: str | None,
    arguments: str,
    call: list[str] | None,
    call_id: str | None,
    depends_on: list[str] | None,
    model: str | None,
    effort: str | None,
    messages: Path | None,
    parallel_tools: bool,
    skill_id: list[str] | None,
    max_model_calls: int | None,
    input_resource: str | None,
    file: list[str] | None,
    output_contract: str | None,
    planning_mode: str | None,
    allow_rule_planning: bool | None,
    plan_attempts: int | None,
    tool_attempts: int | None,
    command_attempts: int | None,
    input_domain: str | None,
    dataset_instructions: str | None = None,
    modules: str | None = None,
    ablation: list[str] | None = None,
) -> TaskInput:
    """Start from TaskInput defaults, or a JSON file, and apply explicit flags."""
    if task_file is None:
        data: dict = {}
    else:
        data = TaskInput.model_validate_json(task_file.read_text(encoding="utf-8")).model_dump()

    def put(name: str, key: str, value):
        if passed(ctx, name) or key not in data:
            data[key] = value

    put("question", "question", question)
    put("model", "model", model)
    put("effort", "effort", effort)
    put("parallel_tools", "parallel_tools", parallel_tools)
    put("skill_id", "skill_ids", list(skill_id or []))
    put("max_model_calls", "max_model_calls", max_model_calls)
    put("file", "file_path_list", list(file or []))
    put("planning_mode", "planning_mode", check_planning_mode(planning_mode))
    put("modules", "modules", parse_object(modules, "--modules") if modules else None)
    put("ablation", "ablations", ablation or [])
    put("allow_rule_planning", "allow_rule_planning", allow_rule_planning)
    put("plan_attempts", "plan_attempts", plan_attempts)
    put("tool_attempts", "tool_attempts", tool_attempts)
    put("command_attempts", "command_attempts", command_attempts)
    put("input_domain", "input_domain", input_domain)
    if passed(ctx, "dataset_instructions") or "dataset_instructions" not in data:
        data["dataset_instructions"] = (
            Path(dataset_instructions[1:]).expanduser().read_text(encoding="utf-8")
            if dataset_instructions and dataset_instructions.startswith("@")
            else dataset_instructions or ""
        )
    if passed(ctx, "messages") or "messages" not in data:
        data["messages"] = load_messages(messages) if messages else []
    if passed(ctx, "input_resource") or "input_resource" not in data:
        data["input_resource"] = (
            parse_json(input_resource, "--input-resource") if input_resource is not None else None
        )
    if passed(ctx, "output_contract") or "output_contract" not in data:
        data["output_contract"] = (
            parse_object(output_contract, "--output-contract")
            if output_contract is not None
            else None
        )
    call_fields = ("tool", "arguments", "call", "call_id", "depends_on")
    if any(passed(ctx, name) for name in call_fields) or "calls" not in data:
        data["calls"] = build_calls(
            tool,
            arguments,
            call,
            call_id,
            depends_on,
            arguments_passed=passed(ctx, "arguments"),
        )
    try:
        return TaskInput.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            ".".join(map(str, error["loc"])) + ": " + error["msg"] for error in exc.errors()
        )
        raise typer.BadParameter(f"Invalid task input: {details}") from None


def select_dataset(root: Path, dataset: Path | None, dataset_name: str | None):
    """Use the prepared GeoToolEval path unless a dataset file or name is given."""
    from agentgs.evaluation.datasets.version import benchmark_dataset

    if dataset is None and dataset_name is None:
        return benchmark_dataset(root)
    source = benchmark_dataset(root) if dataset is None else under_project(root, dataset)
    if dataset_name is None:
        return source
    from agentgs.evaluation.datasets import get_dataset

    try:
        return get_dataset(dataset_name, source)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from None
