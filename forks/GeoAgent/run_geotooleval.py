"""GeoToolEval adapter for GeoAgent's P-UCB code search.

Nodes retain executable prefixes, model token probabilities, execution rewards,
repair attempts and remaining-task replanning. Scientific data access uses the
shared tool adapter; labels never enter generation prompts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/benchmark"))

from _answers import extract_program_answer  # noqa: E402
from _runtime import (  # noqa: E402
    add_arguments,
    atomic_json,
    average_logprob,
    batch,
    chat,
    checkpoint,
    conversation,
    question_text,
)
from geotool_bridge import catalog  # noqa: E402


class GenerationError(ValueError):
    """A completed model request did not produce a usable code candidate."""


@dataclass
class Execution:
    ok: bool
    stdout: str
    stderr: str = ""
    calls: list[dict] = field(default_factory=list)
    execution_id: str = ""
    directory: str = ""

    @property
    def text(self):
        return (self.stdout + "\n" + self.stderr).strip()


class Node:
    def __init__(self, code, logprob, parent, step):
        self.id = uuid4().hex
        self.code, self.logprob, self.parent, self.step = code, logprob, parent, step
        self.prob = math.exp(min(0.0, logprob))
        self.children = []
        self.visits, self.value, self.runs = 1, 0.0, 0
        self.error = ""
        self.fix_attempts = []
        self.observations = ""
        self.stdout = ""

    def backprop(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent is not None:
            self.parent.backprop(reward)


def p_ucb_select(parent, children, c_base=10, c=4):
    beta = math.log((parent.visits + c_base + 1) / c_base) + c
    return max(
        children,
        key=lambda child: (
            child.value
            + beta * child.prob * math.sqrt(math.log(parent.visits)) / (1 + child.visits)
        ),
    )


def _parse_json(text):
    body = re.sub(r"^```(?:json)?\s*", "", (text or "").strip()).removesuffix("```").strip()
    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(body[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Model JSON must be an object")
    return result


def _tool_manual():
    # Every generation request needs both tool names and their actual argument schemas.
    return json.dumps(catalog(), ensure_ascii=False, separators=(",", ":"))


def _generate(env, row, instruction, context, client, events, stage, schema=None):
    field_name = "steps" if stage == "decompose" else "code"
    if schema is None:
        value = {"type": "string", "minLength": 1}
        if field_name in {"steps", "codes"}:
            value = {"type": "array", "minItems": 1, "items": value}
        schema = {
            "type": "object",
            "properties": {field_name: value},
            "required": [field_name],
            "additionalProperties": False,
        }
    event = chat(
        env,
        [
            {
                "role": "system",
                "content": (
                    "You are a geoscience Python coding agent. Use only the registered call_tool(name, arguments) tools for data. "
                    "call_tool returns a Python dict or list, not JSON text; tool failures raise an exception. "
                    "Read result fields from actual observations, never invent tool names, keys or values. "
                    "When all required calculations finish, print FINAL: followed by the required answer (which may span lines). "
                    "Do not print placeholder answers. All paths returned by tools may be passed into later calls.\n"
                    + instruction
                    + "\nRegistered tools and argument schemas:\n"
                    + _tool_manual()
                ),
            },
            {"role": "user", "content": question_text(row) + "\n\n" + context},
        ],
        client,
        events,
        stage=stage,
        logprobs=True,
        response_schema=schema,
    )
    if event.get("finish_reason") == "length":
        raise GenerationError("Model output was truncated by its token budget")
    try:
        result = _parse_json(event["response"].get("content"))
    except (ValueError, TypeError) as exc:
        raise GenerationError(f"Invalid model JSON: {exc}") from exc
    probability = (
        [average_logprob(event, "steps", i) for i in range(len(result.get("steps", [])))]
        if stage == "decompose"
        else average_logprob(event, field_name)
    )
    return result, probability


def decompose(env, row, client, events, prefix=""):
    result, probability = _generate(
        env,
        row,
        'Decompose the remaining task into ordered executable steps. Return JSON {"steps": ["..."]}.',
        "Completed executable prefix (keep it):\n" + prefix,
        client,
        events,
        "decompose",
    )
    steps = result.get("steps")
    if (
        not isinstance(steps, list)
        or not steps
        or any(not isinstance(s, str) or not s.strip() for s in steps)
    ):
        raise ValueError("Model returned no valid remaining plan")
    events.append({"phase": "decompose", "steps": steps, "logprob": probability, "prefix": prefix})
    checkpoint(events)
    return steps, probability


def _propose(env, row, step, code, top_k, client, events, observations=""):
    candidates = []
    # Independent samples retain the upstream candidate-generation mechanism.
    for _ in range(top_k):
        try:
            result, probability = _generate(
                env,
                row,
                'Write only the next step, appended to the prefix. Return JSON {"code": "..."}.',
                f"Next step: {step}\nExecutable prefix:\n{code}\nActual prefix observations:\n{observations}",
                client,
                events,
                "propose",
            )
        except GenerationError as exc:
            events.append({"phase": "generation_error", "stage": "propose", "error": str(exc)})
            continue
        snippet = result.get("code")
        if not isinstance(snippet, str) or not snippet.strip():
            events.append(
                {
                    "phase": "generation_error",
                    "stage": "propose",
                    "error": "Model returned an empty code candidate",
                }
            )
            continue
        candidates.append((snippet, probability))
    return candidates


def _repair(env, row, node, client, events):
    prefix = node.parent.code if node.parent else "from geotool_bridge import call_tool\n"
    result, probability = _generate(
        env,
        row,
        'Repair only the failing step. Keep the executable prefix unchanged. Return JSON {"code": "replacement step"}.',
        f"Prefix:\n{prefix}\nFailing script:\n{node.code}\nActual observations:\n{node.observations}\nError:\n{node.error}",
        client,
        events,
        "repair",
    )
    snippet = result.get("code")
    if not isinstance(snippet, str) or not snippet.strip():
        raise ValueError("Model returned an empty repair")
    return prefix.rstrip() + "\n" + snippet + "\n", probability


def _complete(env, row, code, remaining, client, events, observations=""):
    for step in remaining:
        result, _ = _generate(
            env,
            row,
            'Complete this remaining simulation step only. Append to the prefix; do not repeat it. Return JSON {"code": "..."}.',
            f"Step: {step}\nExecutable prefix:\n{code}\nActual prefix observations:\n{observations}",
            client,
            events,
            "complete",
        )
        snippet = result.get("code")
        if not isinstance(snippet, str) or not snippet.strip():
            raise ValueError("Model returned an empty simulation step")
        yield snippet
        code += "\n" + snippet + "\n"


def _execute(code, workdir, timeout):
    execution_id = uuid4().hex
    directory = Path(workdir) / "executions" / execution_id
    directory.mkdir(parents=True)
    (directory / "program.py").write_text(code, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    )
    env["AGENTGS_TOOL_OUTPUT"] = str(directory / "tools")
    receipts = directory / "tool-calls.jsonl"
    env["AGENTGS_TOOL_RECEIPTS"] = str(receipts)
    stdout, stderr = "", ""
    ok = False
    try:
        completed = subprocess.run(
            [sys.executable, str(directory / "program.py")],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout, stderr, ok = completed.stdout, completed.stderr, completed.returncode == 0
    except subprocess.TimeoutExpired as exc:

        def decode(value):
            return (
                value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
            )

        stdout, stderr = decode(exc.stdout), decode(exc.stderr) + "\nTimeoutError"
    calls = (
        [json.loads(line) for line in receipts.read_text().splitlines()]
        if receipts.exists()
        else []
    )
    # A program cannot hide a failed scientific call by catching the exception.
    ok = ok and all(c["status"] == "success" for c in calls)
    (directory / "stdout.txt").write_text(stdout, encoding="utf-8")
    (directory / "stderr.txt").write_text(stderr, encoding="utf-8")
    atomic_json(
        directory / "execution.json", {"ok": ok, "execution_id": execution_id, "calls": calls}
    )
    return Execution(ok, stdout, stderr, calls, execution_id, str(directory))


def _dump_node(node):
    return {
        "id": node.id,
        "parent_id": node.parent.id if node.parent else None,
        "step": node.step,
        "logprob": node.logprob,
        "prob": node.prob,
        "visits": node.visits,
        "value": node.value,
        "runs": node.runs,
        "error": node.error,
        "code": node.code,
        "fix_attempts": node.fix_attempts,
        "children": [_dump_node(c) for c in node.children],
    }


def search(env, row, workdir, rollouts, top_k, repairs, timeout, client, events=None):
    events = events if events is not None else []
    root = Node("from geotool_bridge import call_tool\n", 0.0, None, 0)
    roots = []
    steps, plan_probability = [], 0.0
    best = {"answer": "", "reward": 0.0, "rollouts": 0, "status": "failed"}
    selected = None
    last = None

    def execute(code, node, kind, rollout):
        nonlocal last
        last = _execute(code, workdir, timeout)
        events.append(
            {
                "phase": "execute",
                "kind": kind,
                "rollout": rollout,
                "node_id": node.id,
                "execution_id": last.execution_id,
                "directory": last.directory,
                "ok": last.ok,
                "stdout": last.stdout,
                "stderr": last.stderr,
                "calls": last.calls,
            }
        )
        checkpoint(events)
        atomic_json(
            Path(workdir) / "search.json",
            {"steps": steps, "events": events, "tree": _dump_node(root), "pruned_trees": roots},
        )
        return last

    def accept(outcome, node, code, *, terminal=None, prefix_stdout=None):
        nonlocal selected
        if not outcome.ok:
            return False
        extraction = extract_program_answer(
            outcome.stdout,
            terminal=node.step >= len(steps) if terminal is None else terminal,
            prefix_stdout=(node.parent.stdout if node.parent else "")
            if prefix_stdout is None
            else prefix_stdout,
        )
        answer = extraction["answer"]
        events.append(
            {"phase": "answer_extraction", "execution_id": outcome.execution_id, **extraction}
        )
        if not answer:
            return False
        selected = outcome
        best.update(
            answer=answer,
            answer_extraction=extraction,
            reward=1.0,
            status="succeeded",
            code=code,
            selected_node_id=node.id,
            selected_execution_id=outcome.execution_id,
            artifacts_directory=outcome.directory,
        )
        return True

    try:
        steps, plan_probability = decompose(env, row, client, events)
        for rollout in range(1, rollouts + 1):
            best["rollouts"] = rollout
            node, path = root, [root.id]
            while node.children:
                node = p_ucb_select(node, node.children)
                node.visits += 1
                path.append(node.id)
            events.append({"phase": "select", "rollout": rollout, "path": path})
            if node.error:
                if node.runs < repairs:
                    node.runs += 1
                    try:
                        repaired, probability = _repair(env, row, node, client, events)
                    except GenerationError as exc:
                        events.append(
                            {
                                "phase": "generation_error",
                                "stage": "repair",
                                "node_id": node.id,
                                "error": str(exc),
                            }
                        )
                        continue
                    outcome = execute(repaired, node, "repair", rollout)
                    reward = float(outcome.ok)
                    node.fix_attempts.append(
                        {
                            "code": repaired,
                            "reward": reward,
                            "error": outcome.text if not outcome.ok else "",
                        }
                    )
                    node.backprop(reward)
                    step_probability = (
                        plan_probability[node.step - 1]
                        if isinstance(plan_probability, list)
                        else plan_probability
                    )
                    prior = 0.5 * (probability + step_probability)
                    node.code, node.logprob, node.prob = repaired, prior, math.exp(min(0.0, prior))
                    node.error = "" if outcome.ok else outcome.text
                    node.observations = json.dumps(outcome.calls, ensure_ascii=False, default=str)
                    node.stdout = outcome.stdout if outcome.ok else ""
                    if accept(outcome, node, repaired):
                        break
                    continue
                anchor = node.parent or root
                # Replan only the remaining tasks from a verified prefix, and
                # prune branches whose step indices belong to the previous plan.
                prefix = anchor.code
                steps, plan_probability = decompose(env, row, client, events, prefix)
                roots.append(_dump_node(root))
                root = Node(prefix, 0.0, None, 0)
                root.observations = anchor.observations
                root.stdout = anchor.stdout
                events.append(
                    {"phase": "replan", "prefix": prefix, "new_root_id": root.id, "steps": steps}
                )
                continue
            if node.step >= len(steps):
                outcome = execute(node.code, node, "finished-plan", rollout)
                if accept(outcome, node, node.code):
                    break
                node.error = outcome.text or "Program completed without an extractable final answer"
                continue
            candidates = _propose(
                env, row, steps[node.step], node.code, top_k, client, events, node.observations
            )
            for snippet, code_probability in candidates:
                full = node.code.rstrip() + "\n" + snippet + "\n"
                step_probability = (
                    plan_probability[node.step]
                    if isinstance(plan_probability, list)
                    else plan_probability
                )
                child = Node(full, 0.5 * (code_probability + step_probability), node, node.step + 1)
                node.children.append(child)
                outcome = execute(full, child, "expand", rollout)
                child.error = "" if outcome.ok else outcome.text
                child.observations = json.dumps(outcome.calls, ensure_ascii=False, default=str)
                child.stdout = outcome.stdout if outcome.ok else ""
                reward = float(outcome.ok)
                answer_ready = accept(outcome, child, full)
                if outcome.ok and not answer_ready and child.step < len(steps):
                    remaining = steps[child.step :]
                    codes = _complete(env, row, full, remaining, client, events, child.observations)
                    successful = 0
                    simulation_code = full
                    simulation_stdout = outcome.stdout
                    try:
                        for code in codes:
                            simulation_code += "\n" + code + "\n"
                            simulation = execute(simulation_code, child, "simulate", rollout)
                            if not simulation.ok:
                                break
                            successful += 1
                            answer_ready = accept(
                                simulation,
                                child,
                                simulation_code,
                                terminal=successful == len(remaining),
                                prefix_stdout=simulation_stdout,
                            )
                            simulation_stdout = simulation.stdout
                            if answer_ready:
                                break
                    except (ValueError, RuntimeError) as exc:
                        events.append(
                            {
                                "phase": "generation_error",
                                "node_id": child.id,
                                "stage": "complete",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    reward = 1.0 if answer_ready else successful / len(remaining)
                child.backprop(reward)
                events.append({"phase": "backprop", "node_id": child.id, "reward": reward})
                if answer_ready:
                    break
            if best["status"] == "succeeded":
                break
        if best["status"] != "succeeded":
            best["error"] = "Search budget exhausted without a successfully executed final answer"
    except Exception as exc:
        best.update(status="failed", answer="", error=f"{type(exc).__name__}: {exc}")
    best["output"] = selected.text if selected else last.text if last else ""
    best["trace"] = {
        "schema": "geoagent-mcts-v2",
        "steps": steps,
        "events": events,
        "tree": _dump_node(root),
        "pruned_trees": roots,
        "selected_execution_id": selected.execution_id if selected else None,
    }
    best["conversation"] = conversation(
        question_text(row), events, calls=selected.calls if selected else last.calls if last else ()
    )
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description="GeoAgent MCTS over 603 GeoToolEval questions")
    add_arguments(parser)
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--repairs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if args.rollouts < 1 or args.top_k < 1 or args.repairs < 0 or args.timeout <= 0:
        parser.error("Invalid search or execution budget")

    def run(env, row, attempt, client, events):
        return search(
            env, row, attempt, args.rollouts, args.top_k, args.repairs, args.timeout, client, events
        )

    return batch(args, "geoagent", __file__, run)


if __name__ == "__main__":
    raise SystemExit(main())
