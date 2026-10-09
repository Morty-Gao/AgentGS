"""GeoToolEval data and tool adapter around Earth-Agent's ReAct loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/benchmark"))

from _answers import extract_answer  # noqa: E402
from _runtime import (  # noqa: E402
    add_arguments,
    batch,
    chat,
    checkpoint,
    conversation,
    question_text,
    rewrite,
)
from geotool_bridge import call_tool, catalog, tool_workspace  # noqa: E402

SYSTEM = """You are a geoscientist. Answer by calling the registered tools.
Use fields from the evaluation input resource exactly when needed.
A benchmark/data/questionN path is available as data/earthbench/questionN.
If a tool returns an error, retry that call at most once with corrected arguments.
Use actual result fields, never invent data or result keys. When observations
are sufficient, reply only with the final answer in the required format."""


def _arguments(raw):
    value = json.loads(raw) if isinstance(raw, str) else raw or {}
    if not isinstance(value, dict):
        raise TypeError("tool arguments must be a JSON object")
    return value


def answer_question(env, tools, row, max_steps, client, attempt, events):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question_text(row)},
    ]
    turns, used = [], []
    failure_counts = {}
    final, error = "", ""
    extraction = {"answer": "", "method": "missing"}
    receipts = Path(attempt) / "tool-calls.jsonl"
    try:
        with tool_workspace(Path(attempt) / "tools", receipts):
            for index in range(max_steps):
                event = chat(
                    env, messages, client, events, stage="react", tools=tools, temperature=0.1
                )
                if event.get("finish_reason") == "length":
                    raise RuntimeError("Model output was truncated by its token budget")
                archived = event["response"]
                calls = archived.get("tool_calls") or []
                content = archived.get("content") or ""
                message = {
                    "role": "assistant",
                    "content": content,
                    **({"tool_calls": calls} if calls else {}),
                }
                messages.append(message)
                observations = []
                turn = {"index": index, "assistant": archived, "observations": observations}
                turns.append(turn)
                if not calls:
                    extraction = extract_answer(content, allow_plain=True)
                    final = extraction["answer"]
                    if not final:
                        error = "Model returned an empty or ambiguous final answer"
                    break
                for call in calls:
                    function = call.get("function") or {}
                    name = function.get("name") or ""
                    used.append(name)
                    try:
                        arguments = _arguments(function.get("arguments"))
                        if failure_counts.get(name, 0) >= 2:
                            result = {
                                "status": "error",
                                "error": "Retry budget exhausted for this tool",
                                "tool": name,
                            }
                        else:
                            result = call_tool(name, arguments)
                    except Exception as exc:
                        arguments = function.get("arguments")
                        result = {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "tool": name,
                        }
                    status = (
                        "error"
                        if isinstance(result, dict) and result.get("status") == "error"
                        else "success"
                    )
                    if status == "error":
                        failure_counts[name] = failure_counts.get(name, 0) + 1
                    result_text = (
                        result
                        if isinstance(result, str)
                        else json.dumps(result, ensure_ascii=False, default=str)
                    )
                    observations.append(
                        {
                            "tool_call_id": call["id"],
                            "name": name,
                            "arguments": arguments,
                            "bound_arguments": rewrite(arguments),
                            "result": result_text,
                            "status": status,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name,
                            "content": result_text,
                        }
                    )
                    checkpoint({"events": events, "turns": turns, "messages": messages})
            else:
                error = "Tool-call budget exhausted without a final answer"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    # Reasoning and provider usage remain in the archived replies; canonical
    # dialogue carries exact arguments and full (untruncated) tool observations.
    canonical = []
    canonical.extend(messages[:2])
    for turn in turns:
        msg = dict(turn["assistant"])
        msg.pop("usage", None)
        canonical.append(msg)
        for observation in turn["observations"]:
            canonical.append(
                {
                    "role": "tool",
                    "tool_call_id": observation["tool_call_id"],
                    "name": observation["name"],
                    "content": observation["result"],
                    "metadata": {
                        "status": observation["status"],
                        "tool_input": observation["bound_arguments"],
                    },
                }
            )
    outcome = {
        "answer": final,
        "answer_extraction": extraction,
        "tools": used,
        "steps": len(used),
        "status": "failed" if error or not final else "succeeded",
        "trace": {
            "schema": "earth-agent-react-v2",
            "system": SYSTEM,
            "prompt": question_text(row),
            "turns": turns,
            "events": events,
        },
        "conversation": conversation(question_text(row), events, dialogue=canonical),
    }
    if error:
        outcome["error"] = error
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description="Earth-Agent ReAct over 603 GeoToolEval questions")
    add_arguments(parser)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    tools = [{"type": "function", "function": row} for row in catalog()]

    def run(env, row, attempt, client, events):
        return answer_question(env, tools, row, args.max_steps, client, attempt, events)

    return batch(args, "earthagent", __file__, run)


if __name__ == "__main__":
    raise SystemExit(main())
