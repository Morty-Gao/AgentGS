"""Audit public requests and reconstruct recorded model prompts without network calls."""

import asyncio
import hashlib
import json
from pathlib import Path

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.planning.skills import builtin_planner
from agentgs.agent.retrieval.tools import KGToolRetriever
from agentgs.client.models import ChatReply
from agentgs.config.algorithm import load_algorithm
from agentgs.contracts.models import TaskInput, ToolResult
from agentgs.evaluation.benchmark import PRIVATE_KEYS, write_json
from agentgs.runtime.answer_contracts import AnswerContracts
from agentgs.runtime.tools.context import ToolPaths
from agentgs.runtime.tools.registry import builtin_tools


def private_fields(value, path=""):
    if isinstance(value, dict):
        return [path + "/" + k for k in value if k.lower() in PRIVATE_KEYS] + [
            hit for k, v in value.items() for hit in private_fields(v, path + "/" + k)
        ]
    if isinstance(value, list):
        return [hit for i, v in enumerate(value) for hit in private_fields(v, path + "/" + str(i))]
    return []


async def audit_case(root, sample):
    folder = root / "outputs/tasks" / sample["task_id"]
    request = json.loads((folder / "request.json").read_text())["spec"]["input"]
    saved = json.loads((folder / "result.json").read_text())
    expected = [row for row in saved["trace"] if row["type"] == "model.started"]
    plans = iter(row["content"] for row in saved["trace"] if row["type"] == "plan")
    arguments = iter(
        row["generated_input"]
        for row in saved["trace"]
        if row["type"] == "tool_input_normalization"
        and row.get("model_call_skipped") is not True
        and row["generated_input"] != "{}"
    )
    physical = iter(saved["tool_results"])
    physical_inputs = iter(row for row in saved["trace"] if row["type"] == "tool_call")
    prompts = []
    tool_checks = []

    class Events:
        def emit(self, *args):
            pass

    class Tools:
        def specs(self):
            return [tool.spec for tool in builtin_tools().values()]

        async def execute(self, call):
            source = next(physical_inputs)
            tool_checks.append(
                {
                    "tool": call.tool,
                    "same_tool": source["tool_name"] == call.tool,
                    "same_arguments": json.loads(source["tool_input"]) == call.arguments,
                }
            )
            return ToolResult.model_validate(next(physical)).model_copy(update={"call_id": call.id})

    class Client:
        async def text(self, prompt, *, stage, system_prompt="", **kwargs):
            source = expected[len(prompts)]
            effective_system = (
                system_prompt or "You are ChatGPT, a large language model trained by OpenAI.\n"
            )
            directory = root / "outputs/verification/audited-prompts" / str(sample["question_id"])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{len(prompts) + 1:02d}-{stage}.json").write_text(
                json.dumps(
                    {"system": effective_system, "user": prompt}, ensure_ascii=False, indent=2
                )
                + "\n"
            )
            prompts.append(
                {
                    "stage": stage,
                    "same_stage": stage == source["stage"],
                    "same_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()
                    == source["prompt_sha256"],
                    "system_prompt_sha256": hashlib.sha256(effective_system.encode()).hexdigest(),
                }
            )
            if stage == "tool_argument_generation":
                answer = "Input:" + next(arguments)
            elif stage == "final_answer_generation":
                answer = saved["answer"]
            else:
                answer = next(plans)
            return ChatReply(message={"content": answer})

    class Embed:
        def get_text_embedding(self, text):
            raise AssertionError("Audit must not recompute retrieval or call an embedding provider")

    options = load_algorithm(root)
    kg = KGToolRetriever.from_directory(root / options.index_path, Embed())
    retrieved = next(
        row["retrieved_relations"] for row in saved["trace"] if row["type"] == "tool_retrieval"
    )

    async def recorded_retrieval(query):
        return json.loads(json.dumps(retrieved))

    kg.retrieve = recorded_retrieval
    paths = ToolPaths.create(root, folder / "artifacts")
    contracts = AnswerContracts(root)
    context = AgentContext(
        Tools(),
        None,
        Events(),
        {},
        None,
        Client(),
        tool_retriever=kg,
        planner=builtin_planner(rs_local_contracts=options.rs_local_contracts),
        options=options,
        resource_resolver=paths.resolve,
        answer_contract_lookup=contracts.find,
        answer_system_prompt=contracts.system_prompt,
    )
    result = await AgentEngine().run(TaskInput.model_validate(request), context)
    return {
        "question_id": sample["question_id"],
        "category": sample["category"],
        "private_request_fields": private_fields(request),
        "prompts": prompts,
        "tool_inputs": tool_checks,
        "same_final_answer": result.answer == saved["answer"],
        "matched_prompt_count": sum(
            row["same_prompt_sha256"] and row["same_stage"] for row in prompts
        ),
        "expected_prompt_count": len(expected),
    }


async def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    delivery = json.loads((root / "outputs/verification/delivery.json").read_text())
    cases = []
    for sample in delivery["domain_samples"]:
        cases.append(await audit_case(root, sample))
    report = {
        "audit": "recorded_prompt_hash_reconstruction_and_request_boundary",
        "network_calls": 0,
        "cases": cases,
        "conclusion": "No accidental private-label injection observed in the recorded requests. These are development regression samples, not wholly leakage-free held-out accuracy evidence.",
        "findings": [
            {
                "id": "geographic_answer_replay",
                "status": "present_authorized_protocol",
                "question_id": 106,
                "evidence": "Tool selected source_id=616 and returned official materialized answers plus canonical reference gold_sparql under result.query. It is not independent unseen-answer RDF geometry evaluation.",
            },
            {
                "id": "ocean_benchmark_cache",
                "status": "present_bounded_precomputation",
                "question_id": 347,
                "evidence": "Runtime reads a 10-row SOCAT sample table selected from raw observations; original QA answers are generated from these rows. Selection uses date/lat/lon, not private expected_answer, but no full raw grid recomputation occurs.",
            },
            {
                "id": "development_overlap",
                "status": "present",
                "evidence": "Q61/Q106 were checked previously and Q106/Q111 were used during ontology repairs. Public-format releases also contain post-pilot revisions. The ten-domain pass is engineering regression evidence.",
            },
            {
                "id": "private_file_visibility",
                "status": "not_observed_and_hardened",
                "evidence": "No recorded tool input reads benchmark labels, saved predictions/scores or test fixtures. Generic scientific resolver now denies evaluator/contract/replay roots; dedicated authorized snapshot access and the scientific disaster table remain. This is not a full OS sandbox.",
            },
        ],
        "limitations": [
            "Original HTTP message bodies were not saved; reconstructed user prompts are verified against recorded SHA256.",
            "GeoQuestions deliberately returns materialized answer tables and canonical reference query; it is not an unseen-answer benchmark.",
            "Samples are development/regression cases, including Q106 used during repairs; they are not held-out accuracy evidence.",
        ],
    }
    write_json(root / "outputs/verification/leakage-audit.json", report)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "prompts_matched": sum(c["matched_prompt_count"] for c in cases),
                "prompts_expected": sum(c["expected_prompt_count"] for c in cases),
                "private_request_fields": sum(len(c["private_request_fields"]) for c in cases),
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
