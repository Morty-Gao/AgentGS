"""Staged GeoToolAgent orchestration with native model function calls."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

from agentgs.agent.answering.service import AnsweringService, FinalAnswerGenerationError
from agentgs.agent.execution import scope
from agentgs.agent.execution.artifacts import ArtifactRegistry
from agentgs.agent.execution.scientific import overlay_scientific_tool_schema
from agentgs.agent.execution.service import ExecutionService
from agentgs.agent.planning.bridge import (
    complete_shared_capability_bridges,
    ground_or_reject_shared_bridge_chain,
    record_shared_bridge_execution,
)
from agentgs.agent.planning.bridge_comparison import run_shared_bridge_comparisons
from agentgs.agent.planning.geosparql import _heuristic_geosparql_query
from agentgs.agent.planning.grounding import ground_plan
from agentgs.agent.planning.model_stages import ModelStages
from agentgs.agent.planning.routing.entry import infer_tools, normalize_plan
from agentgs.agent.planning.skills import builtin_planner
from agentgs.agent.retrieval.scored_skills import RESOURCE_MARKER
from agentgs.agent.state import AgentExecutionError, RunState
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.contracts.catalog import load_catalog
from agentgs.contracts.remote_sensing import contract_scope
from agentgs.contracts.resources import resource_access


class MutablePlan(SimpleNamespace):
    def as_dict(self):
        return deepcopy({k: v for k, v in self.__dict__.items() if not k.startswith("_")})


def available_tool_names(relations, allowed):
    return list(
        dict.fromkeys(
            row[0]
            for row in relations or []
            if len(row) >= 2 and row[1] == "has the functionality that" and row[0] in allowed
        )
    )


def grounding_schemas(retriever, names):
    schemas = {}
    for name in dict.fromkeys(names):
        try:
            info = retriever.get_tool_info(retriever.get_tool_subgraph(name, depth=1, limit=1000))
            if isinstance(info, dict):
                schemas[name] = dict(info)
        except Exception:
            schemas[name] = {}
    return {name: overlay_scientific_tool_schema(name, info) for name, info in schemas.items()}


def snapshot_relations(relations):
    """Expose the authorized Geo protocol while retaining frozen KG ranking."""
    row = next(
        row for row in load_catalog()["specs"] if row["Name"] == "GeoBigDataExecuteGeoSPARQL"
    )
    replacements = {
        "has the functionality that": row["Function"],
        "inputs": row["Input"],
        "outputs": row["Output"],
    }
    return [
        [relation[0], relation[1], replacements[relation[1]]]
        if len(relation) >= 3 and relation[0] == row["Name"] and relation[1] in replacements
        else relation
        for relation in relations
    ]


class AgentEngine:
    async def run(self, task, context):
        context.options = context.options or AlgorithmSettings()
        if task.planning_mode is not None:
            context.options = context.options.model_copy(
                update={"planning_mode": task.planning_mode}
            )
        if task.require_llm is not None:
            context.options = context.options.model_copy(update={"require_llm": task.require_llm})
        state = RunState(context, task)
        # Explicit structured requests use the same tool schemas without a model.
        if task.calls:
            state.emit(
                "plan.created",
                {"source": "explicit_calls", "tools": [call.tool for call in task.calls]},
            )
            for call in task.calls:
                result = await state.execute(call)
                state.save()
                if not result.succeeded:
                    raise AgentExecutionError(
                        f"{result.error.code}: {result.error.message}", state.result()
                    )
            state.answer = json.dumps([result.data for result in state.results], ensure_ascii=False)
            state.tool_chain_completed = True
            state.completion_status = "Completed"
            state.answer_generation_status = "not_requested"
            state.save()
            return state.result()
        if not task.question.strip() or context.client is None:
            raise ValueError("Question mode requires a question and configured model client")
        question = task.question
        if task.input_resource is not None and RESOURCE_MARKER not in question:
            question += (
                "\n\n" + RESOURCE_MARKER + " " + json.dumps(task.input_resource, ensure_ascii=False)
            )
        stages = ModelStages(
            state,
            lambda q: (
                task.output_contract
                or (context.answer_contract_lookup(q) if context.answer_contract_lookup else None)
            ),
            context.answer_system_prompt,
        )
        execution = ExecutionService(state, stages)
        answering = AnsweringService(state, stages)
        permission = scope._permitted.set(context.options.rs_explicit_references)
        from agentgs.agent.planning.bridge import _bridge_enabled

        bridge_token = _bridge_enabled.set(context.options.shared_bridge)
        try:
            with (
                contract_scope(context.options.rs_local_contracts),
                resource_access(context.resource_resolver),
                scope.request_scope(task.input_domain),
            ):
                return await self._staged(question, state, stages, execution, answering)
        except Exception as exc:
            if isinstance(exc, AgentExecutionError):
                raise
            state.save()
            raise AgentExecutionError(
                str(exc), state.result(), getattr(exc, "status_code", None)
            ) from exc
        finally:
            scope._permitted.reset(permission)
            _bridge_enabled.reset(bridge_token)

    async def _staged(self, question, state, stages, execution, answering):
        context, task, options = state.context, state.task, state.context.options
        names = [spec.id for spec in context.tools.specs()]
        allowed = set(names)
        planner = context.planner or builtin_planner(rs_local_contracts=options.rs_local_contracts)
        registry = ArtifactRegistry(question)
        scope.set_registry(registry)
        inferred = [name for name in infer_tools(question) if name in allowed]
        skill_guided = options.planning_mode == "skill_guided"
        plan = None
        bridge_report = None
        # Authorized offline GeoQuestions route retains the original early-route position.
        if "GeoBigDataExecuteGeoSPARQL" in inferred and not options.require_llm:
            if skill_guided:
                plan, bridge_report = self._skill_layer(
                    question, ["GeoBigDataExecuteGeoSPARQL"], names, planner, state
                )
                state.emit(
                    "tool_grounding",
                    {
                        "content": ground_plan(
                            planner.graph,
                            plan.as_dict(),
                            names,
                            ["GeoBigDataExecuteGeoSPARQL"],
                            tool_next=load_catalog()["next"],
                        )
                    },
                )
            state.emit(
                "schema_route",
                {
                    "source": "compact_geosparql_query_planner",
                    "selected_tools": ["GeoBigDataExecuteGeoSPARQL"],
                },
            )
            seed = None
            if not options.geosparql_llm:
                seed = {"query": _heuristic_geosparql_query(question), "dataset": "YAGO2geo_original_RDF"}
            state.emit("plan.created", {"source": "schema_route", "tools": ["GeoBigDataExecuteGeoSPARQL"]})
            result = await execution.run("GeoBigDataExecuteGeoSPARQL", question, registry, "", initial_arguments=seed)
            return await answering.generate(
                question,
                result.render_for_agent(),
                completed=result.succeeded,
                tool_attempts_exhausted=not result.succeeded,
            )
        if context.tool_retriever is None:
            raise FileNotFoundError(
                "Frozen tool KG and embedding backend must be prepared for the staged agent"
            )
        total_output = ""
        resume_prefix = []
        last_output = ""
        attempts = 1 if options.require_llm else 3
        previous_bridge = None
        for attempt in range(attempts):
            try:
                if skill_guided:
                    if plan is None:
                        plan, bridge_report = self._skill_layer(question, [], names, planner, state)
                    state.emit("artifact_registry_init", {"content": registry.compact_context()})
                retrieved = await context.tool_retriever.retrieve(
                    SimpleNamespace(query_str=question)
                )
                retrieved = snapshot_relations(retrieved)
                available = available_tool_names(retrieved, allowed)
                state.emit(
                    "tool_retrieval",
                    {
                        "query": question,
                        "retrieved_tools": available,
                        "retrieved_relations": retrieved,
                        "similarity_top_k": 5,
                        "graph_store_query_depth": 3,
                        "max_tools": 20,
                    },
                )
                if skill_guided:
                    candidates = planner.graph.candidates(
                        [row["capability_id"] for row in plan.capabilities], names, inferred
                    )
                    expanded = list(
                        dict.fromkeys(
                            row["tool_name"]
                            for rows in candidates.values()
                            for row in rows
                            if row["tool_name"] in allowed
                        )
                    )
                    expanded += inferred
                    additions = []
                    for name in expanded:
                        if not any(row[0] == name for row in retrieved):
                            additions += context.tool_retriever.get_tool_subgraph(
                                name, depth=1, limit=1000
                            )
                    retrieved = list(retrieved) + snapshot_relations(additions)
                    available = list(dict.fromkeys(available + expanded))
                    state.emit(
                        "tool_graph_retrieval",
                        {"capability_candidates": candidates, "available_tools": available},
                    )
                schema_route = skill_guided and not options.require_llm and bool(inferred)
                if schema_route:
                    proposed = list(inferred)
                    state.emit(
                        "schema_route",
                        {
                            "source": "structured_resource_and_tool_schema",
                            "selected_tools": proposed,
                        },
                    )
                else:
                    try:
                        proposed = await stages.generate_tools_plan(
                            question,
                            retrieved,
                            total_output,
                            {
                                "skill_plan": plan.as_dict() if plan else {},
                                "schema_routing_hints": inferred,
                                "hint_policy": "Hints identify compatible capabilities; instantiate repeated calls from distinct entity/temporal cohorts.",
                            },
                        )
                        state.emit("plan", {"source": "native_function_call", "tools": proposed})
                    except Exception as exc:
                        if options.require_llm:
                            raise
                        proposed = list(
                            dict.fromkeys(name for name in inferred if name in available)
                        )
                        if not proposed:
                            raise
                        state.emit(
                            "plan_model_fallback",
                            {
                                "reason": type(exc).__name__,
                                "validated_tools": proposed,
                                "source": "resource_inferred_tools",
                            },
                        )
                if options.require_llm:
                    chain = [name for name in proposed if name in set(available)]
                    if not chain:
                        raise ValueError(
                            "Required LLM planning returned no registered executable tool"
                        )
                    plan_source = "actual_llm_plan_registered_filter_only"
                else:
                    chain, plan_source = normalize_plan(proposed, inferred, available)
                chain = [name for name in chain if name in set(available)]
                if not chain and available and not options.require_llm:
                    chain = available[:1]
                state.emit(
                    "plan_validation",
                    {
                        "source": plan_source,
                        "model_tools": proposed,
                        "resource_inferred_tools": inferred,
                        "validated_tools": chain,
                    },
                )
                if not chain:
                    continue
                bridge_state = None
                if plan is not None and bridge_report and bridge_report.get("accepted"):
                    chain, bridge_state, rejection = ground_or_reject_shared_bridge_chain(
                        chain, plan, bridge_report
                    )
                    if rejection:
                        state.emit("shared_bridge_grounding_rejected", rejection)
                    if (
                        bridge_state
                        and previous_bridge
                        and bridge_state["tool_chain"] == previous_bridge["tool_chain"]
                    ):
                        for key in (
                            "execution_records",
                            "prefix_artifacts",
                            "comparisons",
                            "run_id",
                        ):
                            bridge_state[key] = previous_bridge[key]
                    previous_bridge = bridge_state
                if skill_guided:
                    schemas = grounding_schemas(context.tool_retriever, chain)
                    grounded = ground_plan(
                        planner.graph,
                        plan.as_dict(),
                        available,
                        inferred,
                        proposed_chain=chain,
                        tool_schemas=schemas,
                        tool_next=load_catalog()["next"],
                    )
                    state.emit("tool_grounding", {"content": grounded})
                    chain = grounded["tool_chain"]
                if not chain:
                    raise RuntimeError("Skill grounding produced no available tool instances")
                state.emit("plan.created", {"source": plan_source, "tools": chain})
                skip = 0
                while (
                    skip < len(resume_prefix)
                    and skip < len(chain)
                    and resume_prefix[skip] == chain[skip]
                ):
                    skip += 1
                if skip:
                    state.emit(
                        "plan_resume",
                        {"skipped_successful_prefix": chain[:skip], "resume_index": skip + 1},
                    )
                occurrences = {}
                current = ""
                failed = False
                for index, name in enumerate(chain, 1):
                    occurrence = occurrences.get(name, 0)
                    occurrences[name] = occurrence + 1
                    if index <= skip:
                        continue
                    step = {
                        "execution_index": index,
                        "occurrence_index": occurrence,
                        "occurrence_count": chain.count(name),
                        "tool_chain": chain,
                    }
                    if bridge_state and index in bridge_state["step_bindings"]:
                        step["shared_bridge"] = {"state": bridge_state, "execution_index": index}
                    result = await execution.run(
                        name, question, registry, total_output, current, step
                    )
                    last_output = result.render_for_agent()
                    total_output += last_output
                    current = result.output
                    registry.ingest_tool_output(name, f"tool_step_{index:02d}", result.output)
                    state.emit(
                        "artifact_update",
                        {
                            "tool_name": name,
                            "execution_index": index,
                            "content": registry.compact_context(max_records=12),
                        },
                    )
                    if bridge_state:
                        record_shared_bridge_execution(bridge_state, index, result, registry)
                    if not result.succeeded:
                        failed = True
                        resume_prefix = chain[: index - 1]
                        state.emit(
                            "execution_feedback", {"status": "replanning", "tool_name": name}
                        )
                        break
                if failed:
                    if attempt + 1 < attempts:
                        continue
                    return await answering.generate(
                        question, total_output, completed=False, tool_attempts_exhausted=True
                    )
                if bridge_state:
                    await run_shared_bridge_comparisons(
                        bridge_state,
                        tool_runner=execution.comparison,
                        file_path_list=task.file_path_list,
                        messages=state.conversation,
                    )
                # Successful chains are never rerun because final generation or validation failed.
                return await answering.generate(question, total_output)
            except FinalAnswerGenerationError:
                raise
            except Exception as exc:
                if getattr(exc, "status_code", None) in (401, 403):
                    raise
                state.emit("plan_retry", {"attempt": attempt + 1, "error_type": type(exc).__name__})
                if attempt + 1 < attempts:
                    await asyncio.sleep(1)
        return await answering.generate(
            question,
            total_output or last_output or "Execution ended without a supported result.",
            completed=False,
        )

    def _record_skill(self, state, plan):
        state.emit(
            "skill_retrieval", {"query": plan.query, "top_k": 5, "candidates": plan.candidates}
        )
        state.emit("skill_plan", {"content": plan.as_dict()})
        state.emit(
            "capability_expansion",
            {
                "content": plan.capability_expansion,
                "capability_count": len(plan.capabilities),
                "missing_capabilities": plan.missing_capabilities,
            },
        )
        state.emit("geo_constraint_check", {"constraints": plan.constraints})

    def _skill_layer(self, question, inferred, names, planner, state):
        plan = MutablePlan(**planner.plan(question, inferred))
        report = complete_shared_capability_bridges(plan, list(planner.skills.values()), names)
        if report["enabled"]:
            state.emit("shared_capability_bridge_search", {"content": report})
        self._record_skill(state, plan)
        return plan, report
