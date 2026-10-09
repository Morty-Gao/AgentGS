"""Staged GeoToolAgent orchestration with native model function calls."""

from __future__ import annotations

import asyncio
import json
from copy import copy, deepcopy
from types import SimpleNamespace

from agentgs.agent.answering.service import AnsweringService, FinalAnswerGenerationError
from agentgs.agent.data_only import explore
from agentgs.agent.execution import scope
from agentgs.agent.execution.artifacts import ArtifactRegistry
from agentgs.agent.execution.scientific import overlay_scientific_tool_schema
from agentgs.agent.execution.service import ExecutionService
from agentgs.agent.planning.geosparql import _heuristic_geosparql_query
from agentgs.agent.planning.instances import ExecutionPlan, PlanStep, resolve_inputs, reuse_key
from agentgs.agent.planning.model_stages import ModelStages
from agentgs.agent.planning.routing.entry import infer_tools, normalize_plan
from agentgs.agent.state import AgentExecutionError, ModelProtocolError, RunState
from agentgs.client.errors import BudgetExceeded
from agentgs.config.algorithm import AlgorithmSettings, resolve_algorithm
from agentgs.contracts.async_utils import drain, invoke
from agentgs.contracts.catalog import load_catalog
from agentgs.contracts.remote_sensing import contract_scope
from agentgs.contracts.resources import RESOURCE_MARKER, resource_access


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
        context = copy(context)
        context.options = resolve_algorithm(context.options or AlgorithmSettings(), task)
        state = RunState(context, task)
        try:
            return await self._run_task(task, context, state)
        except asyncio.CancelledError:
            await drain(asyncio.create_task(state.cancel()))
            raise

    async def _run_task(self, task, context, state):
        # Explicit structured requests use the same tool schemas without a model.
        if task.calls:
            await state.emit(
                "plan.created",
                {"source": "explicit_calls", "tools": [call.tool for call in task.calls]},
            )
            if task.parallel_tools or any(call.depends_on for call in task.calls):
                results = await state.execute_many(task.calls)
                failed = next((result for result in results if not result.succeeded), None)
                if failed:
                    raise AgentExecutionError(
                        f"{failed.error.code}: {failed.error.message}", state.result()
                    )
            else:
                for call in task.calls:
                    result = await state.execute(call)
                    await state.save()
                    if not result.succeeded:
                        raise AgentExecutionError(
                            f"{result.error.code}: {result.error.message}", state.result()
                        )
            state.answer = json.dumps([result.data for result in state.results], ensure_ascii=False)
            state.tool_chain_completed = True
            state.completion_status = "Completed"
            state.answer_generation_status = "not_requested"
            await state.save()
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
            lambda q: task.output_contract,
        )
        execution = ExecutionService(state, stages)
        answering = AnsweringService(state, stages)
        permission = scope._permitted.set(context.options.rs_explicit_references)
        bridge_token = None
        if context.options.planning_mode == "skill_guided":
            from agentgs.agent.planning.bridge import _bridge_enabled

            bridge_token = _bridge_enabled.set(context.options.shared_bridge)
        try:
            with (
                contract_scope(context.options.rs_local_contracts),
                resource_access(context.resource_resolver),
                scope.request_scope(task.input_domain),
            ):
                if context.options.planning_mode == "data_only":
                    return await explore(question, state, stages, answering)
                return await self._staged(question, state, stages, execution, answering)
        except Exception as exc:
            if isinstance(exc, AgentExecutionError):
                raise
            await state.save()
            raise AgentExecutionError(
                str(exc), state.result(), getattr(exc, "status_code", None)
            ) from exc
        finally:
            scope._permitted.reset(permission)
            if bridge_token is not None:
                _bridge_enabled.reset(bridge_token)

    async def _staged(self, question, state, stages, execution, answering):
        context, task, options = state.context, state.task, state.context.options
        names = [spec.id for spec in context.tools.specs()]
        allowed = set(names)
        skill_guided = options.planning_mode == "skill_guided"
        planner = None
        if skill_guided:
            from agentgs.agent.planning.bridge import (
                ground_or_reject_shared_bridge_chain,
                record_shared_bridge_execution,
            )
            from agentgs.agent.planning.bridge_comparison import run_shared_bridge_comparisons
            from agentgs.agent.planning.grounding import ground_plan
            from agentgs.agent.planning.skills import builtin_planner

            planner = context.planner or await invoke(
                builtin_planner, rs_local_contracts=options.rs_local_contracts,
                predecessor_expansion=options.predecessor_expansion, capability_completion=options.capability_completion
            )
            planner = copy(planner)
            planner.predecessor_expansion = options.modules.predecessor_expansion
            planner.capability_completion = options.modules.capability_completion
            await state.emit("module_policy", {"effective": options.modules.model_dump(), **options.module_resolution})
        registry = await invoke(ArtifactRegistry, question)
        scope.set_registry(registry)
        inferred = [name for name in infer_tools(question) if name in allowed] if options.routing_hints else []
        plan = None
        bridge_report = None
        # Authorized offline GeoQuestions route retains the original early-route position.
        if "GeoBigDataExecuteGeoSPARQL" in inferred and options.allow_rule_planning:
            if skill_guided:
                plan, bridge_report = await self._skill_layer(
                    question, ["GeoBigDataExecuteGeoSPARQL"], names, planner, state
                )
                await state.emit(
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
            await state.emit(
                "schema_route",
                {
                    "source": "compact_geosparql_query_planner",
                    "selected_tools": ["GeoBigDataExecuteGeoSPARQL"],
                },
            )
            seed = None
            if not options.geosparql_llm:
                seed = {
                    "query": _heuristic_geosparql_query(question),
                    "dataset": "YAGO2geo_original_RDF",
                }
            await state.emit(
                "plan.created", {"source": "schema_route", "tools": ["GeoBigDataExecuteGeoSPARQL"]}
            )
            result = await execution.run(
                "GeoBigDataExecuteGeoSPARQL", question, registry, "", initial_arguments=seed
            )
            return await answering.generate(
                question,
                result.render_for_agent(),
                completed=result.succeeded,
                tool_attempts_exhausted=not result.succeeded,
            )
        if context.tool_retriever is None and options.candidate_pool != "all_registered":
            raise FileNotFoundError(
                "Frozen tool KG and embedding backend must be prepared for the staged agent"
            )
        total_output = ""
        completed_instances = {}
        last_output = ""
        attempts = options.plan_attempts
        previous_bridge = None
        for attempt in range(attempts):
            execution_started = False
            try:
                if skill_guided:
                    if plan is None:
                        plan, bridge_report = await self._skill_layer(
                            question, [], names, planner, state
                        )
                    await state.emit(
                        "artifact_registry_init", {"content": registry.compact_context()}
                    )
                if options.candidate_pool == "all_registered":
                    retrieved = [[spec.id, "has the functionality that", spec.description]
                                 for spec in context.tools.specs()]
                    retrieved += [[spec.id, "inputs", spec.input_schema] for spec in context.tools.specs()]
                else:
                    retrieved = await context.tool_retriever.retrieve(SimpleNamespace(query_str=question))
                retrieved = await invoke(snapshot_relations, retrieved)
                available = available_tool_names(retrieved, allowed)
                await state.emit(
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
                if skill_guided and options.candidate_pool != "all_registered":
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
                            additions += await invoke(
                                context.tool_retriever.get_tool_subgraph, name, depth=1, limit=1000
                            )
                    retrieved = list(retrieved) + await invoke(snapshot_relations, additions)
                    available = list(dict.fromkeys(available + expanded))
                    await state.emit(
                        "tool_graph_retrieval",
                        {"capability_candidates": candidates, "available_tools": available},
                    )
                schema_route = skill_guided and options.allow_rule_planning and bool(inferred)
                if schema_route:
                    proposed = list(inferred)
                    await state.emit(
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
                            planning_mode=options.planning_mode,
                            skill_plan=plan.as_dict() if skill_guided else None,
                            tool_hints=inferred,
                        )
                        await state.emit(
                            "plan", {"source": "native_function_call", "tools": proposed}
                        )
                    except Exception as exc:
                        if not options.allow_rule_planning:
                            raise
                        proposed = list(
                            dict.fromkeys(name for name in inferred if name in available)
                        )
                        if not proposed:
                            raise
                        await state.emit(
                            "plan_model_fallback",
                            {
                                "reason": type(exc).__name__,
                                "validated_tools": proposed,
                                "source": "resource_inferred_tools",
                            },
                        )
                instances = proposed if isinstance(proposed, ExecutionPlan) else ExecutionPlan([
                    PlanStep(id=f"rule_{i}", tool=name, purpose=f"Execute {name}") for i, name in enumerate(proposed, 1)
                ])
                if not options.allow_rule_planning:
                    chain = [name for name in proposed if name in set(available)]
                    if not chain:
                        raise ValueError("Model planning returned no registered executable tool")
                    plan_source = "model_plan_registered_filter_only"
                else:
                    chain, plan_source = normalize_plan(proposed, inferred, available)
                chain = [name for name in chain if name in set(available)]
                if not chain and available and options.allow_rule_planning:
                    chain = available[:1]
                await state.emit(
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
                if skill_guided and bridge_report and bridge_report.get("accepted"):
                    chain, bridge_state, rejection = ground_or_reject_shared_bridge_chain(
                        chain, plan, bridge_report
                    )
                    if rejection:
                        await state.emit("shared_bridge_grounding_rejected", rejection)
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
                instances = instances.align(chain)
                if skill_guided:
                    if options.modules.tool_selection == "first_valid":
                        from agentgs.agent.planning.module_selection import select_first_valid

                        fixed = {
                            index: chain[index - 1]
                            for index in (bridge_state or {}).get("capability_indices", {}).values()
                            if 0 < index <= len(chain)
                        }
                        instances, decisions = select_first_valid(
                            planner.graph, plan.as_dict(), instances, available,
                            {spec.id: spec for spec in context.tools.specs()}, fixed_tools=fixed,
                        )
                        chain = list(instances)
                        await state.emit("tool_selection", {"policy": "first_valid", "instances": decisions})
                    schemas = await invoke(grounding_schemas, context.tool_retriever, chain) if context.tool_retriever else {}
                    grounded = ground_plan(
                        planner.graph,
                        plan.as_dict(),
                        available,
                        inferred,
                        proposed_chain=chain,
                        tool_schemas=schemas,
                        tool_next=load_catalog()["next"],
                        execution_plan=instances,
                        tool_selection="model",
                    )
                    await state.emit("tool_grounding", {"content": grounded})
                    if not grounded["validation"]["valid"]:
                        total_output += "\nGrounding rejected: " + json.dumps(grounded["validation"])
                        raise ValueError("Grounding constraints were not satisfied")
                    chain = grounded["tool_chain"]
                    instances = instances.align(chain)
                if not chain:
                    raise RuntimeError("Planning produced no available tool instances")
                await state.emit("plan.created", {"source": plan_source, "tools": chain})
                outputs, upstream_keys = {}, {}
                occurrences = {}
                current = ""
                failed = False
                for index, name in enumerate(chain, 1):
                    occurrence = occurrences.get(name, 0)
                    occurrences[name] = occurrence + 1
                    instance = instances.steps[index - 1]
                    explicit_inputs = resolve_inputs(instance, outputs)
                    cache_key = reuse_key(instance, upstream_keys, question)
                    if cache_key and cache_key in completed_instances:
                        cached = completed_instances[cache_key]
                        outputs[instance.id] = json.loads(cached.output)
                        upstream_keys[instance.id] = cache_key
                        current = cached.output
                        await state.emit("plan_resume", {"instance_id": instance.id, "reuse_key": cache_key,
                                                         "tool_name": name, "physical_execution": False})
                        continue
                    step = {
                        "execution_index": index,
                        "occurrence_index": occurrence,
                        "occurrence_count": chain.count(name),
                        "tool_chain": chain,
                        "instance": instance.model_dump(mode="json"),
                        "resolved_inputs": explicit_inputs,
                    }
                    if bridge_state and index in bridge_state["step_bindings"]:
                        step["shared_bridge"] = {"state": bridge_state, "execution_index": index}
                    execution_started = True
                    result = await execution.run(
                        name, question, registry, total_output, current, step,
                        initial_arguments=explicit_inputs,
                    )
                    if result.succeeded:
                        try:
                            decoded = json.loads(result.output)
                        except ValueError:
                            decoded = {"text": result.output}
                        outputs[instance.id] = decoded
                        def numeric_tree(value):
                            return isinstance(value, (int, float, bool)) or (isinstance(value, list) and all(numeric_tree(v) for v in value)) or (isinstance(value, dict) and all(numeric_tree(v) for v in value.values()))
                        if cache_key and numeric_tree(decoded) and json.loads(result.final_input) == explicit_inputs:
                            completed_instances[cache_key] = result
                            upstream_keys[instance.id] = cache_key
                    last_output = result.render_for_agent()
                    total_output += last_output
                    current = result.output
                    await invoke(
                        registry.ingest_tool_output, name, f"tool_step_{index:02d}", result.output
                    )
                    await state.emit(
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
                        await state.emit(
                            "execution_feedback", {"status": "replanning" if options.execution_recovery else "recovery_disabled", "tool_name": name}
                        )
                        break
                if failed:
                    if options.execution_recovery and attempt + 1 < attempts:
                        continue
                    return await answering.generate(
                        question, total_output, completed=False, tool_attempts_exhausted=True
                    )
                if bridge_state and options.bridge_diagnostics:
                    await run_shared_bridge_comparisons(
                        bridge_state,
                        tool_runner=execution.comparison,
                        file_path_list=task.file_path_list,
                        messages=state.conversation,
                    )
                # Successful chains are never rerun because final generation or validation failed.
                return await answering.generate(question, total_output)
            except (FinalAnswerGenerationError, ModelProtocolError, BudgetExceeded):
                raise
            except Exception as exc:
                if getattr(exc, "status_code", None) in (401, 403):
                    raise
                if not options.execution_recovery and execution_started:
                    await state.emit("execution_feedback", {"status": "recovery_disabled", "error_type": type(exc).__name__})
                    return await answering.generate(question, total_output, completed=False, tool_attempts_exhausted=True)
                await state.emit(
                    "plan_retry", {"attempt": attempt + 1, "error_type": type(exc).__name__}
                )
                if attempt + 1 < attempts:
                    await asyncio.sleep(1)
        return await answering.generate(
            question,
            total_output or last_output or "Execution ended without a supported result.",
            completed=False,
        )

    async def _record_skill(self, state, plan):
        await state.emit(
            "skill_retrieval", {"query": plan.query, "top_k": 5, "candidates": plan.candidates}
        )
        await state.emit("skill_plan", {"content": plan.as_dict()})
        await state.emit(
            "capability_expansion",
            {
                "content": plan.capability_expansion,
                "capability_count": len(plan.capabilities),
                "missing_capabilities": plan.missing_capabilities,
            },
        )
        await state.emit("geo_constraint_check", {"constraints": plan.constraints})

    async def _skill_layer(self, question, inferred, names, planner, state):
        from agentgs.agent.planning.bridge import complete_shared_capability_bridges

        plan = MutablePlan(**(await invoke(planner.plan, question, inferred)))
        report = await invoke(
            complete_shared_capability_bridges, plan, list(planner.skills.values()), names
        )
        if report["enabled"]:
            await state.emit("shared_capability_bridge_search", {"content": report})
        selected = list(dict.fromkeys(row["skill_id"] for row in plan.steps))
        selected += [sid for sid in plan.capability_expansion.get("skill_ids", []) if sid not in selected]
        plan.declarations = [deepcopy(planner.skills[sid]) for sid in selected]
        from agentgs.agent.planning.module_context import planning_context

        representation = planning_context(plan.as_dict(), state.context.options.modules.planning_structure)
        await state.emit("planning_representation", {
            "structure": representation["representation"],
            "declaration_sha256": representation["declaration_sha256"],
        })
        await self._record_skill(state, plan)
        return plan, report
