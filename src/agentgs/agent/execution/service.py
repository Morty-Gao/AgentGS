"""Native tool calls, declared bindings and the original atomic repair policy."""

import json

from agentgs.agent.execution import references, scope
from agentgs.agent.execution.bindings import prepare_tool_input
from agentgs.agent.execution.feedback import PreparedToolCall, run_tool_with_feedback
from agentgs.agent.execution.scientific import (
    overlay_scientific_tool_schema,
    preserve_scientific_arguments,
    requires_scientific_argument_generation,
    scientific_input_issues,
)
from agentgs.contracts.arguments import normalize_arguments
from agentgs.contracts.async_utils import invoke
from agentgs.contracts.catalog import get_tool_spec
from agentgs.contracts.models import ToolCall


def object_arguments(text):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Function arguments must be a JSON object")
    return value


class ExecutionService:
    def __init__(self, state, stages):
        self.state, self.stages = state, stages
        self.specs = {spec.id: spec for spec in state.context.tools.specs()}

    def split_arguments(self, name, text):
        """Keep program-injected annotations out of the public function schema."""
        params = normalize_arguments(object_arguments(text))
        properties = self.specs[name].input_schema.get("properties")
        public = {
            key: value for key, value in params.items() if properties is None or key in properties
        }
        internal = {
            key: value
            for key, value in params.items()
            if key == "_output_keys" or key.startswith("_rs_")
        }
        return public, internal

    async def physical(self, name, text, files, *, call_id=None):
        if call_id is None:
            pending = set(self.state.conversation.function_calls()) - {
                m.tool_call_id for m in self.state.conversation.messages if m.role == "tool"
            }
            matching = [
                key
                for key in pending
                if self.state.conversation.function_calls()[key].function.name == name
            ]
            if len(matching) != 1:
                raise ValueError("Tool execution needs exactly one pending invocation")
            call_id = matching[0]
        arguments, internal = self.split_arguments(name, text)
        call = ToolCall(id=call_id, tool=name, arguments=arguments)
        call._runtime_arguments = internal
        result = await self.state.context.tools.execute(call)
        self.state.results.append(result)
        await self.state.save()
        if result.raw_output is not None:
            return result.raw_output
        if result.succeeded:
            return json.dumps(result.data, ensure_ascii=False)
        return json.dumps(
            {"status": "error", "error_type": result.error.code, "error": result.error.message},
            ensure_ascii=False,
        )

    async def comparison(self, name, text, files):
        arguments, internal = self.split_arguments(name, text)
        call = ToolCall(tool=name, arguments=arguments)
        call._runtime_arguments = internal
        result = await self.state.context.tools.execute(call)
        if result.raw_output is not None:
            return result.raw_output
        return json.dumps(
            result.data if result.succeeded else {"status": "error", "error": result.error.message},
            ensure_ascii=False,
        )

    async def run(
        self,
        name,
        question,
        registry,
        total_output,
        current_output="",
        execution_context=None,
        *,
        initial_arguments=None,
    ):
        row = await invoke(get_tool_spec, name)
        if row is None or name not in self.specs:
            raise ValueError(f"Tool is outside the registered scope: {name}")
        info = await invoke(
            overlay_scientific_tool_schema,
            name,
            {"input_format": row["Input"], "output_format": row["Output"]},
        )
        scientific = requires_scientific_argument_generation(name)
        bridge = bool((execution_context or {}).get("shared_bridge"))
        if bridge:
            if (
                self.state.context.options.planning_mode != "skill_guided"
                or not self.state.context.options.shared_bridge
            ):
                raise ValueError("Shared bridge inputs require enabled skill-guided planning")
            from agentgs.agent.planning.bridge import apply_shared_bridge_inputs

        async def input_provider(attempt, history):
            # Declared bindings remain deterministic, but every question-mode
            # execution originates from a real native model call.
            seed = json.dumps(initial_arguments or {}, ensure_ascii=False)
            candidate, _ = await invoke(
                prepare_tool_input,
                name,
                seed,
                question,
                current_output,
                total_output,
                registry,
                execution_context,
            )
            observed = dict(registry.compact_context() if registry else {})
            observed["declared_binding_candidate"] = object_arguments(candidate)
            feedback = history[-1].to_prompt_context() if history else None
            # There is no fallback to parsing model prose or manufacturing arguments.
            message = await self.stages.generate_tool_call(
                name,
                question,
                observed,
                {k: v for k, v in (execution_context or {}).items() if k != "shared_bridge"},
                references.argument_contract(info["input_format"], registry, tool_name=name),
                info["output_format"],
                feedback,
                [row.to_prompt_context() for row in history],
            )
            calls = message.tool_calls
            error = None
            generated = None
            try:
                if len(calls) != 1 or calls[0].function.name != name:
                    raise ValueError(f"Return exactly one native function call for {name}")
                generated = object_arguments(calls[0].function.arguments)
                properties = self.specs[name].input_schema.get("properties")
                if properties is not None and set(generated) - set(properties):
                    raise ValueError(
                        "Unknown function arguments: "
                        + ", ".join(sorted(set(generated) - set(properties)))
                    )
            except (ValueError, TypeError) as exc:
                error = json.dumps(
                    {"status": "error", "error_type": "invalid_input", "error": str(exc)},
                    ensure_ascii=False,
                )
            # Answer every original call, even if a provider ignores tool_choice or
            # parallel_tool_calls=false. Unexpected batches never execute.
            for raw in calls[1:]:
                self.state.conversation.finish_tool(
                    raw.id, error, status="failed", argument_source="native_function_call"
                )
            if error:
                await self.state.record(
                    {
                        "type": "tool_call_protocol_error",
                        "tool_name": name,
                        "attempt": attempt,
                        "error": error,
                    }
                )
                return PreparedToolCall(
                    calls[0].id if calls else None,
                    calls[0].function.arguments if calls else "{}",
                    error,
                )
            raw = calls[0]
            try:
                if initial_arguments:
                    generated = {**generated, **initial_arguments}
                generated_text = json.dumps(generated, ensure_ascii=False)
                reference_binding = (
                    await invoke(references.bind_parameters, name, generated, registry)
                    if scope.active()
                    else None
                )
                if reference_binding is not None:
                    bound, event = reference_binding
                    if bridge:
                        diagnostic = {
                            "code": "UNSUPPORTED_REFERENCE_CONTEXT",
                            "detail": "Explicit references with shared bridge projections are unsupported",
                        }
                        bound = {references.DIAGNOSTICS: [diagnostic]}
                        event["diagnostics"] = [diagnostic]
                    event["attempt"] = attempt
                    await self.state.record(event)
                    return PreparedToolCall(raw.id, json.dumps(bound, ensure_ascii=False))
                prepared, event = await invoke(
                    prepare_tool_input,
                    name,
                    generated_text,
                    question,
                    current_output,
                    total_output,
                    registry,
                    execution_context,
                )
                prepared = preserve_scientific_arguments(name, generated_text, prepared)
                if bridge:
                    prepared = apply_shared_bridge_inputs(generated_text, execution_context)
                    event["shared_bridge_binding"] = True
                if scientific:
                    event["scientific_argument_contract"] = {
                        "source": "native_function_arguments",
                        "issues": scientific_input_issues(name, prepared),
                    }
                event.update(
                    attempt=attempt, call_id=raw.id, argument_source="native_function_call"
                )
                await self.state.record(event)
                return PreparedToolCall(raw.id, prepared)
            except Exception as exc:
                error = json.dumps(
                    {
                        "status": "error",
                        "error_type": "argument_binding_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                )
                return PreparedToolCall(raw.id, raw.function.arguments, error)

        return await run_tool_with_feedback(
            tool_name=name,
            file_path_list=self.state.task.file_path_list,
            input_provider=input_provider,
            tool_runner=self.physical,
            input_format=info["input_format"],
            max_attempts=self.state.context.options.tool_attempts,
            messages=self.state.conversation,
            checkpoint=self.state.save,
            producer_step=f"tool_step_{execution_context['execution_index']:02d}"
            if execution_context and isinstance(execution_context.get("execution_index"), int)
            else "",
        )
