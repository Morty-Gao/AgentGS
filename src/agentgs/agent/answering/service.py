"""Required final model generation and non-mutating, versioned evidence checks."""

import asyncio
import json

from agentgs.agent.answering.evidence.legacy_validation import validate_final_answer
from agentgs.agent.answering.formats import check_fixed_answer
from agentgs.agent.answering.validation import validate_returned_evidence
from agentgs.agent.planning.parsing import extract_answer_parts
from agentgs.contracts.async_utils import invoke


class FinalAnswerGenerationError(RuntimeError):
    pass


class AnsweringService:
    def __init__(self, state, stages):
        self.state, self.stages = state, stages

    async def generate(
        self, question, output, *, completed=True, tool_attempts_exhausted=False, answer_text=None
    ):
        state = self.state
        completed = completed and not tool_attempts_exhausted
        state.tool_chain_completed = completed
        route = (
            "tool_attempts_exhausted"
            if tool_attempts_exhausted
            else "tool_chain_completed"
            if completed
            else "task_incomplete"
        )
        evidence = output
        if not completed:
            evidence = json.dumps(
                {
                    "execution_status": "Not Completed",
                    "successful_tool_results": [
                        {
                            k: event.get(k)
                            for k in ("tool_name", "attempt", "tool_input", "tool_output")
                        }
                        for event in state.execution_events()
                        if event.get("type") == "tool_call" and event.get("status") == "success"
                    ],
                    "failed_tool_attempts": [
                        {
                            k: event.get(k)
                            for k in ("tool_name", "attempt", "tool_input", "tool_output")
                        }
                        for event in state.execution_events()
                        if event.get("type") in {"tool_call", "tool_retry"}
                        and event.get("status") != "success"
                    ],
                    "execution_context": str(output),
                },
                ensure_ascii=False,
                default=str,
            )
        budget_kind = (
            "bash"
            if state.action_budget_exhausted
            else "tool"
            if tool_attempts_exhausted
            else None
        )
        try:
            contract = self.stages.find_contract(question)
            text = (
                answer_text
                if answer_text is not None
                else await self.stages.generate_output(
                    question,
                    evidence,
                    execution_incomplete=not completed,
                    budget_exhausted=budget_kind,
                )
            )
            status, answer = (
                ("Completed" if completed else "Not Completed", text.strip())
                if contract
                else extract_answer_parts(text)
            )
            if status not in {"Completed", "Not Completed"} or not str(answer or "").strip():
                raise ValueError("The final-answer model returned an empty or malformed answer")
            outputs = [
                event.get("tool_output", "")
                for event in state.execution_events()
                if event.get("type") == "tool_call" and event.get("status") == "success"
            ]
            suggested, diagnostic = await invoke(
                validate_final_answer, question, answer, outputs, completed=completed
            )
            answer, event = await invoke(
                validate_returned_evidence,
                question,
                answer,
                outputs,
                completed=completed,
                legacy=diagnostic,
                suggested=suggested,
            )
            event.pop("evaluation_llm_required", None)
            state.answer_evidence_status = "consistent" if event["valid"] else "inconsistent"
            event["allow_rule_planning"] = state.context.options.allow_rule_planning
            event["answer_llm_required"] = True
            if contract:
                format_event = await invoke(check_fixed_answer, answer, contract)
                format_event["diagnostic_only"] = True
                await state.record(format_event)
                event.update(
                    diagnostic_only=True,
                    completion_policy="execution_and_answer_presence",
                )
                if format_event["has_unknown"]:
                    status = "Not Completed"
            elif not event["valid"]:
                status = "Not Completed"
            if not completed:
                status = "Not Completed"
            if state.context.options.planning_mode == "data_only" and not event["valid"]:
                status = "Not Completed"
            if state.context.options.planning_mode == "data_only":
                state.tool_chain_completed = status == "Completed"
            await state.record(event)
            state.answer, state.completion_status, state.answer_generation_status = (
                answer,
                status,
                "success",
            )
            await state.emit(
                "final_answer_generation",
                {
                    "status": status,
                    "content": answer,
                    "answer_route": route,
                    "tool_chain_completed": completed,
                    "answer_generation_status": "success",
                    "strategy": "model_with_evidence_validation",
                    "model_call_skipped": answer_text is not None,
                },
            )
            return state.result()
        except asyncio.CancelledError:
            state.answer_generation_status = "cancelled"
            await state.emit(
                "final_answer_generation", {"status": "cancelled", "answer_route": route}
            )
            raise
        except Exception as exc:
            state.answer_generation_status = "failed"
            await state.emit(
                "final_answer_generation",
                {
                    "status": "failed",
                    "answer_route": route,
                    "tool_chain_completed": completed,
                    "answer_generation_status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
            status_code = getattr(exc, "status_code", None)
            message = f"LLM final-answer generation failed: {type(exc).__name__}"
            if status_code is not None:
                message += f" (HTTP {status_code})"
            failure = FinalAnswerGenerationError(message)
            failure.status_code = status_code
            raise failure from exc
