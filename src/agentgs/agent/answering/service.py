"""Required final model generation and non-mutating, versioned evidence checks."""

import json

from agentgs.agent.answering.evidence.legacy_validation import validate_final_answer
from agentgs.agent.answering.formats import check_fixed_answer
from agentgs.agent.answering.validation import validate_returned_evidence
from agentgs.agent.planning.parsing import extract_answer_parts


class FinalAnswerGenerationError(RuntimeError):
    pass


class AnsweringService:
    def __init__(self, state, stages):
        self.state, self.stages = state, stages

    async def generate(self, question, output, *, completed=True, tool_attempts_exhausted=False):
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
                        for event in state.trace
                        if event.get("type") == "tool_call" and event.get("status") == "success"
                    ],
                    "failed_tool_attempts": [
                        {
                            k: event.get(k)
                            for k in ("tool_name", "attempt", "tool_input", "tool_output")
                        }
                        for event in state.trace
                        if event.get("type") in {"tool_call", "tool_retry"}
                        and event.get("status") != "success"
                    ],
                    "execution_context": str(output),
                },
                ensure_ascii=False,
                default=str,
            )
        try:
            contract = self.stages.find_contract(question)
            text = await self.stages.generate_output(
                question, evidence, model=None, execution_incomplete=not completed
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
                for event in state.trace
                if event.get("type") == "tool_call" and event.get("status") == "success"
            ]
            suggested, diagnostic = validate_final_answer(
                question, answer, outputs, completed=completed
            )
            answer, event = validate_returned_evidence(
                question,
                answer,
                outputs,
                completed=completed,
                legacy=diagnostic,
                suggested=suggested,
            )
            event["evaluation_llm_required"] = state.context.options.require_llm
            event["answer_llm_required"] = True
            if contract:
                format_event = check_fixed_answer(answer, contract)
                state.trace.append(format_event)
                event.update(
                    diagnostic_only=True, completion_policy="geotooleval_fixed_format_v1.1"
                )
                if not format_event["valid"] or format_event["has_unknown"]:
                    status = "Not Completed"
            elif not event["valid"]:
                status = "Not Completed"
            if not completed:
                status = "Not Completed"
            state.trace.append(event)
            state.answer, state.completion_status, state.answer_generation_status = (
                answer,
                status,
                "success",
            )
            state.emit(
                "final_answer_generation",
                {
                    "status": status,
                    "content": answer,
                    "answer_route": route,
                    "tool_chain_completed": completed,
                    "answer_generation_status": "success",
                    "strategy": "model_with_evidence_validation",
                    "model_call_skipped": False,
                },
            )
            return state.result()
        except Exception as exc:
            state.answer_generation_status = "failed"
            state.emit(
                "final_answer_generation",
                {
                    "status": "failed",
                    "answer_route": route,
                    "tool_chain_completed": completed,
                    "answer_generation_status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise FinalAnswerGenerationError(
                f"LLM final-answer generation failed: {type(exc).__name__}"
            ) from exc
