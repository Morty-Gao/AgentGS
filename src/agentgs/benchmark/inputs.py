"""Runtime requests built from an example. Gold fields stay out of the request."""

import json

from agentgs.contracts.models import TaskInput

PRIVATE_KEYS = frozenset(
    {
        "gold_answer",
        "gold_sparql",
        "gold_chain",
        "gold_steps",
        "expected_answer",
        "reference_answer",
        "reference_tools",
        "expected_tools",
        "gold_chain_text",
    }
)


def public_resource(value):
    if isinstance(value, dict):
        return {
            key: public_resource(item) for key, item in value.items() if key not in PRIVATE_KEYS
        }
    if isinstance(value, list):
        return [public_resource(item) for item in value]
    return value


def public_input(
    task,
    *,
    planning_mode="skill_guided",
    allow_rule_planning=False,
    plan_attempts=None,
    tool_attempts=None,
    command_attempts=None,
    output_contract=None,
):
    resource = task.input_resource
    if isinstance(resource, str):
        try:
            resource = json.loads(resource) if resource else None
        except ValueError:
            pass
    return TaskInput(
        question=task.question,
        input_resource=public_resource(resource),
        planning_mode=planning_mode,
        allow_rule_planning=False if planning_mode == "data_only" else allow_rule_planning,
        plan_attempts=plan_attempts,
        tool_attempts=tool_attempts,
        command_attempts=command_attempts,
        output_contract=output_contract,
        input_domain="remote_sensing" if task.category == "RemoteSensingTools" else None,
    )
