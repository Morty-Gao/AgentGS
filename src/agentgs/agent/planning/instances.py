"""Execution instances with explicit identities, dependencies and input references."""

import hashlib
import json
from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field


class InputBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    path: list[str | int] = Field(default_factory=list)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    input_bindings: dict[str, InputBinding] = Field(default_factory=dict)
    arguments: dict = Field(default_factory=dict)
    capability_id: str | None = None
    skill_id: str | None = None


class ExecutionPlan(list):
    """Tool-name view for graph algorithms; instances retain the execution contract."""

    def __init__(self, steps):
        self.steps = [s if isinstance(s, PlanStep) else PlanStep.model_validate(s) for s in steps]
        seen = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"Duplicate execution step: {step.id}")
            required = set(step.depends_on)
            if not required <= seen:
                raise ValueError(f"Missing, forward or cyclic dependency for {step.id}")
            if not {b.step_id for b in step.input_bindings.values()} <= required:
                raise ValueError(f"Input bindings must name declared dependencies for {step.id}")
            seen.add(step.id)
        super().__init__(s.tool for s in self.steps)

    def align(self, chain):
        if list(self) == list(chain):
            return self
        steps, cursor = [], 0
        used = {step.id for step in self.steps}
        for i, name in enumerate(chain, 1):
            if cursor < len(self.steps) and self.steps[cursor].tool == name:
                steps.append(self.steps[cursor].model_copy(deep=True))
                cursor += 1
            else:
                identity = f"runtime_{i}"
                while identity in used:
                    identity += "_"
                used.add(identity)
                steps.append(PlanStep(id=identity, tool=name, purpose=f"Execute grounded operation {name}"))
        if cursor != len(self.steps):
            raise ValueError("A rewrite would remove or reorder declared execution instances; replan explicitly")
        return ExecutionPlan(steps)


def resolve_inputs(step, outputs):
    arguments = deepcopy(step.arguments)
    for field, binding in step.input_bindings.items():
        if binding.step_id not in outputs:
            raise ValueError(f"Missing upstream result: {binding.step_id}")
        value = outputs[binding.step_id]
        for key in binding.path:
            try:
                value = value[key]
            except (KeyError, TypeError, IndexError) as exc:
                raise ValueError(f"Missing upstream field for {step.id}.{field}") from exc
        arguments[field] = deepcopy(value)
    return arguments


def reuse_key(step, upstream_keys, query):
    """Reuse only completely specified scalar computations, never implicit file state."""
    if not step.arguments and not step.input_bindings:
        return None
    def numeric(value):
        return isinstance(value, (int, float, bool)) or (
            isinstance(value, list) and all(numeric(v) for v in value)
        )
    if not all(numeric(v) for v in step.arguments.values()):
        return None
    if any(not upstream_keys.get(ref) for ref in step.depends_on):
        return None
    payload = step.model_dump(mode="json", exclude={"id"})
    payload["upstream"] = {ref: upstream_keys[ref] for ref in step.depends_on}
    payload["question"] = query
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
