from agentgs.contracts.models import Model, TaskInput


class EvaluationCase(Model):
    id: str
    input: TaskInput
    expected_answer: str | None = None
    expected_tools: list[str] | None = None
