"""Select a dataset and run it. Metrics and dataset definitions stay in ``agentgs.evaluation``."""

from agentgs.benchmark.cases import EvaluationCase, evaluate, load_cases
from agentgs.benchmark.inputs import public_input
from agentgs.benchmark.runner import run_benchmark
from agentgs.benchmark.score import score_predictions

__all__ = [
    "EvaluationCase",
    "evaluate",
    "load_cases",
    "public_input",
    "run_benchmark",
    "score_predictions",
]
