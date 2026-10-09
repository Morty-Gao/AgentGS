"""Load public answer-format contracts, independently of private evaluation gold."""

from pathlib import Path

from agentgs.contracts.answer_formats import canonical_question
from agentgs.datasets.geotooleval import (
    ANSWER_SYSTEM_PROMPT,
    benchmark_dataset,
    benchmark_manifest,
    load_answer_contracts,
)


class AnswerContracts:
    def __init__(self, root: Path):
        path = benchmark_dataset(root)
        if not path.exists() and benchmark_manifest(root).get("selected_dataset"):
            raise FileNotFoundError(f"Selected GeoToolEval dataset is missing: {path}")
        self.contracts = load_answer_contracts(path) if path.exists() else {}
        self.system_prompt = ANSWER_SYSTEM_PROMPT

    def find(self, question):
        return self.contracts.get(canonical_question(question))
