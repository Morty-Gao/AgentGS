"""Load public answer-format contracts, independently of private evaluation gold."""

from pathlib import Path

from agentgs.agent.answering.formats import _decode_contracts, canonical_question


class AnswerContracts:
    def __init__(self, root: Path):
        folder = root / "data/answer_formats"
        path = folder / "geotooleval_v1_1.jsonl"
        self.contracts = _decode_contracts(path.read_bytes()) if path.exists() else {}
        system = folder / "geotooleval_v1_1_system.txt"
        self.system_prompt = system.read_text("utf-8") if system.exists() else ""

    def find(self, question):
        return self.contracts.get(canonical_question(question))
