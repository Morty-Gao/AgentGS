"""Load public answer-format contracts, independently of private evaluation gold."""

from pathlib import Path

from agentgs.agent.answering.formats import _decode_contracts, canonical_question
from agentgs.config.paths import data_resource


class AnswerContracts:
    def __init__(self, root: Path):
        folder = root / "data/answer_formats"
        import json

        manifest = root / "data/geotooleval/version_manifest.json"
        selected = json.loads(manifest.read_text("utf-8")) if manifest.exists() else {}
        path = (
            data_resource(root, selected["public_contracts"])
            if selected
            else folder / "geotooleval_v1_1.jsonl"
        )
        if selected and not path.exists():
            raise FileNotFoundError(f"Selected public answer contracts are missing: {path}")
        self.contracts = _decode_contracts(path.read_bytes()) if path.exists() else {}
        system = (
            data_resource(root, selected["public_system_prompt"])
            if selected
            else folder / "geotooleval_v1_1_system.txt"
        )
        self.system_prompt = system.read_text("utf-8") if system.exists() else ""

    def find(self, question):
        return self.contracts.get(canonical_question(question))
