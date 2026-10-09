"""Load public answer-format contracts, independently of private evaluation gold."""

from pathlib import Path

from agentgs.config.paths import data_resource
from agentgs.contracts.answer_formats import _decode_contracts, canonical_question


class AnswerContracts:
    def __init__(self, root: Path):
        folder = root / "data/answer_formats"
        import json

        manifest = root / "data/metadata/geotooleval.json"
        selected = (
            json.loads(manifest.read_text("utf-8")).get("version_manifest", {})
            if manifest.exists()
            else {}
        )
        fixed = folder / "geotooleval.jsonl"
        path = (
            data_resource(root, selected["public_contracts"])
            if selected
            else fixed
            if fixed.exists()
            else folder / "geotooleval_v1_1.jsonl"
        )
        if not path.exists() and (selected or (root / "data/geotooleval.jsonl").exists()):
            raise FileNotFoundError(f"Selected public answer contracts are missing: {path}")
        self.contracts = _decode_contracts(path.read_bytes()) if path.exists() else {}
        system = (
            data_resource(root, selected["public_system_prompt"])
            if selected
            else folder / "system.txt"
            if path == fixed
            else folder / "geotooleval_v1_1_system.txt"
        )
        if path == fixed and not system.exists():
            raise FileNotFoundError(f"Fixed-answer system prompt is missing: {system}")
        self.system_prompt = system.read_text("utf-8") if system.exists() else ""

    def find(self, question):
        return self.contracts.get(canonical_question(question))
