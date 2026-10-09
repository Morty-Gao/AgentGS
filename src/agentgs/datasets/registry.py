import csv
from pathlib import Path

from agentgs.datasets.layout import prepared_name, prepared_path


class DatasetRegistry:
    def __init__(self, root: Path):
        self.project = root
        self.root = root / "data"

    def resolve(self, dataset: str, relative: str) -> Path:
        if not dataset or "/" in dataset or "\\" in dataset or dataset in {".", ".."}:
            raise ValueError("Invalid dataset id")
        path = prepared_path(self.project, prepared_name(dataset, relative))
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset resource: {dataset}/{relative}")
        return path

    def csv_rows(self, dataset, relative):
        with self.resolve(dataset, relative).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
