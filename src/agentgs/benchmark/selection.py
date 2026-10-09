"""Choose which dataset a benchmark run reads."""

from pathlib import Path

from agentgs.evaluation.datasets.base import Dataset
from agentgs.evaluation.datasets.geotooleval import GeoToolEvalDataset


def resolve_dataset(dataset: Dataset | Path) -> Dataset:
    """Use a dataset instance as given. A path is the builtin GeoToolEval JSONL."""
    if isinstance(dataset, Dataset):
        return dataset
    return GeoToolEvalDataset(Path(dataset))


def task_dirname(question_id) -> str:
    if isinstance(question_id, int) and not isinstance(question_id, bool):
        return f"{question_id:04d}"
    text = str(question_id)
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"Invalid example id: {question_id!r}")
    return text


def id_from_dirname(name: str, known: set):
    if name.isdigit():
        number = int(name)
        if number in known:
            return number
    return name
