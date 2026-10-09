"""Evaluation datasets. Importing this package registers the builtin datasets."""

from agentgs.evaluation.datasets.base import Dataset, Example, get_dataset
from agentgs.evaluation.datasets.geotooleval import EvalTask, GeoToolEvalDataset, load_tasks

__all__ = [
    "Dataset",
    "EvalTask",
    "Example",
    "GeoToolEvalDataset",
    "get_dataset",
    "load_tasks",
]
