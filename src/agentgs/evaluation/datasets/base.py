"""Dataset contract. Subclasses turn one source into examples; they do not run the agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True)
class Example:
    """One scored item. Gold stays on the example and is not a runtime request."""

    question_id: int | str
    category_id: int | str
    category: str
    question: str
    expected_answer: str
    answer_type: str
    input_resource: Any = None
    gold_steps: tuple[dict[str, Any], ...] = ()
    answer_format: dict[str, Any] | None = None
    dataset_instructions: str = ""

    def __post_init__(self):
        object.__setattr__(self, "gold_steps", tuple(self.gold_steps))


class Dataset(ABC):
    """A loadable evaluation dataset.

    Set ``name`` to register the subclass. Import the subclass, then pass that
    name to ``get_dataset``. Builtin datasets live in this package; another
    dataset is a subclass with its own ``name``.
    """

    name: ClassVar[str] = ""
    domain_count: ClassVar[int | None] = None
    registry: ClassVar[dict[str, type[Dataset]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        registered = cls.__dict__.get("name", "")
        if not isinstance(registered, str) or not registered:
            return
        previous = Dataset.registry.get(registered)
        if previous is not None and previous is not cls:
            raise TypeError(
                f"Dataset name {registered!r} is already used by {previous.__qualname__}"
            )
        Dataset.registry[registered] = cls

    def __init__(self, source: Path):
        self.source = Path(source)

    @abstractmethod
    def load(self) -> list[Example]:
        """Return every example. Reject duplicate ids and an empty source."""

    def __iter__(self):
        return iter(self.load())

    def __len__(self):
        return len(self.load())

    def by_id(self) -> dict[int | str, Example]:
        return {example.question_id: example for example in self.load()}


def get_dataset(name: str, source: Path) -> Dataset:
    try:
        kind = Dataset.registry[name]
    except KeyError as exc:
        known = ", ".join(sorted(Dataset.registry)) or "(none)"
        raise KeyError(f"Unknown dataset {name!r}. Known datasets: {known}") from exc
    return kind(source)
