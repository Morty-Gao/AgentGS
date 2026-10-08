"""Task-scoped scientific input identities and output directories."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path
from uuid import uuid4

_current = ContextVar("scientific_task_paths", default=None)


@dataclass
class ToolPaths:
    root: Path
    outputs: Path
    aliases: dict

    @classmethod
    def create(cls, root, outputs):
        root, outputs = Path(root).resolve(), Path(outputs).resolve()
        manifest = root / "data/processed/resource_manifest.json"
        aliases = json.loads(manifest.read_text("utf-8"))["aliases"] if manifest.exists() else {}
        outputs.mkdir(parents=True, exist_ok=True)
        return cls(root, outputs, aliases)

    def dataset(self, name):
        return self.root / "data/processed" / name

    def resolve(self, value):
        if value is None:
            raise FileNotFoundError("No input resource supplied")
        text = str(value).strip().strip("\"'").replace("\\", "/")
        if "=" in text and not (len(text) > 1 and text[1] == ":"):
            text = text.rsplit("=", 1)[-1].strip()
        text = text.replace("{asset_dir}", "")
        relative = self.aliases.get(text)
        if relative:
            candidate = (self.root / relative).resolve()
        else:
            candidate = Path(text).expanduser()
            candidate = (
                candidate.resolve()
                if candidate.is_absolute()
                else (self.root / candidate).resolve()
            )
        roots = (self.root / "data", self.outputs)
        if candidate.exists() and any(candidate.is_relative_to(p.resolve()) for p in roots):
            return candidate
        raise FileNotFoundError(f"Input resource is absent from this task's declared data: {value}")

    def output(self, value, default):
        # File locations change, while requested formats and algorithm outputs stay intact.
        name = Path(str(value).replace("\\", "/")).name if value else default
        if name in {"", ".", ".."}:
            raise ValueError("Invalid output artifact name")
        directory = self.outputs / "files" / uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        return directory / name


def paths():
    value = _current.get()
    if value is None:
        raise RuntimeError("Scientific tools require an explicit task resource context")
    return value


def task_cache(maxsize=None):
    """Keep data caches within a task rather than across unrelated data roots."""

    def decorate(function):
        @lru_cache(maxsize=maxsize)
        def cached(identity, args):
            return function(*args)

        @wraps(function)
        def invoke(*args):
            context = paths()
            return cached((str(context.root), str(context.outputs)), args)

        return invoke

    return decorate


@contextmanager
def tool_context(root, outputs):
    token = _current.set(ToolPaths.create(root, outputs))
    try:
        yield paths()
    finally:
        _current.reset(token)
