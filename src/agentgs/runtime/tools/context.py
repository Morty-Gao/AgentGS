"""Task-scoped scientific input identities and output directories."""

import json
import re
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
    artifact_root: Path

    @classmethod
    def create(cls, root, outputs, artifact_root=None):
        root, outputs = Path(root).resolve(), Path(outputs).resolve()
        manifest = root / "data/processed/resource_manifest.json"
        aliases = json.loads(manifest.read_text("utf-8"))["aliases"] if manifest.exists() else {}
        outputs.mkdir(parents=True, exist_ok=True)
        return cls(
            root, outputs, aliases, Path(artifact_root).resolve() if artifact_root else outputs
        )

    def dataset(self, name):
        return self.root / "data/processed" / name

    def resolve(self, value):
        if value is None:
            raise FileNotFoundError("No input resource supplied")
        text = str(value).strip().strip("\"'").replace("\\", "/")
        text = re.sub(r"/+", "/", text)
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
        data_root = (self.root / "data").resolve()
        if candidate.is_relative_to(data_root):
            relative = candidate.relative_to(data_root)
            # Evaluation labels/reports and application contracts are not
            # scientific input assets. The dedicated Geo snapshot adapter
            # reads its authorized replay data separately.
            if relative.parts and relative.parts[0] in {"answer_formats", "geotooleval"}:
                if relative != Path("geotooleval/10.自然灾害与极端地震.xlsx"):
                    raise ValueError(
                        "Evaluation labels and reports are unavailable to scientific tools"
                    )
            if relative.parts and relative.parts[0] == "geoquestions1089":
                raise ValueError("GeoQuestions replay data requires the dedicated snapshot tool")
        roots = (self.root / "data", self.artifact_root)
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
def tool_context(root, outputs, artifact_root=None):
    token = _current.set(ToolPaths.create(root, outputs, artifact_root))
    try:
        yield paths()
    finally:
        _current.reset(token)
