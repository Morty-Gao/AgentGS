"""Frozen inputs and shared contracts for the module experiment scripts."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

from agentgs.benchmark.inputs import public_input
from agentgs.config.algorithm import load_algorithm, resolve_algorithm
from agentgs.contracts.models import TaskInput
from agentgs.contracts.models.modules import ModuleSettings, resolve_modules

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY = ROOT / "outputs/ablations/module-mechanism-study"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker_only():
    if "worker" not in socket.gethostname():
        raise SystemExit(
            "Run on a currently discovered CPU worker; use on_cpu_worker.sh. No local fallback."
        )


def source_snapshot(root):
    paths = [root / name for name in ("pyproject.toml", "uv.lock", "config.toml", "llm.toml")]
    for directory in (
        root / "src/agentgs",
        root / "scripts/benchmark",
        root / "scripts/evaluation",
        root / "scripts/ablation",
    ):
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".json", ".toml", ".sh"}
            and "__pycache__" not in path.parts
        )
    return {str(path.relative_to(root)): file_hash(path) for path in sorted(set(paths))}


def data_inventory(root):
    """Bounded metadata inventory; raw downloads and symlink trees are excluded."""
    entries = {}
    for directory, folders, files in os.walk(root / "data", followlinks=False):
        folders[:] = sorted(
            name for name in folders if name not in {"raw", "__pycache__", ".cache"}
        )
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_file():
                stat = path.stat()
                entries[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return entries


def cell_modules(cell):
    return resolve_modules(ModuleSettings(), cell["ablations"])[0].model_dump()


def request_for(row, study, cell, contracts):
    base = public_input(
        row,
        output_contract=contracts.find(row.question),
        dataset_instructions=contracts.system_prompt,
    )
    return TaskInput.model_validate(
        {
            **base.model_dump(),
            **study["design"]["common_task"],
            "modules": cell_modules(cell),
            "ablations": cell["ablations"],
        }
    )


def effective_policy(root, design, cell):
    task = TaskInput.model_validate(
        {**design["common_task"], "modules": cell_modules(cell), "ablations": cell["ablations"]}
    )
    return resolve_algorithm(load_algorithm(root), task).model_dump(mode="json")


def load_study(directory, *, check_source=True):
    directory = Path(directory).resolve()
    study = json.loads((directory / "study.json").read_text())
    identity = study.pop("study_id")
    if digest(study) != identity:
        raise ValueError("Study manifest was modified; prepare a new study directory")
    study["study_id"] = identity
    root = Path(study["root"])
    for relative, expected in study["frozen_inputs"].items():
        if file_hash(directory / relative) != expected:
            raise ValueError(f"Frozen study input changed: {relative}")
    if check_source and source_snapshot(root) != study["source_snapshot"]:
        raise ValueError(
            "Code, declarations, model or project configuration changed; prepare a new study"
        )
    return study


def chain_group(row):
    return "single" if len(row.gold_steps) == 1 else "short" if len(row.gold_steps) <= 3 else "long"
