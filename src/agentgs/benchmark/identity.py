"""Content identities for packaged runtime declarations and request controls."""

import hashlib
import os
from importlib.resources import files

from agentgs.client.openai_compatible import request_extras


def planning_resource_hashes(root, planning_mode):
    packages = ["contracts"]
    if planning_mode == "skill_guided":
        packages.append("skills")
    result = {}

    def walk(folder, prefix):
        for entry in sorted(folder.iterdir(), key=lambda path: path.name):
            name = prefix + "/" + entry.name
            if entry.is_dir() and entry.name != "__pycache__":
                walk(entry, name)
            elif entry.is_file() and entry.name.endswith((".json", ".md")):
                result[name] = hashlib.sha256(entry.read_bytes()).hexdigest()

    for package in packages:
        source = root / "src" / "agentgs" / package
        walk(source if source.is_dir() else files("agentgs." + package), package)
    if planning_mode != "data_only":
        from agentgs.client.embedding import CACHE_FILES

        for name in CACHE_FILES:
            path = root / "embedding" / name
            result["embedding/" + name] = (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            )
    return result


def model_request_controls():
    return {
        "reasoning_effort_override": os.environ.get("AGENTGS_REASONING_EFFORT", "").strip(),
        **request_extras(),
    }
