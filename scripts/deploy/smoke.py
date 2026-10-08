"""Check the transferred data and run two offline tasks without model calls."""

import json
import subprocess
import sys
from pathlib import Path


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    required = (
        "data/processed/resource_manifest.json",
        "data/indexes/geotoolkg/index_store.json",
        "data/models/fastembed",
        "data/geotooleval/version_manifest.json",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise SystemExit(
            "Transfer the entire prepared data/ directory. Missing: " + ", ".join(missing)
        )
    from agentgs.runtime.tools.registry import builtin_tools
    from agentgs.skills.registry import builtin_skills

    if len(builtin_tools()) != 101 or len(builtin_skills()) != 24:
        raise SystemExit("Unexpected tool or Skill registry")
    for tool, arguments in (
        ("DeepEarthQueryUSGSCatalog", {"year": 2011, "operation": "largest"}),
        ("GeoBigDataExecuteGeoSPARQL", {"source_id": 591}),
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "agentgs",
                "run",
                "--project",
                str(root),
                "--tool",
                tool,
                "--arguments",
                json.dumps(arguments),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode:
            raise SystemExit(f"Offline task failed: {tool}; check outputs/tasks/ on this server")
        record = json.loads(result.stdout)
        if record["status"] != "succeeded":
            raise SystemExit(f"Offline task did not succeed: {tool}")
        print(f"OK {tool}", flush=True)


if __name__ == "__main__":
    main()
