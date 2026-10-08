"""Install the original scientific/embedding versions in a project-local Python 3.12 environment."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    destination = root / "tools/envs/reproduction"
    python = destination / "bin/python"
    constraints = root / "scripts/tools/manifests/reproduction.constraints.txt"
    if not args.verify:
        if not python.exists():
            subprocess.run(["uv", "venv", "--python", args.python, str(destination)], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(python), "--constraint", str(constraints),
                        "-e", str(root) + "[scientific,retrieval,visualization]"], check=True)
    code = "import importlib.metadata as m,json,sys;print(json.dumps({'python':sys.version,'packages':{p:m.version(p) for p in " + repr([line.split("==")[0] for line in constraints.read_text().splitlines() if "==" in line]) + "}}))"
    result = subprocess.run([str(python), "-c", code], check=True, capture_output=True, text=True)
    receipt = json.loads(result.stdout)
    for line in constraints.read_text().splitlines():
        if "==" in line:
            name, version = line.split("==")
            if receipt["packages"][name] != version:
                raise ValueError(f"Reproduction distribution differs: {name}")
    if not receipt["python"].startswith("3.12."):
        raise ValueError("Reproduction profile requires Python 3.12")
    state = root / "tools/state/reproduction.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"verified": True, "receipt": str(state)}))


if __name__ == "__main__":
    main()
