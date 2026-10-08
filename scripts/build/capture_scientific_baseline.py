"""Capture original numerical decisions with synthetic data and isolated outputs."""

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

from agentgs.evaluation.scientific import bind_inputs, create_inputs, observe_result


def original_runtime(source, output):
    for name, folder in [
        ("source_science", source / "ToolsAgent/ToolsFuns/GeoEval"),
        ("app", source / "app"),
        ("app.tools", source / "app/tools"),
    ]:
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    path = source / "app/tools/rs_migration_scope.py"
    spec = importlib.util.spec_from_file_location("app.tools.rs_migration_scope", path)
    scope = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scope
    spec.loader.exec_module(scope)
    path = source / "ToolsAgent/ToolsFuns/GeoEval/runtime.py"
    module = types.ModuleType("source_science.runtime")
    module.__file__, module.__package__ = str(path), "source_science"
    sys.modules[module.__name__] = module

    class OutputOwnership(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(n, ast.Name) and n.id == "_RUNTIME_OUTPUTS" for n in node.targets):
                node.value = ast.Call(
                    func=ast.Name(id="Path", ctx=ast.Load()),
                    args=[ast.Constant(str(output))],
                    keywords=[],
                )
            return node

    tree = OutputOwnership().visit(ast.parse(path.read_text("utf-8")))
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=root / "outputs/alignment/scientific.json")
    parser.add_argument(
        "--cases", type=Path, default=root / "src/tests/fixtures/scientific/synthetic.json"
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.is_relative_to(source):
        parser.error("The original source is read-only")
    contract = json.loads((root / "docs/algorithm_alignment.json").read_text())
    for relative, identity in contract["baseline"]["files"].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != identity["sha256"]:
            parser.error(f"Original baseline differs: {relative}")
    profile = json.loads(args.cases.read_text())
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        inputs = create_inputs(work)
        runtime = original_runtime(source, work / "outputs")
        for case in profile["cases"]:
            result = json.loads(
                runtime._tool(case["tool"])(json.dumps(bind_inputs(case["arguments"], inputs)))
            )
            case["expected"] = observe_result(result, inputs, work / "outputs")
    profile["source_tree_sha256"] = contract["baseline"]["tree_sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(profile["cases"]), "output": str(output)}))


if __name__ == "__main__":
    main()
