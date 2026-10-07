import ast
from pathlib import Path


def test_agent_does_not_import_runtime_or_web():
    root = Path(__file__).parents[2] / "agentgs" / "agent"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [x.name for x in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(
                name == "agentgs.runtime"
                or name.startswith(("agentgs.runtime.", "agentgs.visualization.", "agentgs.cli."))
                for name in names
            ), str(path)
