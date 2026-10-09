"""FastMCP transport for the 101 GeoToolEval tools.

The runnable batch entry is run_geotooleval.py, which uses the same call_tool
bridge inside the ReAct loop. This process exists so an Earth-Agent config can
mount the benchmark tools the same way it mounts Index, Inversion, Perception,
Analysis, and Statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from geotool_bridge import call_tool, catalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp_dir", default="")
    args, _unknown = parser.parse_known_args()
    if args.temp_dir:
        os.environ["AGENTGS_TOOL_OUTPUT"] = args.temp_dir
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit(
            "fastmcp is not installed. Run forks/Earth-Agent/run_geotooleval.py, "
            "which uses the same tools without a separate MCP process."
        ) from exc
    mcp = FastMCP("GeoToolEval")
    for spec in catalog():

        def _register(name: str, description: str) -> None:
            def run(arguments: dict | None = None) -> str:
                result = call_tool(name, arguments or {})
                if isinstance(result, str):
                    return result
                return json.dumps(result, ensure_ascii=False, default=str)

            run.__name__ = name
            run.__doc__ = description
            mcp.tool(name=name, description=description)(run)

        _register(spec["name"], spec["description"][:2000])
    mcp.run()


if __name__ == "__main__":
    main()
