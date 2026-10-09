#!/usr/bin/env python3
"""Export the first page of each editable paper diagram to PDF and PNG."""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawio", default=os.environ.get("DRAWIO", "drawio"))
    args = parser.parse_args()
    for name in ("overview", "performance", "case_study", "architecture", "interface"):
        source = ROOT / "figures/source" / f"{name}.drawio"
        for extension in ("pdf", "png"):
            output = ROOT / "figures" / f"{name}.{extension}"
            command = [
                args.drawio, "--no-sandbox", "--disable-gpu", "--export",
                "--timeout", "45", "--page-index", "1", "--theme", "light",
                "--border", "2", "--output", str(output), str(source),
            ]
            command += ["--crop"] if extension == "pdf" else ["--scale", "2"]
            subprocess.run(command, check=True, timeout=60)


if __name__ == "__main__":
    main()
