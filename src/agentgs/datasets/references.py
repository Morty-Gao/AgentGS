"""Load versioned dataset selections and reference values from project data/."""

import json
from pathlib import Path

from agentgs.config.paths import project_root, within

REFERENCE_FILES = (
    "reference/energy.json",
    "reference/ocean.json",
    "reference/geography.json",
    "geoquestions/profile.json",
)


def load_reference(relative: str, root: Path | None = None):
    candidates = []
    if root is not None:
        candidates.append(Path(root).resolve())
    try:
        candidates.append(project_root())
    except ValueError:
        pass
    # Editable installations and task fixtures share the checked-out data inputs.
    candidates.append(Path(__file__).resolve().parents[3])
    for candidate in dict.fromkeys(candidates):
        path = within(candidate / "data", relative)
        if path.is_file():
            return json.loads(path.read_text("utf-8"))
    raise FileNotFoundError(f"Missing dataset reference: data/{relative}")
