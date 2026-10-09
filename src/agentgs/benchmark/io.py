"""Atomic JSON writes shared by benchmark runs."""

import json
from uuid import uuid4


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + "." + uuid4().hex + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
