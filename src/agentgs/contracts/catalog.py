"""One metadata authority for tools, planning, scientific contracts and indexes."""

import json
from importlib.resources import files


def load_catalog():
    return json.loads(files("agentgs.contracts").joinpath("catalog.json").read_text("utf-8"))


def get_tool_spec(name):
    return next((row for row in load_catalog()["specs"] if row["Name"] == name), None)
