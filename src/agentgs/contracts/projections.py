"""Strict JSON value projection for observed producer outputs."""

from copy import deepcopy


def project_json(value, path):
    if not isinstance(path, (list, tuple)):
        raise ValueError("JSON projection path must be a sequence of keys and indices")
    for key in path:
        if isinstance(value, dict) and isinstance(key, str):
            if key not in value:
                raise ValueError(f"Missing JSON object field: {key}")
            value = value[key]
        elif isinstance(value, list) and type(key) is int and 0 <= key < len(value):
            value = value[key]
        else:
            raise ValueError(f"Invalid JSON projection segment: {key!r}")
    return deepcopy(value)
