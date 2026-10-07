import os
import re
from pathlib import Path

from dotenv import dotenv_values


def environment(root: Path) -> dict[str, str]:
    return {
        **{k: v for k, v in dotenv_values(root / ".env").items() if v is not None},
        **os.environ,
    }


def interpolate(value, env: dict[str, str]):
    if isinstance(value, str):

        def replace(match):
            name = match.group(1)
            if not env.get(name):
                raise ValueError(f"Missing environment variable: {name}")
            return env[name]

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
    if isinstance(value, dict):
        return {k: interpolate(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, env) for v in value]
    return value
