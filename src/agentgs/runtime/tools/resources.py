"""Resource adapters shared by the preserved scientific kernels."""

from importlib import import_module

from agentgs.runtime.tools.context import paths


def _resolve_path(value):
    return paths().resolve(value)


def _output_path(value, default_name):
    return paths().output(value, default_name)


def _load_dataset_generator(name):
    modules = {
        "generate_energy_resources_toolqa": "agentgs.datasets.parsers.energy",
        "generate_ocean_carbon_toolqa": "agentgs.datasets.parsers.ocean",
    }
    if name not in modules:
        raise ValueError(f"Unknown shared dataset parser: {name}")
    return import_module(modules[name])
