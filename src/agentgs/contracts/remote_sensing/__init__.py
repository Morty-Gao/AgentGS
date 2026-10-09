"""Opt-in remote-sensing contracts, shared by binding and runtime dispatch.

Use contract_scope() for request-local policy; metadata inspection and step
preparation retain their existing report format and numerical policies.
"""

from .assets import align_scene_pairs, asset_metadata
from .policy import (
    FLAG,
    PAIR_FIELDS,
    RASTER_STEPS,
    SCALAR_CONSUMERS,
    SKILLS,
    VERSION,
    contract_scope,
    enabled,
    skill_profile,
)
from .steps import prepare_step
from .validation import inspect_inputs

__all__ = [
    "FLAG",
    "PAIR_FIELDS",
    "RASTER_STEPS",
    "SCALAR_CONSUMERS",
    "SKILLS",
    "VERSION",
    "align_scene_pairs",
    "asset_metadata",
    "contract_scope",
    "enabled",
    "inspect_inputs",
    "prepare_step",
    "skill_profile",
]
