"""Build public tool argument schemas from shared types and domain declarations.

Runtime annotations are kept separate from model arguments. GeoSPARQL supplies
its own input model in the runtime registry.
"""

from functools import lru_cache, partial

from . import hazard, lookup, remote_sensing, spatial
from .base import (
    ArtifactReference,
    ArtifactSelector,
    RasterPair,
    ToolArguments,
    create_argument_model,
)

__all__ = [
    "ArtifactReference",
    "ArtifactSelector",
    "RasterPair",
    "ToolArguments",
    "tool_input_models",
]


@lru_cache(maxsize=1)
def tool_input_models() -> dict[str, type[ToolArguments]]:
    """Return the cached argument model registry in the original declaration order."""
    result = {}

    def add(name, types, required=(), alternatives=(), exclusive=(), *, base=ToolArguments):
        result[name] = create_argument_model(
            name, types, required, alternatives, exclusive, base=base
        )

    for domain, base in (
        (lookup, ToolArguments),
        (spatial, spatial.SpatialArguments),
        (remote_sensing, remote_sensing.RemoteSensingArguments),
        (hazard, hazard.HazardArguments),
    ):
        domain.register(partial(add, base=base))
    return result
