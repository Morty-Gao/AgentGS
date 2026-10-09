"""Scientific kernel dispatch, independent of scheduling and agent policies."""

import importlib.util

from agentgs.runtime.tools.domains.earthquake import (
    CatalogQuery,
    EventQuery,
    get_event,
    query_catalog,
)


def compute_tool(name, arguments, datasets, registry):
    if (
        name == "DeepEarthQueryUSGSCatalog"
        and arguments
        and set(arguments).issubset({"start_year", "end_year", "min_magnitude", "limit"})
    ):
        return query_catalog(CatalogQuery.model_validate(arguments), datasets)
    if name == "DeepEarthGetUSGSEvent" and importlib.util.find_spec("pandas") is None:
        return get_event(EventQuery.model_validate(arguments), datasets)
    if name == "GeoBigDataExecuteGeoSPARQL":
        return registry[name].handler(registry[name].input_type.model_validate(arguments), datasets)
    return registry[name].handler(arguments)
