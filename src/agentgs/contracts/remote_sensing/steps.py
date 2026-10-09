"""Prepare agent raster steps using the preserved binding and statistics policy."""

from .policy import RASTER_STEPS, SCALAR_CONSUMERS, VERSION, enabled
from .validation import inspect_inputs


def prepare_step(tool: str, params: dict, context: dict) -> dict:
    """Agent-only opt-in and local fixes. Never touches a non-RS tool."""
    if (
        not enabled()
        or tool not in RASTER_STEPS
        or context.get("remote_sensing_contracts") is False
    ):
        return {}
    params["_rs_profile"] = VERSION
    issues = []
    if (
        tool == "RemoteSensingTiffDifference"
        and isinstance(params.get("image_a_paths"), list)
        and isinstance(params.get("image_b_paths"), list)
    ):
        first, second = (params["image_a_paths"], params["image_b_paths"])
        count = max(1, int(context.get("occurrence_count", 1)))
        index = int(context.get("occurrence_index", 0))
        partition = bool(context.get("rs_auto_consecutive_batch"))
        if len(first) == len(second) and count > 1 and partition:
            if count > len(first) or not 0 <= index < count:
                issues.append(
                    "more repeated difference steps than available pairs; replan instead of replaying"
                )
            else:
                start, end = (len(first) * index // count, len(first) * (index + 1) // count)
                first, second = (first[start:end], second[start:end])
                params["image_a_paths"], params["image_b_paths"] = (first, second)
        params.pop("image_a_path", None)
        params.pop("image_b_path", None)
        if count > 1 and partition and (len(first) == len(second) == 1):
            params["image_a_path"], params["image_b_path"] = (first[0], second[0])
            params.pop("image_a_paths", None)
            params.pop("image_b_paths", None)
    if tool == "RemoteSensingBatchImageMean" and "rs_statistic_mode" not in params:
        chain = list(context.get("tool_chain") or [])
        position = int(context.get("execution_index", 0))
        following = chain[position:] if position else []
        if (
            not context.get("rs_requires_composite")
            and following
            and any((t in SCALAR_CONSUMERS for t in following))
            and all(
                (
                    t in SCALAR_CONSUMERS
                    or t
                    in {
                        "RemoteSensingBatchImageMean",
                        "RemoteSensingDifference",
                        "RemoteSensingDivision",
                        "RemoteSensingPercentageChange",
                        "RemoteSensingKelvinToCelsius",
                    }
                    for t in following
                )
            )
        ):
            params["rs_statistic_mode"] = "per_scene_statistics"
    report = inspect_inputs(tool, params, inspect_raster=True)
    report["issues"].extend(issues)
    params["_rs_binding_issues"] = report["issues"]
    report["rs_statistic_mode"] = params.get("rs_statistic_mode", "legacy")
    return report
