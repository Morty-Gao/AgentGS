"""Remote-sensing contract policy and request-local enablement."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_enabled = ContextVar("rs_contracts_enabled", default=True)
VERSION = "rs-local-v1"
FLAG = "GEOTOOL_RS_LOCAL_CONTRACTS"
PAIR_FIELDS = {
    "RemoteSensingNDVI": (
        ("nir_band_files", "input_nir_paths", "nir_band_file", "input_nir_path"),
        ("red_band_files", "input_red_paths", "red_band_file", "input_red_path"),
        ("nir", "red"),
    ),
    "RemoteSensingNDWI": (
        ("input_swir_paths", "green_band_files", "input_swir_path", "green_band_file"),
        ("input_nir_paths", "nir_band_files", "input_nir_path", "nir_band_file"),
        ("green", "nir"),
    ),
    "RemoteSensingNBR": (
        ("input_nir_paths", "input_nir_path"),
        ("input_swir_paths", "input_swir_path"),
        ("nir", "swir"),
    ),
    "RemoteSensingBatchNDSI": (("green_file_list",), ("swir_file_list",), ("green", "swir")),
    "RemoteSensingCloudMask": (
        ("sr_band_paths", "sr_band_path"),
        ("qa_pixel_paths", "qa_pixel_path"),
        (None, "qa"),
    ),
}
RASTER_STEPS = set(PAIR_FIELDS) | {
    "RemoteSensingNDTI",
    "RemoteSensingTiffDifference",
    "RemoteSensingBatchImageMean",
    "RemoteSensingBatchImageSum",
    "RemoteSensingTiffAverage",
}
SCALAR_CONSUMERS = {
    "RemoteSensingMean",
    "RemoteSensingLinearTrend",
    "RemoteSensingSensSlope",
    "RemoteSensingMannKendallTest",
    "RemoteSensingCoefficientOfVariation",
    "RemoteSensingKurtosis",
    "RemoteSensingMaxValueAndIndex",
    "RemoteSensingMinValueAndIndex",
    "RemoteSensingArgmax",
}
SKILLS = {
    "remote-sensing-data-discovery",
    "remote-sensing-preprocessing",
    "remote-sensing-raster-statistics",
    "remote-sensing-change-detection",
    "remote-sensing-trend-distribution",
}


def enabled() -> bool:
    return _enabled.get()


def skill_profile(skill_id: str) -> dict[str, Any]:
    if not enabled() or skill_id not in SKILLS:
        return {}
    return {
        "version": VERSION,
        "pair_by": ["acquisition_date", "scene", "band_role"],
        "unknown_metadata": "report_unknown; never treat it as verified",
        "date_window": "use rs_date_window=[start,end] only when explicitly specified",
        "batch_cardinality": "one batch OR disjoint per-pair calls, never both",
        "rs_statistic_mode": "per_scene_statistics returns scalars; pixelwise_composite requires aligned grids; legacy retains both outputs",
    }


@contextmanager
def contract_scope(active: bool) -> Iterator[None]:
    """Enable or disable contracts in this context and restore on every exit."""
    token = _enabled.set(active)
    try:
        yield
    finally:
        _enabled.reset(token)
