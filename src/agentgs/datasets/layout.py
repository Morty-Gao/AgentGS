"""Names of prepared data files and legacy resource locators."""

from pathlib import PurePosixPath

PREPARED_FILES = {
    "geotooleval/GeoToolEval.jsonl": "geotooleval.jsonl",
    "geotooleval/dataset_manifest.json": "geotooleval_manifest.json",
    "geotooleval/workbook_metadata.jsonl": "geotooleval_workbook_metadata.jsonl",
    "geotooleval/version_manifest.json": "geotooleval_version_manifest.json",
    "answer_formats/geotooleval_fixed.jsonl": "geotooleval_answer_formats.jsonl",
    "answer_formats/geotooleval_fixed_system.txt": "geotooleval_answer_system.txt",
    "geoquestions1089/processed/geobigdata_snapshot.json": "geoquestions_queries.jsonl",
    "geoquestions1089/processed/questions_40.jsonl": "geoquestions_questions.jsonl",
    "geoquestions1089/processed/manifest.json": "geoquestions_manifest.json",
    "processed/resource_manifest.json": "resource_manifest.json",
    "processed/habitable/metal_minerals_subset.csv": "mineral_deposits.csv",
    "processed/habitable/deep_time_bintanja2005_temp_sealevel.csv": "climate_bintanja2005.csv",
    "processed/habitable/deep_time_friedrich2016_temp.csv": "climate_friedrich2016.csv",
    "processed/habitable/future_climate_ssp_warming_summary.csv": "climate_ssp_warming.csv",
    "processed/habitable/present_climate_gistemp_global_annual.csv": "climate_gistemp_annual.csv",
    "processed/hazard/events.jsonl": "hazard_events.jsonl",
    "processed/deep_earth/manifest.json": "deep_earth_manifest.json",
    "processed/deep_earth/source_snapshot/usgs_2010_2020_m7.csv": "deep_earth_usgs_source.csv",
    "processed/assets/biodeeptime_geospatial_subset.csv": "biodeeptime_observations.csv",
    "processed/assets/biodeeptime_series_summary.csv": "biodeeptime_series.csv",
    "processed/assets/deep_time_analysis_zones.geojson": "biodeeptime_zones.geojson",
    "processed/assets/ccs_analysis_zones.geojson": "ccs_zones.geojson",
    "processed/assets/ccs_full_dataset_v1.0.csv": "ccs_cases.csv",
    "processed/assets/ccs_injection_daily_v1.0.csv": "ccs_injection_daily.csv",
    "processed/assets/ccs_injection_monthly_v1.0.csv": "ccs_injection_monthly.csv",
}
for name in [
    "cross_event_samples",
    "event_semucb_samples",
    "location_model_samples",
    "location_semucb_samples",
    "usgs_catalog",
]:
    PREPARED_FILES[f"processed/deep_earth/derived/{name}.csv"] = f"deep_earth_{name}.csv"
for name in [
    "country_transition_metrics",
    "gip_operating_country_type",
    "gip_top_operating_facilities",
    "mrds_critical_mineral_snapshot",
    "opsd_capacity_year_end",
    "opsd_capital_nearest",
    "opsd_country_energy_summary",
    "opsd_landmark_facilities",
    "power_mineral_nearest_samples",
]:
    PREPARED_FILES[f"processed/energy/derived/{name}.csv"] = f"energy_{name}.csv"
for name in ["gip_renewable_points", "mrds_critical_mineral_points"]:
    PREPARED_FILES[f"processed/energy/full/{name}.csv.gz"] = f"energy_{name}.csv"
for name in [
    "glodap_surface_deep_profiles",
    "snapo_cruise_dedup_summary",
    "snapo_cruise_qc_summary",
    "snapo_station_samples",
    "socat_glodap_nearest_samples",
    "socat_monthly_grid_samples",
    "socat_north_pacific_year_summary",
    "socat_region_map_samples",
    "socat_region_trends",
]:
    PREPARED_FILES[f"processed/ocean/derived/{name}.csv"] = f"ocean_{name}.csv"
for name in [
    "ccs_crs_inventory.csv",
    "ccs_facilities.geojson",
    "ccs_facilities_webmercator.csv",
    "ccs_injection_monthly_geo.csv",
]:
    PREPARED_FILES[f"processed/assets/{name}"] = name

FLAT_FILES = dict(PREPARED_FILES)
for key, name in list(PREPARED_FILES.items()):
    for prefix in [
        "climate",
        "energy",
        "ocean",
        "biodeeptime",
        "ccs",
        "deep_earth",
        "geoquestions",
    ]:
        if name.startswith(prefix + "_"):
            PREPARED_FILES[key] = prefix + "/" + name.removeprefix(prefix + "_")
            break
PREPARED_FILES.update(
    {
        "usgs/usgs_2010_2020_m7.csv": "deep_earth/usgs_source.csv",
        "processed/habitable/metal_minerals_subset.csv": "mineral.csv",
        "processed/hazard/events.jsonl": "hazard.jsonl",
        "answer_formats/geotooleval_fixed.jsonl": "answer_formats/geotooleval.jsonl",
        "answer_formats/geotooleval_fixed_system.txt": "answer_formats/system.txt",
        "processed/resource_manifest.json": "metadata/resources.json",
        "processed/assets/schema_crosswalk.csv": "ccs/schema_crosswalk.csv",
    }
)
RETAINED_FILES = {
    "gem": ["gem.jsonl"],
    "biodeeptime": [
        "biodeeptime/observations.csv",
        "biodeeptime/series.csv",
        "biodeeptime/zones.geojson",
    ],
    "ccs": [
        "ccs/zones.geojson",
        "ccs/crs_inventory.csv",
        "ccs/facilities.geojson",
        "ccs/facilities_webmercator.csv",
        "ccs/injection_monthly_geo.csv",
    ],
    "hazard": ["hazard.jsonl"],
    "deep_earth": [
        name for key, name in PREPARED_FILES.items() if key.startswith("processed/deep_earth/")
    ],
    "geotoolkg": [
        "geotoolkg/" + name
        for name in [
            "default__vector_store.json",
            "docstore.json",
            "graph_store.json",
            "image__vector_store.json",
            "import_manifest.json",
            "index_store.json",
        ]
    ],
}
DATASET_DIRECTORIES = {
    name: name
    for name in [
        "climate",
        "energy",
        "ocean",
        "biodeeptime",
        "ccs",
        "deep_earth",
        "geotoolkg",
        "answer_formats",
    ]
}
DATASET_DIRECTORIES.update({"geoquestions": "geoquestions", "remote_sensing": "earthbench"})


def prepared_name(dataset: str, relative: str) -> str:
    key = f"{dataset}/{relative}"
    if key in PREPARED_FILES:
        return PREPARED_FILES[key]
    key = f"processed/{key}"
    if key in PREPARED_FILES:
        return PREPARED_FILES[key]
    raise ValueError(f"Unknown prepared dataset resource: {dataset}/{relative}")


def prepared_locator(value: str) -> str:
    """Translate historical benchmark locators without changing their meaning."""
    text = value.replace("\\", "/")
    relative = text.removeprefix("data/")
    if relative in PREPARED_FILES:
        return "data/" + PREPARED_FILES[relative]
    if relative == "processed/earthbench":
        return "data/earthbench"
    if relative.startswith("processed/earthbench/"):
        return "data/earthbench/" + relative.removeprefix("processed/earthbench/")
    if relative.startswith("indexes/geotoolkg/"):
        return "data/geotoolkg/" + relative.removeprefix("indexes/geotoolkg/")
    name = PurePosixPath(text).name
    candidates = {new for old, new in PREPARED_FILES.items() if PurePosixPath(old).name == name}
    return "data/" + candidates.pop() if len(candidates) == 1 else value


def prepared_path(root, relative: str):
    """Resolve prepared data; ZIP extraction is handled by prepare commands."""
    from agentgs.config.paths import within

    return within(root / "data", relative)
