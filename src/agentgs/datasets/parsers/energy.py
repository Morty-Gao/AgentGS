"""Shared energy input parsers; original column, selection and unit policies preserved."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from agentgs.datasets.references import load_reference

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

_REFERENCE = load_reference("reference/energy.json")
CAPITALS = _REFERENCE["capitals"]
REGION_BOXES = _REFERENCE["region_boxes"]
SELECTED_COUNTRY_CODES = _REFERENCE["selected_country_codes"]
TRANSITION_COUNTRIES = _REFERENCE["transition_countries"]
RENEWABLE_TYPES = set(_REFERENCE["renewable_types"])
CRITICAL_PATTERNS = _REFERENCE["critical_patterns"]
EARTH_RADIUS_KM = load_reference("reference/geography.json")["mean_earth_radius_km"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def haversine_vector(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    p1 = np.radians(lat)
    p2 = np.radians(lats)
    dlat = p2 - p1
    dlon = np.radians(lons - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def bbox_area_km2(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> float:
    lon_span = math.radians(max_lon - min_lon)
    return abs(
        EARTH_RADIUS_KM**2
        * lon_span
        * (math.sin(math.radians(max_lat)) - math.sin(math.radians(min_lat)))
    )


def as_float(value: object) -> float | None:
    if value in (None, "", "--", "-"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def clean_scalar(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def column_index(reference: str) -> int:
    letters = re.match("[A-Z]+", reference).group(0)
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - 64
    return result - 1


def xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    result: list[str] = []
    with archive.open("xl/sharedStrings.xml") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag == f"{{{MAIN_NS}}}si":
                result.append(
                    "".join((node.text or "" for node in element.iter(f"{{{MAIN_NS}}}t")))
                )
                element.clear()
    return result


def xlsx_sheet_target(archive: zipfile.ZipFile, wanted_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        relation.attrib["Id"]: relation.attrib["Target"]
        for relation in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        if sheet.attrib["name"] != wanted_name:
            continue
        relationship_id = sheet.attrib[f"{{{REL_NS}}}id"]
        target = targets[relationship_id].replace("\\", "/")
        if target.startswith("/"):
            return target.lstrip("/")
        return target if target.startswith("xl/") else f"xl/{target}"
    raise KeyError(wanted_name)


def iter_xlsx_rows(path: Path, sheet_name: str) -> Iterable[list[object | None]]:
    with zipfile.ZipFile(path) as archive:
        strings = xlsx_shared_strings(archive)
        target = xlsx_sheet_target(archive, sheet_name)
        with archive.open(target) as handle:
            for _, element in ET.iterparse(handle, events=("end",)):
                if element.tag != f"{{{MAIN_NS}}}row":
                    continue
                values: list[object | None] = []
                for cell in element.findall(f"{{{MAIN_NS}}}c"):
                    index = column_index(cell.attrib.get("r", "A1"))
                    while len(values) <= index:
                        values.append(None)
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(f"{{{MAIN_NS}}}v")
                    inline_node = cell.find(f"{{{MAIN_NS}}}is")
                    value: object | None = None
                    if (
                        cell_type == "s"
                        and value_node is not None
                        and (value_node.text is not None)
                    ):
                        value = strings[int(value_node.text)]
                    elif cell_type == "inlineStr" and inline_node is not None:
                        value = "".join(
                            (node.text or "" for node in inline_node.iter(f"{{{MAIN_NS}}}t"))
                        )
                    elif value_node is not None:
                        value = value_node.text
                    values[index] = value
                yield values
                element.clear()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_scalar(row.get(key)) for key in names})


def iter_gip_rows(path: Path, sheet_name: str) -> Iterable[list[object | None]]:
    if not str(path).endswith((".jsonl", ".jsonl.gz")):
        yield from iter_xlsx_rows(path, sheet_name)
        return
    found = False
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record") == "row" and record.get("sheet") == sheet_name:
                found = True
                yield record["values"]
    if not found:
        raise KeyError(f"GEM sheet is missing: {sheet_name}")


def process_gip(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = iter_gip_rows(path, "Power facilities")
    header = [str(value) if value is not None else "" for value in next(rows)]
    index = {name: position for position, name in enumerate(header)}
    required = [
        "Type",
        "Country/area",
        "Plant / Project name",
        "Capacity (MW)",
        "Status",
        "Latitude",
        "Longitude",
    ]
    missing = [name for name in required if name not in index]
    if missing:
        raise KeyError(missing)
    summaries: dict[tuple[str, str], dict] = {}
    top: dict[tuple[str, str], dict] = {}
    renewable_points: list[dict] = []
    for values in rows:

        def get(name: str) -> object | None:
            position = index[name]
            return values[position] if position < len(values) else None

        energy_type = str(get("Type") or "").strip().lower()
        country = str(get("Country/area") or "").strip()
        status = str(get("Status") or "").strip().lower()
        capacity = as_float(get("Capacity (MW)"))
        latitude = as_float(get("Latitude"))
        longitude = as_float(get("Longitude"))
        if not energy_type or not country or capacity is None or (capacity < 0):
            continue
        if status == "operating":
            key = (country, energy_type)
            summary = summaries.setdefault(
                key,
                {"country": country, "type": energy_type, "facility_count": 0, "capacity_mw": 0.0},
            )
            summary["facility_count"] += 1
            summary["capacity_mw"] += capacity
            record = {
                "country": country,
                "type": energy_type,
                "plant_name": str(get("Plant / Project name") or "").strip(),
                "capacity_mw": capacity,
                "latitude": latitude,
                "longitude": longitude,
            }
            if key not in top or capacity > top[key]["capacity_mw"]:
                top[key] = record
            if energy_type in RENEWABLE_TYPES and latitude is not None and (longitude is not None):
                if -90 <= latitude <= 90 and -180 <= longitude <= 180:
                    renewable_points.append(record)
    summary_frame = (
        pd.DataFrame(summaries.values()).sort_values(["country", "type"]).reset_index(drop=True)
    )
    summary_frame["capacity_mw"] = summary_frame["capacity_mw"].round(3)
    top_frame = pd.DataFrame(top.values()).sort_values(["country", "type"]).reset_index(drop=True)
    points_frame = pd.DataFrame(renewable_points)
    return (summary_frame, top_frame, points_frame)


def process_opsd(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usecols = [
        "electrical_capacity",
        "energy_source_level_2",
        "energy_source_level_3",
        "technology",
        "lon",
        "lat",
        "municipality",
        "country",
        "commissioning_date",
    ]
    summary_acc: dict[tuple[str, str], dict] = {}
    candidates: list[pd.DataFrame] = []
    nearest: dict[tuple[str, str], dict] = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200000, low_memory=False):
        chunk["electrical_capacity"] = pd.to_numeric(chunk["electrical_capacity"], errors="coerce")
        chunk["lat"] = pd.to_numeric(chunk["lat"], errors="coerce")
        chunk["lon"] = pd.to_numeric(chunk["lon"], errors="coerce")
        chunk = chunk[
            chunk["country"].isin(SELECTED_COUNTRY_CODES)
            & chunk["lat"].between(-90, 90)
            & chunk["lon"].between(-180, 180)
            & chunk["electrical_capacity"].gt(0)
        ].copy()
        if chunk.empty:
            continue
        grouped = chunk.groupby(["country", "energy_source_level_2"], dropna=False)[
            "electrical_capacity"
        ].agg(["count", "sum"])
        for (country, source), row in grouped.iterrows():
            key = (str(country), str(source))
            item = summary_acc.setdefault(
                key,
                {
                    "country": key[0],
                    "energy_source": key[1],
                    "facility_count": 0,
                    "capacity_mw": 0.0,
                },
            )
            item["facility_count"] += int(row["count"])
            item["capacity_mw"] += float(row["sum"])
        selected = chunk[
            chunk["energy_source_level_2"].isin(_REFERENCE["opsd_landmark_sources"])
        ].copy()
        if not selected.empty:
            candidates.append(
                selected.sort_values("electrical_capacity", ascending=False)
                .groupby(["country", "energy_source_level_2"], group_keys=False)
                .head(8)
            )
        for country, (capital_lat, capital_lon, capital_en, capital_zh) in CAPITALS.items():
            country_rows = chunk[chunk["country"] == country]
            for source in _REFERENCE["opsd_capital_sources"]:
                subset = country_rows[country_rows["energy_source_level_2"] == source]
                if subset.empty:
                    continue
                distances = haversine_vector(
                    capital_lat,
                    capital_lon,
                    subset["lat"].to_numpy(dtype=float),
                    subset["lon"].to_numpy(dtype=float),
                )
                position = int(np.argmin(distances))
                row = subset.iloc[position]
                record = {
                    "country": country,
                    "capital_en": capital_en,
                    "capital_zh": capital_zh,
                    "energy_source": source,
                    "municipality": str(row.get("municipality") or ""),
                    "technology": str(row.get("technology") or ""),
                    "capacity_mw": float(row["electrical_capacity"]),
                    "latitude": float(row["lat"]),
                    "longitude": float(row["lon"]),
                    "distance_km": float(distances[position]),
                }
                key = (country, source)
                if key not in nearest or record["distance_km"] < nearest[key]["distance_km"]:
                    nearest[key] = record
    summary = (
        pd.DataFrame(summary_acc.values())
        .sort_values(["country", "energy_source"])
        .reset_index(drop=True)
    )
    summary["capacity_mw"] = summary["capacity_mw"].round(6)
    candidate_frame = pd.concat(candidates, ignore_index=True)
    landmarks = (
        candidate_frame.sort_values("electrical_capacity", ascending=False)
        .groupby(["country", "energy_source_level_2"], group_keys=False)
        .head(3)
        .rename(
            columns={
                "electrical_capacity": "capacity_mw",
                "energy_source_level_2": "energy_source",
                "lat": "latitude",
                "lon": "longitude",
            }
        )
        .reset_index(drop=True)
    )
    nearest_frame = (
        pd.DataFrame(nearest.values())
        .sort_values(["country", "energy_source"])
        .reset_index(drop=True)
    )
    return (summary, landmarks, nearest_frame)


def identify_commodity(row: pd.Series) -> str | None:
    text = " | ".join(
        (str(row.get(column) or "") for column in ("commod1", "commod2", "commod3"))
    ).lower()
    for label, pattern in CRITICAL_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return None


def process_mrds(path: Path) -> pd.DataFrame:
    usecols = [
        "dep_id",
        "site_name",
        "latitude",
        "longitude",
        "country",
        "state",
        "commod1",
        "commod2",
        "commod3",
        "dev_stat",
        "prod_size",
        "score",
    ]
    frames: list[pd.DataFrame] = []
    pattern = "|".join((f"(?:{value})" for value in CRITICAL_PATTERNS.values()))
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100000, low_memory=False):
        commodity_text = (
            chunk[["commod1", "commod2", "commod3"]].fillna("").astype(str).agg(" | ".join, axis=1)
        )
        subset = chunk[
            commodity_text.str.contains(pattern, case=False, regex=True, na=False)
        ].copy()
        subset["latitude"] = pd.to_numeric(subset["latitude"], errors="coerce")
        subset["longitude"] = pd.to_numeric(subset["longitude"], errors="coerce")
        subset = subset[
            subset["latitude"].between(-90, 90) & subset["longitude"].between(-180, 180)
        ]
        if subset.empty:
            continue
        subset["commodity_group"] = subset.apply(identify_commodity, axis=1)
        frames.append(subset)
    result = pd.concat(frames, ignore_index=True)
    result = result[result["commodity_group"].notna()].drop_duplicates("dep_id")
    return result.sort_values(["commodity_group", "country", "dep_id"]).reset_index(drop=True)


def process_capacity_timeseries(path: Path) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    frame = pd.read_csv(path, low_memory=False)
    frame["day"] = pd.to_datetime(frame["day"], errors="coerce")
    frame = frame.dropna(subset=["day"]).sort_values("day")
    year_end = frame.groupby(frame["day"].dt.year, as_index=False).tail(1).copy()
    selected_series = _REFERENCE["capacity_series"]
    final_year = int(year_end["day"].dt.year.max())
    single_rows: list[dict] = []
    growth_rows: list[dict] = []
    for column, label_en, label_zh in selected_series:
        if column not in year_end.columns:
            continue
        valid = year_end[["day", column]].dropna().copy()
        final = valid[valid["day"].dt.year == final_year]
        if final.empty:
            final = valid.tail(1)
        final_record = final.iloc[-1]
        single_rows.append(
            {
                "series": column,
                "label_en": label_en,
                "label_zh": label_zh,
                "year": int(final_record["day"].year),
                "capacity_mw": float(final_record[column]),
            }
        )
        candidates = valid[valid["day"].dt.year.isin([2010, 2015, final_year])]
        start = candidates[candidates["day"].dt.year == 2010]
        if start.empty or float(start.iloc[-1][column]) <= 0:
            start = candidates[candidates["day"].dt.year == 2015]
        end = candidates[candidates["day"].dt.year == final_year]
        if start.empty or end.empty:
            continue
        start_row = start.iloc[-1]
        end_row = end.iloc[-1]
        start_value = float(start_row[column])
        end_value = float(end_row[column])
        growth_rows.append(
            {
                "series": column,
                "label_en": label_en,
                "label_zh": label_zh,
                "start_year": int(start_row["day"].year),
                "end_year": int(end_row["day"].year),
                "start_capacity_mw": start_value,
                "end_capacity_mw": end_value,
                "absolute_growth_mw": end_value - start_value,
                "percent_growth": (end_value / start_value - 1) * 100 if start_value else None,
            }
        )
    snapshot_columns = ["day"] + [
        item[0] for item in selected_series if item[0] in year_end.columns
    ]
    return (year_end[snapshot_columns], single_rows, growth_rows)


def latest_world_bank_values(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=4, low_memory=False)
    year_columns = [column for column in frame.columns if re.fullmatch("\\d{4}", str(column))]
    rows: list[dict] = []
    for _, row in frame.iterrows():
        country = str(row["Country Name"])
        if country not in TRANSITION_COUNTRIES:
            continue
        latest_year = None
        latest_value = None
        for year in reversed(year_columns):
            value = as_float(row[year])
            if value is not None:
                latest_year = int(year)
                latest_value = value
                break
        if latest_year is not None:
            rows.append(
                {
                    "country": country,
                    "renewable_consumption_year": latest_year,
                    "renewable_consumption_percent": latest_value,
                }
            )
    return pd.DataFrame(rows)


def population_values(path: Path, countries: list[str], target_year: int = 2021) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame["Entity"].isin(countries) & (frame["Year"] == target_year)].copy()
    frame["population"] = frame["Population"].fillna(frame["Population (Projected)"])
    return frame[["Entity", "Code", "Year", "population"]].rename(
        columns={"Entity": "country", "Code": "country_code", "Year": "population_year"}
    )


def build_transition_metrics(
    gip_summary: pd.DataFrame, wb: pd.DataFrame, population: pd.DataFrame
) -> pd.DataFrame:
    renewable = (
        gip_summary[gip_summary["type"].isin(RENEWABLE_TYPES)]
        .groupby("country", as_index=False)["capacity_mw"]
        .sum()
    )
    renewable = renewable.rename(columns={"capacity_mw": "operating_renewable_capacity_mw"})
    result = renewable.merge(population, on="country", how="inner").merge(
        wb, on="country", how="inner"
    )
    result["renewable_capacity_mw_per_million_people"] = (
        result["operating_renewable_capacity_mw"] / result["population"] * 1000000
    )
    return result.sort_values("country").reset_index(drop=True)
