from __future__ import annotations

from typing import Any

from agentgs.agent.answering.evidence.core import (
    _casefold_value,
    _first_dict_with_keys,
    _number_text,
)


def _project_record_evidence(value: Any, question: str = "") -> str | None:
    """Project common nested records without exposing irrelevant JSON fields."""
    text = str(question or "").casefold()
    disaster_record = _first_dict_with_keys(value, {"disno."})
    if disaster_record is not None and any(
        (token in text for token in ("which disno", "disno.", "disno "))
    ):
        return str(_casefold_value(disaster_record, "DisNo."))
    earth_model = _first_dict_with_keys(
        value, {"moho_depth_km", "lab_depth_km", "semucb_dlnvs_100km_percent"}
    )
    if earth_model is not None and any(
        (token in text for token in ("moho", "lid-bottom", "dlnvs"))
    ):
        return f"Moho {_number_text(earth_model['moho_depth_km'], 2)} km; LID-BOTTOM {_number_text(earth_model['lab_depth_km'], 2)} km; dlnVs(100 km) {_number_text(earth_model['semucb_dlnvs_100km_percent'], 2)}%"
    event = _first_dict_with_keys(value, {"id", "mag", "place"})
    if event is not None:
        if "depth" in text or "deepest" in text:
            depth = event.get("depth")
            return f"{event['id']}, {float(depth):.1f} km, {event['place']}"
        return f"{event['id']}, M{event['mag']}, {event['place']}"
    deposit = _first_dict_with_keys(
        value, {"site_name", "country", "commod1", "latitude", "longitude"}
    )
    requests_full_deposit_record = (
        ("site_name" in text or "site name" in text)
        and "country" in text
        and ("commod1" in text)
        and ("coordinate" in text)
    )
    if deposit is not None and requests_full_deposit_record:
        commodity = deposit["commod1"]
        if commodity in (None, "", "None"):
            commodity = "nan"
        return f"{deposit['site_name']}; {deposit['country']}; {commodity}; ({deposit['latitude']}, {deposit['longitude']})"
    requests_site_and_coordinates = (
        "site_name" in text or "site name" in text
    ) and "coordinate" in text
    if deposit is not None and requests_site_and_coordinates:
        rendered = f"{deposit['site_name']}; ({deposit['latitude']}, {deposit['longitude']})"
        if "dep_id" in text and deposit.get("dep_id") is not None:
            rendered += f"; dep_id={deposit['dep_id']}"
        return rendered
    ocean_cross = _first_dict_with_keys(
        value, {"distance_km", "socat_fco2_uatm", "surface_tco2_umol_kg"}
    )
    if ocean_cross is not None:
        return f"{_number_text(ocean_cross['distance_km'], 2)} km; SOCAT {_number_text(ocean_cross['socat_fco2_uatm'], 3)} μatm; GLODAP surface TCO₂ {_number_text(ocean_cross['surface_tco2_umol_kg'], 1)} μmol/kg"
    nearest = _first_dict_with_keys(value, {"distance_km"})
    if nearest is not None and any((token in text for token in ("nearest", "closest"))):
        location = nearest.get("municipality")
        if location in (None, "", "nan"):
            location = "nan"
        return f"{location}, {_number_text(nearest['distance_km'], 2)} km"
    climate = _first_dict_with_keys(value, {"age_kyr_bp", "temp_anomaly_c"})
    if climate is not None:
        age = _number_text(climate["age_kyr_bp"], 2)
        temp = _number_text(_casefold_value(climate, "temp_anomaly_c"), 2)
        if "sea_level_m" in climate:
            return f"age≈{age} kyr BP, T≈{temp} °C, SL≈{_number_text(climate['sea_level_m'], 1)} m"
        return f"age≈{age} kyr BP, temp_anomaly≈{temp} °C"
    ocean_map = _first_dict_with_keys(
        value,
        {"year", "month", "weighted_mean_fco2_uatm", "weighted_mean_sst_c", "grid_cell_count"},
    )
    if ocean_map is not None:
        return f"{int(float(ocean_map['year']))}-{int(float(ocean_map['month'])):02d}; {_number_text(ocean_map['weighted_mean_fco2_uatm'], 3)} μatm; {_number_text(ocean_map['weighted_mean_sst_c'], 3)} °C; {int(float(ocean_map['grid_cell_count']))} cells"
    annual_ocean = _first_dict_with_keys(
        value, {"good_record_count", "mean_fco2_uatm", "mean_sst_c"}
    )
    if annual_ocean is not None:
        return f"{int(float(annual_ocean['good_record_count']))} records; {_number_text(annual_ocean['mean_fco2_uatm'], 3)} μatm; {_number_text(annual_ocean['mean_sst_c'], 3)} °C"
    ocean_dedup = _first_dict_with_keys(value, {"unique_good_sample_count", "duplicates_removed"})
    if ocean_dedup is not None:
        return f"{int(float(ocean_dedup['unique_good_sample_count']))} unique records; {int(float(ocean_dedup['duplicates_removed']))} duplicates removed"
    qc = _first_dict_with_keys(value, {"good_dic_talk_count", "good_dic_talk_percent"})
    if qc is not None:
        return f"{int(float(qc['good_dic_talk_count']))} records ({_number_text(qc['good_dic_talk_percent'], 2)}%)"
    overlay = _first_dict_with_keys(
        value, {"facility_count", "mineral_site_count", "total_capacity_mw"}
    )
    if overlay is not None:
        return f"{int(float(overlay['facility_count']))} renewable facilities, {_number_text(overlay['total_capacity_mw'], 2)} MW, {int(float(overlay['mineral_site_count']))} critical-mineral sites"
    return None
