"""Shared ocean input parsers; original column, selection and unit policies preserved."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
OCEAN_BOXES = [('north_pacific_west', '西北太平洋', 120.0, 20.0, 180.0, 60.0), ('north_pacific_east', '东北太平洋', -180.0, 20.0, -120.0, 60.0), ('tropical_pacific_west', '西部热带太平洋', 120.0, -10.0, 180.0, 20.0), ('tropical_pacific_east', '东部热带太平洋', -180.0, -10.0, -80.0, 20.0), ('north_atlantic', '北大西洋', -80.0, 20.0, 10.0, 65.0), ('tropical_atlantic', '热带大西洋', -60.0, -15.0, 20.0, 20.0), ('indian_ocean', '印度洋', 20.0, -40.0, 120.0, 25.0), ('southern_ocean', '南大洋', -180.0, -65.0, 180.0, -40.0), ('arctic_ocean', '北冰洋', -180.0, 65.0, 180.0, 85.0), ('mediterranean', '地中海', -6.0, 30.0, 36.0, 46.0)]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
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
    return abs(EARTH_RADIUS_KM ** 2 * lon_span * (math.sin(math.radians(max_lat)) - math.sin(math.radians(min_lat))))

def clean_scalar(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value

def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows for {path}')
    names = fieldnames or list(rows[0].keys())
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_scalar(row.get(key)) for key in names})

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float('nan')
    return float(np.average(values[mask].to_numpy(dtype=float), weights=weights[mask].to_numpy(dtype=float)))

def evenly_select(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy()
    positions = np.linspace(0, len(frame) - 1, count).round().astype(int)
    return frame.iloc[positions].copy()

def process_socat_grid(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [column.strip() for column in frame.columns]
    rename = {'valid_tax_date': 'date', 'valid_lat_month': 'latitude', 'valid_lon_month': 'longitude', 'valid_count_ncruise_month': 'cruise_count', 'valid_fco2_count_nobs_month': 'fco2_nobs', 'valid_fco2_ave_weighted_month': 'fco2_uatm', 'valid_sst_count_nobs_month': 'sst_nobs', 'valid_sst_ave_weighted_month': 'sst_c', 'valid_salinity_count_nobs_month': 'salinity_nobs', 'valid_salinity_ave_weighted_month': 'salinity'}
    frame = frame[list(rename)].rename(columns=rename)
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
    for column in ['latitude', 'longitude', 'cruise_count', 'fco2_nobs', 'fco2_uatm', 'sst_nobs', 'sst_c', 'salinity_nobs', 'salinity']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    for column in ['fco2_uatm', 'sst_c', 'salinity']:
        frame.loc[frame[column] < -1e+20, column] = np.nan
    frame = frame.dropna(subset=['date', 'latitude', 'longitude', 'fco2_uatm'])
    frame['year'] = frame['date'].dt.year.astype(int)
    frame['month'] = frame['date'].dt.month.astype(int)
    samples: list[pd.Series] = []
    for _, _, min_lon, min_lat, max_lon, max_lat in OCEAN_BOXES:
        sample_pool = frame[frame['longitude'].between(min_lon, max_lon) & frame['latitude'].between(min_lat, max_lat) & frame['fco2_nobs'].ge(10) & frame['sst_c'].notna()].sort_values('fco2_nobs', ascending=False)
        if sample_pool.empty:
            raise AssertionError((min_lon, min_lat, max_lon, max_lat))
        samples.append(sample_pool.iloc[0])
    sample_frame = pd.DataFrame(samples).reset_index(drop=True)
    trend_rows: list[dict] = []
    map_rows: list[dict] = []
    for region_key, region_zh, min_lon, min_lat, max_lon, max_lat in OCEAN_BOXES:
        region = frame[frame['longitude'].between(min_lon, max_lon) & frame['latitude'].between(min_lat, max_lat)].copy()
        if region.empty:
            raise AssertionError(region_key)
        annual_rows: list[dict] = []
        for year, group in region.groupby('year'):
            total_obs = float(group['fco2_nobs'].fillna(0).sum())
            if total_obs < 20:
                continue
            annual_rows.append({'year': int(year), 'grid_cell_month_count': int(len(group)), 'fco2_observation_count': int(total_obs), 'weighted_mean_fco2_uatm': weighted_mean(group['fco2_uatm'], group['fco2_nobs']), 'weighted_mean_sst_c': weighted_mean(group['sst_c'], group['sst_nobs'])})
        annual = pd.DataFrame(annual_rows).dropna(subset=['weighted_mean_fco2_uatm']).sort_values('year')
        if len(annual) < 2:
            raise AssertionError(f'Too few annual rows for {region_key}')
        start = annual.iloc[0]
        end = annual.iloc[-1]
        trend_rows.append({'region': region_key, 'region_zh': region_zh, 'start_year': int(start.year), 'end_year': int(end.year), 'start_fco2_uatm': float(start.weighted_mean_fco2_uatm), 'end_fco2_uatm': float(end.weighted_mean_fco2_uatm), 'fco2_change_uatm': float(end.weighted_mean_fco2_uatm - start.weighted_mean_fco2_uatm), 'start_observation_count': int(start.fco2_observation_count), 'end_observation_count': int(end.fco2_observation_count), 'bbox': [min_lon, min_lat, max_lon, max_lat]})
        monthly_rows: list[dict] = []
        for (year, month), group in region.groupby(['year', 'month']):
            total_obs = float(group['fco2_nobs'].fillna(0).sum())
            if total_obs <= 0:
                continue
            monthly_rows.append({'year': int(year), 'month': int(month), 'grid_cell_count': int(len(group)), 'fco2_observation_count': int(total_obs), 'weighted_mean_fco2_uatm': weighted_mean(group['fco2_uatm'], group['fco2_nobs']), 'weighted_mean_sst_c': weighted_mean(group['sst_c'], group['sst_nobs'])})
        monthly = pd.DataFrame(monthly_rows).sort_values(['fco2_observation_count', 'year', 'month'], ascending=[False, True, True])
        best = monthly.iloc[0]
        map_rows.append({'region': region_key, 'region_zh': region_zh, 'year': int(best.year), 'month': int(best.month), 'grid_cell_count': int(best.grid_cell_count), 'fco2_observation_count': int(best.fco2_observation_count), 'weighted_mean_fco2_uatm': float(best.weighted_mean_fco2_uatm), 'weighted_mean_sst_c': float(best.weighted_mean_sst_c), 'bbox': [min_lon, min_lat, max_lon, max_lat]})
    return (frame, sample_frame, pd.DataFrame(trend_rows), pd.DataFrame(map_rows))

def process_socat_north_pacific(path: Path) -> pd.DataFrame:
    usecols = ['Expocode', 'QC_Flag', 'yr', 'mon', 'longitude [dec.deg.E]', 'latitude [dec.deg.N]', 'sample_depth [m]', 'sal', 'SST [deg.C]', 'fCO2rec [uatm]', 'fCO2rec_flag']
    accum: dict[int, dict] = {}
    for chunk in pd.read_csv(path, sep='\t', skiprows=1531, usecols=usecols, chunksize=250000, low_memory=False):
        for column in ['yr', 'mon', 'longitude [dec.deg.E]', 'latitude [dec.deg.N]', 'sample_depth [m]', 'sal', 'SST [deg.C]', 'fCO2rec [uatm]', 'fCO2rec_flag']:
            chunk[column] = pd.to_numeric(chunk[column], errors='coerce')
        good = chunk[(chunk['fCO2rec_flag'] == 2) & chunk['fCO2rec [uatm]'].notna() & chunk['longitude [dec.deg.E]'].notna() & chunk['latitude [dec.deg.N]'].notna()].copy()
        if good.empty:
            continue
        grouped = good.groupby('yr')
        for year, group in grouped:
            year = int(year)
            item = accum.setdefault(year, {'year': year, 'good_record_count': 0, 'fco2_sum': 0.0, 'fco2_count': 0, 'sst_sum': 0.0, 'sst_count': 0, 'salinity_sum': 0.0, 'salinity_count': 0, 'expocodes': set()})
            item['good_record_count'] += int(len(group))
            values = group['fCO2rec [uatm]'].dropna()
            item['fco2_sum'] += float(values.sum())
            item['fco2_count'] += int(values.count())
            sst = group['SST [deg.C]'].dropna()
            item['sst_sum'] += float(sst.sum())
            item['sst_count'] += int(sst.count())
            salinity = group['sal'].dropna()
            item['salinity_sum'] += float(salinity.sum())
            item['salinity_count'] += int(salinity.count())
            item['expocodes'].update(group['Expocode'].dropna().astype(str).unique().tolist())
    rows: list[dict] = []
    for year, item in sorted(accum.items()):
        rows.append({'year': year, 'good_record_count': item['good_record_count'], 'cruise_count': len(item['expocodes']), 'mean_fco2_uatm': item['fco2_sum'] / item['fco2_count'], 'mean_sst_c': item['sst_sum'] / item['sst_count'] if item['sst_count'] else None, 'mean_salinity': item['salinity_sum'] / item['salinity_count'] if item['salinity_count'] else None})
    frame = pd.DataFrame(rows)
    eligible = frame[frame['good_record_count'] >= 1000].sort_values('year')
    return evenly_select(eligible, 10).reset_index(drop=True)

def row_record(row: object) -> dict:
    return {'depth_m': float(row.G2depth), 'temperature_c': None if row.G2temperature <= -9000 else float(row.G2temperature), 'salinity': None if row.G2salinity <= -9000 else float(row.G2salinity), 'tco2_umol_kg': None if row.G2tco2 <= -9000 or row.G2tco2f != 2 else float(row.G2tco2), 'talk_umol_kg': None if row.G2talk <= -9000 or row.G2talkf != 2 else float(row.G2talk), 'ph_total_25c': None if row.G2phts25p0 <= -9000 or row.G2phts25p0f != 2 else float(row.G2phts25p0)}

def process_glodap(path: Path) -> pd.DataFrame:
    usecols = ['G2expocode', 'G2station', 'G2year', 'G2latitude', 'G2longitude', 'G2depth', 'G2temperature', 'G2salinity', 'G2tco2', 'G2tco2f', 'G2talk', 'G2talkf', 'G2phts25p0', 'G2phts25p0f']
    profiles: dict[tuple, dict] = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=150000, low_memory=False):
        for column in usecols[1:]:
            chunk[column] = pd.to_numeric(chunk[column], errors='coerce')
        chunk = chunk[chunk['G2latitude'].between(-90, 90) & chunk['G2longitude'].between(-180, 360) & chunk['G2depth'].ge(0) & chunk['G2tco2'].gt(-9000) & chunk['G2tco2f'].eq(2)]
        for row in chunk.itertuples(index=False):
            longitude = float(row.G2longitude)
            if longitude > 180:
                longitude -= 360
            key = (str(row.G2expocode), int(row.G2station), int(row.G2year), round(float(row.G2latitude), 5), round(longitude, 5))
            item = profiles.setdefault(key, {'expocode': key[0], 'station': key[1], 'year': key[2], 'latitude': key[3], 'longitude': key[4], 'surface': None, 'deep': None})
            record = row_record(row)
            if item['surface'] is None or record['depth_m'] < item['surface']['depth_m']:
                item['surface'] = record
            if item['deep'] is None or record['depth_m'] > item['deep']['depth_m']:
                item['deep'] = record
    rows: list[dict] = []
    for item in profiles.values():
        surface = item['surface']
        deep = item['deep']
        if surface is None or deep is None or surface['depth_m'] > 25 or (deep['depth_m'] < 1000):
            continue
        rows.append({'expocode': item['expocode'], 'station': item['station'], 'year': item['year'], 'latitude': item['latitude'], 'longitude': item['longitude'], 'surface_depth_m': surface['depth_m'], 'surface_tco2_umol_kg': surface['tco2_umol_kg'], 'surface_temperature_c': surface['temperature_c'], 'surface_salinity': surface['salinity'], 'surface_talk_umol_kg': surface['talk_umol_kg'], 'surface_ph_total_25c': surface['ph_total_25c'], 'deep_depth_m': deep['depth_m'], 'deep_tco2_umol_kg': deep['tco2_umol_kg'], 'deep_temperature_c': deep['temperature_c'], 'deep_salinity': deep['salinity'], 'deep_talk_umol_kg': deep['talk_umol_kg'], 'deep_ph_total_25c': deep['ph_total_25c'], 'tco2_vertical_difference_umol_kg': deep['tco2_umol_kg'] - surface['tco2_umol_kg']})
    frame = pd.DataFrame(rows).sort_values(['year', 'expocode', 'station'])
    diverse = frame.groupby('expocode', group_keys=False).head(1).reset_index(drop=True)
    return evenly_select(diverse, 10).reset_index(drop=True)

def process_snapo(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_excel(path, header=28, skiprows=[29])
    numeric_columns = ['Day', 'Month', 'Year', 'Hour', 'Minute', 'Seconde', 'Latitude', 'Longitude', 'Depth', 'TALK', 'TALK-Flag', 'DIC', 'DIC-Flag', 'SALINITY', 'SAL-Flag', 'TEMP', 'TEMP-Flag', 'N-sample-cruise', 'Sampling', 'Version', 'n-sample']
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['Cruise', 'Latitude', 'Longitude', 'Depth'])
    frame = frame[frame['Latitude'].between(-90, 90) & frame['Longitude'].between(-180, 180)]
    frame['all_carbon_good'] = (frame['DIC-Flag'] == 2) & (frame['TALK-Flag'] == 2)
    key_columns = ['Cruise', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Seconde', 'Latitude', 'Longitude', 'Depth']
    summaries: list[dict] = []
    sample_rows: list[dict] = []
    for cruise, group in frame.groupby('Cruise'):
        good = group[group['all_carbon_good']]
        if len(good) < 20:
            continue
        unique_good = good.drop_duplicates(key_columns)
        first = unique_good.sort_values(['Year', 'Month', 'Day', 'Depth']).iloc[0]
        summaries.append({'cruise': str(cruise), 'year_min': int(group['Year'].min()), 'year_max': int(group['Year'].max()), 'record_count': int(len(group)), 'good_dic_talk_count': int(len(good)), 'good_dic_talk_percent': float(len(good) / len(group) * 100), 'unique_good_sample_count': int(len(unique_good)), 'duplicates_removed': int(len(good) - len(unique_good)), 'latitude': float(first['Latitude']), 'longitude': float(first['Longitude']), 'sample_depth_m': float(first['Depth']), 'dic_umol_kg': float(first['DIC']), 'talk_umol_kg': float(first['TALK'])})
        sample_rows.append({'cruise': str(cruise), 'latitude': float(first['Latitude']), 'longitude': float(first['Longitude']), 'depth_m': float(first['Depth']), 'dic_umol_kg': float(first['DIC']), 'talk_umol_kg': float(first['TALK'])})
    summary_frame = pd.DataFrame(summaries).sort_values(['record_count', 'cruise'], ascending=[False, True])
    qc_selected = evenly_select(summary_frame.reset_index(drop=True), 10).reset_index(drop=True)
    dedup_pool = summary_frame.sort_values(['duplicates_removed', 'record_count'], ascending=[False, False]).reset_index(drop=True)
    dedup_selected = dedup_pool.head(10).copy()
    sample_frame = pd.DataFrame(sample_rows)
    return (qc_selected, dedup_selected, sample_frame)
