"""Prepare fixed raw inputs into empty roots and exercise scientific readers."""

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentgs.runtime.tools.context import tool_context
from agentgs.runtime.tools.dispatch import execute_domain

ROOT = Path(__file__).parents[3]
PREPARE = ROOT / "scripts/prepare"


def restore(domain, target):
    result = subprocess.run(
        [
            sys.executable,
            str(PREPARE / f"prepare_{domain}.py"),
            "--offline",
            "--cache-root",
            str(ROOT / "data/raw"),
            "--source-root",
            str(ROOT / "data"),
            "--output-root",
            str(target),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 2), result.stderr
    report = json.loads((target / f"data/metadata/prepare_{domain}.json").read_text())
    assert report["mode"] == "raw"
    return report


@pytest.mark.parametrize(
    "domain", ["mineral", "climate", "energy", "ocean", "biodeeptime", "ccs", "hazard"]
)
def test_fixed_raw_sources_prepare_identically(domain, tmp_path):
    first = restore(domain, tmp_path / "first")
    second = restore(domain, tmp_path / "second")
    assert first == second
    assert first["status"] == "complete"
    assert first["files"]
    for relative, digest in first["files"].items():
        for directory in ["first", "second"]:
            assert (
                hashlib.sha256((tmp_path / directory / relative).read_bytes()).hexdigest() == digest
            )


def test_full_population_energy_overlay_is_preserved(tmp_path):
    restore("energy", tmp_path)
    with tool_context(tmp_path, tmp_path / "outputs"):
        result = json.loads(
            execute_domain(
                "EnergyRegionalOverlaySummary",
                {
                    "region": "western_us",
                    "bbox": [-125, 30, -100, 50],
                },
            )
        )
    # Independent evidence from the supplied original EnergyResources-ToolQA package.
    assert result["facility_count"] == 1025
    assert result["mineral_site_count"] == 47182
    assert result["total_capacity_mw"] == pytest.approx(128757.9)


def test_original_mineral_subset_preserves_counts_rankings_and_row_order(tmp_path):
    restore("mineral", tmp_path)
    with tool_context(tmp_path, tmp_path / "outputs"):
        summary = json.loads(
            execute_domain("MineralResourcesCountrySummary", {"country": "United States"})
        )
        search = json.loads(
            execute_domain(
                "MineralResourcesSearchDeposits",
                {
                    "country": "United States",
                    "commodity": "Copper",
                    "limit": 3,
                },
            )
        )
        ranking = json.loads(
            execute_domain(
                "MineralResourcesRankCommodityCountries",
                {
                    "commodity": "Lithium",
                    "limit": 5,
                },
            )
        )
    # Independent frozen benchmark facts for the supplied original subset.
    assert summary["record_count"] == 6665
    assert summary["top_commodities"] == {"Gold": 2167, "Unknown": 650, "Iron": 499}
    assert [row["site_name"] for row in search["deposits"]] == [
        "Sunny Day Prospect",
        "Unnamed Occurrence",
        "Unnamed Occurrence",
    ]
    assert ranking["countries"] == {
        "United States": 5,
        "Canada": 2,
        "Ecuador": 2,
        "Bolivia": 1,
        "Mexico": 1,
    }


def test_hazard_jsonl_reader_preserves_numeric_missing_value_policy(tmp_path):
    restore("hazard", tmp_path)
    with tool_context(tmp_path, tmp_path / "outputs"):
        metadata = json.loads(execute_domain("NaturalHazardLoadEventTable", {}))
        maximum = json.loads(
            execute_domain("NaturalHazardSelectMaxRecord", {"measure": "Total Deaths"})
        )
    assert metadata["record_count"] == 10764
    assert len(metadata["columns"]) == 47
    assert "error" not in maximum
    assert maximum["event"]["Total Deaths"] > 0


@pytest.mark.parametrize(
    "start,end",
    [(1880, 1900), (1951, 1980), (1980, 1999), (2000, 2019), (2015, 2024), (2020, 2024)],
)
def test_gistemp_means_follow_the_available_nasa_release(start, end, tmp_path):
    import pandas as pd

    restore("climate", tmp_path)
    source = pd.read_csv(
        ROOT / "data/raw/climate/NASA_GISTEMP_current.csv", skiprows=1, na_values=["***"]
    )
    observed = source[source["Year"].between(start, end)]["J-D"].dropna()
    with tool_context(tmp_path, tmp_path / "outputs"):
        result = json.loads(
            execute_domain(
                "ClimateQueryPresentMean",
                {
                    "year_start": start,
                    "year_end": end,
                },
            )
        )
    assert result["n"] == len(observed)
    assert result["mean_temp_anomaly_c"] == pytest.approx(observed.mean())


def test_missing_retained_asset_fails_before_any_output_is_written(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PREPARE))
    common = importlib.import_module("_common")
    source = tmp_path / "source/biodeeptime"
    source.mkdir(parents=True)
    (source / "observations.csv").write_text("id,value\n1,42\n")
    with pytest.raises(FileNotFoundError, match="Retained dataset input"):
        common.existing_assets("biodeeptime", source.parent, tmp_path / "output")
    assert not (tmp_path / "output/data/biodeeptime/observations.csv").exists()


def test_ssp_temperature_is_bound_as_temperature_instead_of_horizon_year(tmp_path):
    from agentgs.agent.execution.bindings.helpers import _numeric_series

    restore("climate", tmp_path)
    with tool_context(tmp_path, tmp_path / "outputs"):
        result = json.loads(
            execute_domain(
                "ClimateQueryFutureSSP",
                {
                    "scenario": "SSP1-1.9",
                    "horizon_year": 2050,
                },
            )
        )
    assert result["estimate"]["warming_c_mid"] == 1.6
    assert _numeric_series([result]) == [[1.6]]


@pytest.mark.parametrize("domain", ["mineral", "ocean", "energy", "climate"])
def test_default_preparation_never_falls_back_to_repository_snapshots(domain, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(PREPARE / f"prepare_{domain}.py"),
            "--offline",
            "--cache-root",
            str(tmp_path / "empty-cache"),
            "--output-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    report = json.loads((tmp_path / f"data/metadata/prepare_{domain}.json").read_text())
    assert report["mode"] == "raw"
    assert report["status"] == "partial"
    assert report["missing"]
    assert report["files"] == {}


def test_generated_answer_contracts_preserve_public_formats(tmp_path):
    report = restore("answer_formats", tmp_path)
    assert report["status"] == "complete"
    rows = [
        json.loads(line)
        for line in (tmp_path / "data/answer_formats/geotooleval.jsonl").read_text().splitlines()
    ]
    originals = [
        json.loads(line) for line in (ROOT / "data/geotooleval.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(originals) == 603
    for row, original in zip(rows, originals, strict=True):
        assert set(row) == {"question_id", "question", "question_sha256", "answer_format"}
        assert row["question_id"] == original["题目序号"]
        assert row["question"] == original["题目正文"]
        assert row["answer_format"]["template"] == original["固定回答格式"]
        assert row["answer_format"]["notes"] == original["格式说明"]
    # Repeated records and precision must survive regeneration.
    by_id = {r["question_id"]: r["answer_format"] for r in rows}
    assert by_id[1]["decimals"] == 2
    assert by_id[287]["decimals"] == 3
    assert by_id[137]["repeat_lines"] == [0]
    assert by_id[137]["max_records"] == 5
    assert "repeat_lines" not in by_id[465]
    assert by_id[495]["allowed_options"] == ["A", "B", "C", "D"]


def test_geoquestions_regenerates_the_pinned_replay(tmp_path):
    from agentgs.datasets.geoquestions import snapshot_profile

    report = restore("geoquestions", tmp_path)
    assert report["status"] == "complete"
    snapshot = tmp_path / "data/geoquestions/queries.jsonl"
    rows = [json.loads(line) for line in snapshot.read_text().splitlines()]
    from agentgs.datasets.geoquestions import snapshot_bytes

    assert hashlib.sha256(snapshot_bytes(rows)).hexdigest() == snapshot_profile()["snapshot_sha256"]
    assert len(rows) == 40


def test_generated_data_are_ignored_and_original_assets_remain_visible():
    generated = [
        "data/answer_formats/geotooleval.jsonl",
        "data/geoquestions/queries.jsonl",
        "data/mineral.csv",
        "data/gem.jsonl",
        "data/climate/gistemp_annual.csv",
        "data/energy/gip_renewable_points.csv",
        "data/ocean/socat_region_trends.csv",
        "data/ccs/injection_daily.csv",
        "data/earthbench.zip",
        "data/metadata/prepare_all.json",
        "data/raw/climate/NASA_GISTEMP_current.csv",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *generated], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == generated
    from agentgs.datasets.layout import RETAINED_FILES
    from agentgs.datasets.references import REFERENCE_FILES

    originals = ["data/" + name for names in RETAINED_FILES.values() for name in names]
    originals += ["data/" + name for name in REFERENCE_FILES]
    originals += [
        "data/geotooleval.jsonl",
        "data/biodeeptime/zones.geojson",
        "data/hazard.jsonl",
        "data/geotoolkg/index_store.json",
        "data/metadata/geotooleval.json",
    ]
    for original in originals:
        result = subprocess.run(["git", "check-ignore", "--no-index", original], cwd=ROOT, capture_output=True)
        assert result.returncode == 1


def test_updated_cached_raw_source_is_used_and_its_actual_hash_is_recorded(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PREPARE))
    common = importlib.import_module("_common")
    path = tmp_path / "observations.csv"
    path.write_text("year,value\n2026,42\n")
    spec = {"observations.csv": {"sha256": "0" * 64, "urls": []}}
    assert common.fetch("observations.csv", tmp_path, True, spec) == path
    provenance = common.USED_SOURCES["observations.csv"]
    assert provenance["matches_preferred_version"] is False
    assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_old_download_falls_back_to_current_source(monkeypatch, tmp_path):
    import io
    import urllib.error

    monkeypatch.syspath_prepend(str(PREPARE))
    common = importlib.import_module("_common")
    calls = []

    def download(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/old"):
            raise urllib.error.HTTPError(request.full_url, 404, "old release removed", {}, None)
        return io.BytesIO(b"year,value\n2026,42\n")

    monkeypatch.setattr(common.urllib.request, "urlopen", download)
    spec = {
        "observations.csv": {
            "sha256": "0" * 64,
            "urls": ["https://source/old"],
            "latest_urls": ["https://source/current"],
        }
    }
    path = common.fetch("observations.csv", tmp_path, False, spec)
    assert calls == ["https://source/old", "https://source/current"]
    assert path.read_bytes() == b"year,value\n2026,42\n"
    provenance = json.loads(path.with_name(path.name + ".source.json").read_text())
    assert provenance["url"] == "https://source/current"
    assert provenance["matches_preferred_version"] is False


def test_world_bank_current_archive_is_parsed_without_old_filename(monkeypatch, tmp_path):
    import zipfile

    from agentgs.datasets.parsers.energy import latest_world_bank_values

    monkeypatch.syspath_prepend(str(PREPARE))
    domains = importlib.import_module("_domains")
    directory = tmp_path / "world_bank"
    directory.mkdir()
    csv = "\n\n\n\nCountry Name,Country Code,2023,2024\nChina,CHN,19.0,20.0\n"
    with zipfile.ZipFile(directory / "renewable_current.zip", "w") as archive:
        archive.writestr("API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_NEW.csv", csv)
    source = domains.world_bank_source(tmp_path, True)
    frame = latest_world_bank_values(source)
    assert frame.iloc[0]["renewable_consumption_year"] == 2024
    assert frame.iloc[0]["renewable_consumption_percent"] == 20.0


def test_geoquestions_reader_accepts_new_generation_and_still_checks_integrity(tmp_path):
    from agentgs.datasets.geoquestions import load_snapshot, snapshot_profile
    from agentgs.datasets.registry import DatasetRegistry

    directory = tmp_path / "data/geoquestions"
    directory.mkdir(parents=True)
    rows = [
        {"source_id": source_id, "result": {"rows": [["new observation"]]}}
        for source_id in snapshot_profile()["source_ids"]
    ]
    content = json.dumps(rows).encode()
    path = directory / "queries.jsonl"
    path.write_bytes(content)
    fingerprint = hashlib.sha256(content).hexdigest()
    assert fingerprint != snapshot_profile()["snapshot_sha256"]
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "outputs": {path.name: fingerprint},
            }
        )
    )
    assert load_snapshot(DatasetRegistry(tmp_path)) == (rows, fingerprint)
    path.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="data hash differs"):
        load_snapshot(DatasetRegistry(tmp_path))


def test_archive_preparation_keeps_ready_to_use_files_in_data(tmp_path):
    first = tmp_path / "first"
    result = subprocess.run(
        [
            sys.executable,
            str(PREPARE / "prepare_biodeeptime.py"),
            "--offline",
            "--source-root",
            str(ROOT / "data"),
            "--output-root",
            str(first),
            "--archive-threshold-mb",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    archive = first / "data/biodeeptime.zip"
    assert archive.is_file()
    assert (first / "data/biodeeptime/observations.csv").is_file()
    assert not (first / "data/raw/prepared").exists()
    second = tmp_path / "second"
    result = subprocess.run(
        [
            sys.executable,
            str(PREPARE / "prepare_biodeeptime.py"),
            "--archive",
            str(archive),
            "--output-root",
            str(second),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in ["observations.csv", "series.csv", "zones.geojson"]:
        assert (first / "data/biodeeptime" / name).read_bytes() == (
            second / "data/biodeeptime" / name
        ).read_bytes()
    assert not (second / "data/raw/prepared").exists()


def test_archive_blocks_path_traversal_and_checks_content(monkeypatch, tmp_path):
    import zipfile

    monkeypatch.syspath_prepend(str(PREPARE))
    formats = importlib.import_module("_formats")
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as bundle:
        bundle.writestr("../escape.csv", "invalid")
    with pytest.raises(ValueError, match="Unsafe"):
        formats.unpack_dataset(bad, tmp_path / "unpacked")
    with zipfile.ZipFile(bad, "w") as bundle:
        bundle.writestr(
            "_archive.json", json.dumps({"files": {"data.csv": {"sha256": "0" * 64, "size": 7}}})
        )
        bundle.writestr("data.csv", "changed")
    with pytest.raises(ValueError, match="content hash mismatch"):
        formats.unpack_dataset(bad, tmp_path / "unpacked")
    assert not (tmp_path / "unpacked/data.csv").exists()
