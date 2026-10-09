import pytest

from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.tools.domains.earthquake import CatalogQuery, query_catalog
from agentgs.runtime.tools.domains.geoscience import BBox, Points, distance


def test_catalog_coverage_filter_and_order(project):
    result = query_catalog(
        CatalogQuery(start_year=2011, end_year=2011, min_magnitude=8, limit=1),
        DatasetRegistry(project),
    )
    assert result["count"] == 2 and result["records"][0]["id"] == "c" and result["truncated"]


async def test_invalid_paths_coordinates_and_bbox(project):
    with pytest.raises(ValueError):
        DatasetRegistry(project).resolve("usgs", "../../pyproject.toml")
    with pytest.raises(ValueError):
        await LocalArtifactStore(project / "artifacts").write_json("../escape", {}, "test")
    with pytest.raises(ValueError):
        Points(lat1=91, lon1=0, lat2=0, lon2=0)
    with pytest.raises(ValueError):
        BBox(west=170, south=0, east=-170, north=1)
    assert distance(Points(lat1=0, lon1=0, lat2=0, lon2=0))["distance_km"] == 0
