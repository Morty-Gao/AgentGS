import pytest


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="0.0.0"')
    data = tmp_path / "data" / "deep_earth"
    data.mkdir(parents=True)
    (data / "usgs_source.csv").write_text(
        "id,time,mag,place\na,2010-01-01T00:00:00Z,7.0,A\nb,2011-01-01T00:00:00Z,8.0,B\nc,2011-02-01T00:00:00Z,8.5,C\n"
    )
    return tmp_path
