from pydantic import Field, model_validator

from agentgs.contracts.models import Model


class CatalogQuery(Model):
    start_year: int = Field(default=2010, ge=1900, le=2200)
    end_year: int = Field(default=2020, ge=1900, le=2200)
    min_magnitude: float = Field(default=7, ge=-5, le=12, allow_inf_nan=False)
    limit: int = Field(default=20, ge=1, le=1000)

    @model_validator(mode="after")
    def ordered(self):
        if self.start_year > self.end_year:
            raise ValueError("Year range is reversed")
        return self


class EventQuery(Model):
    event_id: str = Field(min_length=1)


def query_catalog(p, datasets):
    rows = datasets.csv_rows("usgs", "usgs_2010_2020_m7.csv")
    selected = [
        row
        for row in rows
        if p.start_year <= int(row["time"][:4]) <= p.end_year
        and float(row["mag"]) >= p.min_magnitude
    ]
    selected.sort(key=lambda x: (-float(x["mag"]), x["id"]))
    return {
        "source": "USGS frozen 2010-2020 M7+ catalog",
        "count": len(selected),
        "records": selected[: p.limit],
        "truncated": len(selected) > p.limit,
    }


def get_event(p, datasets):
    rows = datasets.csv_rows("usgs", "usgs_2010_2020_m7.csv")
    matches = [row for row in rows if row["id"] == p.event_id]
    if not matches:
        raise ValueError(f"No event with id {p.event_id}")
    return {"source": "USGS frozen 2010-2020 M7+ catalog", "event": matches[0]}
