"""Hazard tool argument declarations; field policies are preserved."""

from typing import Annotated

from pydantic import Field, model_validator

from .base import Filters, Text, ToolArguments


class HazardArguments(ToolArguments):
    @model_validator(mode="after")
    def check_filter_records(self):
        if self.tool_name == "NaturalHazardFilterRecords" and not self.filters:
            raise ValueError("filters must be a non-empty exact-match object")
        return self


def register(add):
    common = {"source": Text, "sheet": Text, "filters": Filters, "filtered_records_path": Text}
    add("NaturalHazardLoadEventTable", {"source": Text, "sheet": Text}, ("source", "sheet"))
    add("NaturalHazardFilterRecords", common, ("filters",))
    add("NaturalHazardCountRecords", common)
    for suffix in ("SumField", "SelectMaxRecord"):
        add("NaturalHazard" + suffix, {**common, "measure": Text}, ("measure",))
    add("NaturalHazardGroupAggregate", {**common, "group_by": Text, "measure": Text}, ("group_by",))
    for suffix in ("BuildStartDate", "SelectLatest"):
        add(
            "NaturalHazard" + suffix,
            {**common, "date_fields": Annotated[list[Text], Field(min_length=3, max_length=3)]},
        )
