"""Shared field types, input-form validation and argument model construction."""

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, create_model, model_validator

Text = Annotated[str, Field(min_length=1)]
Number = Annotated[float, Field(allow_inf_nan=False)]
Numbers = Annotated[list[Number], Field(min_length=1)]
Paths = Annotated[list[Text], Field(min_length=1)]
BBox = Annotated[list[Number], Field(min_length=4, max_length=4)]
Identifier = Text | int
Identifiers = Annotated[list[Identifier], Field(min_length=2, max_length=2)]
Latitude = Annotated[Number, Field(ge=-90, le=90)]
Longitude = Annotated[Number, Field(ge=-180, le=180)]
PositiveInt = Annotated[int, Field(ge=1)]
Filters = dict[str, JsonValue]


class RasterPair(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_green_path: Text
    input_red_path: Text
    output_filename: Text | None = None


class ArtifactSelector(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    artifact_id: Text
    field: str = Field(description="Exact registered JSON pointer in the observed artifact")


class ArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: ArtifactSelector = Field(alias="_rs_source")


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    tool_name: ClassVar[str] = ""
    alternatives: ClassVar[tuple[tuple[str, ...], ...]] = ()
    exclusive_groups: ClassVar[tuple[tuple[str, ...], ...]] = ()

    @model_validator(mode="after")
    def check_input_forms(self):
        data = self.model_dump(exclude_unset=True)

        def present(key):
            return key in data and data[key] is not None and data[key] != [] and data[key] != ""

        if self.alternatives and not any(
            all(present(key) for key in group) for group in self.alternatives
        ):
            options = " OR ".join(" + ".join(group) for group in self.alternatives)
            raise ValueError(f"{self.tool_name} requires {options}")
        if self.exclusive_groups:
            selected = [
                group for group in self.exclusive_groups if any(present(key) for key in group)
            ]
            if len(selected) != 1 or not all(present(key) for key in selected[0]):
                raise ValueError("Supply exactly one complete scalar or batch input form")
            if len(selected[0]) == 2:
                a, b = (data[key] for key in selected[0])
                if isinstance(a, list) and isinstance(b, list) and len(a) != len(b):
                    raise ValueError("Paired arrays must have equal non-zero lengths")
        return self


def argument_field(annotation, description, *, required=False, **constraints):
    return (
        annotation if required else annotation | None,
        Field(default=... if required else None, description=description, **constraints),
    )


def argument_fields(types, required=()):
    return {
        name: argument_field(annotation, name.replace("_", " "), required=name in required)
        for name, annotation in types.items()
    }


def create_argument_model(
    name, types, required=(), alternatives=(), exclusive=(), *, base=ToolArguments
):
    result = create_model(
        name + "Arguments",
        __base__=base,
        __module__="agentgs.contracts.tool_inputs",
        **argument_fields(types, required),
    )
    result.tool_name = name
    result.alternatives = tuple(tuple(group) for group in alternatives)
    result.exclusive_groups = tuple(tuple(group) for group in exclusive)
    return result
