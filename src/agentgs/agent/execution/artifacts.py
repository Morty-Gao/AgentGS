"""Typed artifact tracking for Skill-guided GeoToolAgent execution.

The registry is intentionally independent of GeoToolEval question identifiers.  It
discovers resources from ordinary locators, records producer/consumer provenance,
and exposes compact summaries to the planner so large raster manifests do not have
to be copied into every model prompt.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agentgs.agent.execution.bindings import split_question_resource
from agentgs.contracts.resources import resolve_resource_path

_PATH_SUFFIXES = {
    ".csv": "Table",
    ".xlsx": "Table",
    ".xls": "Table",
    ".json": "Table",
    ".jsonl": "Table",
    ".geojson": "Vector",
    ".shp": "Vector",
    ".gpkg": "Vector",
    ".tif": "Raster",
    ".tiff": "Raster",
    ".npy": "Raster",
    ".png": "Image",
    ".html": "Map",
}


@dataclass
class ArtifactRecord:
    artifact_id: str
    semantic_type: str
    value: Any
    producer: str
    producer_step: str = ""
    role: str = "intermediate"
    metadata: dict[str, Any] = field(default_factory=dict)
    valid: bool = True

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def _path_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path),
        "format": path.suffix.casefold().lstrip("."),
        "exists": path.exists(),
    }
    if not path.exists() or path.is_dir():
        return metadata
    try:
        metadata["size_bytes"] = path.stat().st_size
    except OSError:
        pass
    if path.suffix.casefold() not in (".tif", ".tiff"):
        return metadata
    try:
        import rasterio

        with rasterio.open(path) as dataset:
            metadata.update(
                {
                    "crs": str(dataset.crs) if dataset.crs else None,
                    "spatial_extent": list(dataset.bounds),
                    "spatial_resolution": [abs(dataset.transform.a), abs(dataset.transform.e)],
                    "bands": dataset.count,
                    "nodata": dataset.nodata,
                    "shape": [dataset.height, dataset.width],
                }
            )
    except Exception:
        metadata["metadata_status"] = "unavailable"
    return metadata


class ArtifactRegistry:
    """Request-local registry of resources and produced artifacts."""

    def __init__(self, query: str = ""):
        self.query = query
        self.records: list[ArtifactRecord] = []
        self._sequence = 0
        from agentgs.agent.execution.scope import current_request_id

        self.request_id = current_request_id()
        if self.request_id:
            from uuid import uuid4

            self._reference_namespace = uuid4().hex
        if query:
            self.register_query_resources(query)

    def _next_id(self, semantic_type: str) -> str:
        self._sequence += 1
        if self.request_id:
            return f"artifact_{self.request_id}_{self._reference_namespace}_{self._sequence:04d}_{semantic_type.casefold()}"
        return f"artifact_{self._sequence:04d}_{semantic_type.casefold()}"

    def add(
        self,
        value: Any,
        *,
        semantic_type: str,
        producer: str,
        producer_step: str = "",
        role: str = "intermediate",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        production_metadata = getattr(self, "_production_metadata", None)
        if production_metadata:
            metadata = {**(metadata or {}), **production_metadata}
        normalized_value = str(value) if isinstance(value, Path) else value
        for record in self.records:
            if (
                record.value == normalized_value
                and record.producer == producer
                and (record.producer_step == producer_step)
                and (
                    not self.request_id
                    or (
                        record.metadata.get("call_id") == (metadata or {}).get("call_id")
                        and record.metadata.get("output_field")
                        == (metadata or {}).get("output_field")
                    )
                )
            ):
                return record
        record = ArtifactRecord(
            artifact_id=self._next_id(semantic_type),
            semantic_type=semantic_type,
            value=normalized_value,
            producer=producer,
            producer_step=producer_step,
            role=role,
            metadata=dict(metadata or {}),
        )
        self.records.append(record)
        return record

    def _register_path(self, value: str | Path, *, producer: str, role: str) -> None:
        if getattr(self, "_production_metadata", None):
            path = Path(str(value))
            if not path.is_absolute() or not path.is_file():
                return
        else:
            path = resolve_resource_path(str(value)) or Path(str(value))
        if path.is_dir():
            self.add(
                path,
                semantic_type="Collection",
                producer=producer,
                role=role,
                metadata={"path": str(path), "exists": path.exists(), "format": "directory"},
            )
            try:
                children = sorted((item for item in path.iterdir() if item.is_file()))
            except OSError:
                children = []
            for child in children:
                semantic = _PATH_SUFFIXES.get(child.suffix.casefold(), "File")
                self.add(
                    child,
                    semantic_type=semantic,
                    producer=producer,
                    role=role,
                    metadata=_path_metadata(child),
                )
            return
        semantic = _PATH_SUFFIXES.get(path.suffix.casefold(), "File")
        self.add(
            path,
            semantic_type=semantic,
            producer=producer,
            role=role,
            metadata=_path_metadata(path),
        )

    def register_query_resources(self, query: str) -> None:
        _clean, resource = split_question_resource(query)
        if resource is None:
            return
        if isinstance(resource, str):
            self._register_path(resource, producer="query_resource", role="input")
            return
        if not isinstance(resource, dict):
            return
        assets = resource.get("input_assets") or resource.get("assets") or []
        if isinstance(assets, str):
            assets = [assets]
        for asset in assets:
            self._register_path(asset, producer="query_resource", role="input")
        locator = resource.get("source_locator")
        if isinstance(locator, dict):
            self.add(
                locator,
                semantic_type="Parameter",
                producer="query_resource",
                role="input",
                metadata={"keys": sorted(locator)},
            )

    def ingest_tool_output(
        self, tool_name: str, step_id: str, output: Any, *, execution_info=None
    ) -> list[ArtifactRecord]:
        if self.request_id:
            from agentgs.agent.execution.references import ingest_production

            return ingest_production(self, tool_name, step_id, output, execution_info)
        return self._ingest_legacy_tool_output(tool_name, step_id, output)

    def _ingest_legacy_tool_output(
        self, tool_name: str, step_id: str, output: Any
    ) -> list[ArtifactRecord]:
        try:
            value = json.loads(output) if isinstance(output, str) else output
        except (json.JSONDecodeError, TypeError):
            value = output
        added: list[ArtifactRecord] = []
        rs_contract = None
        if (
            tool_name.startswith("RemoteSensing")
            and isinstance(value, dict)
            and ("_rs_contract" in value)
        ):
            rs_contract = value["_rs_contract"]
            value = {key: item for key, item in value.items() if key != "_rs_contract"}
        reference_only_tools = {
            "RemoteSensingBatchImageMean",
            "RemoteSensingBatchImageSum",
            "RemoteSensingBatchHotspotPercentage",
            "RemoteSensingCountAboveThreshold",
        }
        output_only_tools = {
            "RemoteSensingBatchNDSI",
            "RemoteSensingNDWI",
            "RemoteSensingNBR",
            "RemoteSensingNDVI",
            "RemoteSensingNDTI",
            "RemoteSensingCloudMask",
            "RemoteSensingTiffAverage",
            "RemoteSensingTiffDifference",
            "RemoteSensingBatchHotspotRaster",
            "RemoteSensingSubtract",
            "RemoteSensingFireIncreaseMap",
        }
        if tool_name in output_only_tools:
            output_keys = {
                "output",
                "outputs",
                "output_path",
                "output_file",
                "average_raster",
                "map_png",
                "map_html",
            }

            def register_outputs(candidate: Any) -> None:
                if isinstance(candidate, dict):
                    for key, item in candidate.items():
                        if str(key).casefold() in output_keys:
                            values = item if isinstance(item, (list, tuple)) else [item]
                            for path_value in values:
                                if (
                                    isinstance(path_value, str)
                                    and Path(path_value).suffix.casefold() in _PATH_SUFFIXES
                                ):
                                    before = len(self.records)
                                    self._register_path(
                                        path_value, producer=tool_name, role="intermediate"
                                    )
                                    for record in self.records[before:]:
                                        record.producer_step = step_id
                                        added.append(record)
                        register_outputs(item)
                elif isinstance(candidate, (list, tuple)):
                    for item in candidate:
                        register_outputs(item)

            register_outputs(value)
            reference_only_tools.add(tool_name)
        if tool_name == "RemoteSensingBatchImageMean" and isinstance(value, dict):
            average_raster = value.get("average_raster")
            if (
                isinstance(average_raster, str)
                and Path(average_raster).suffix.casefold() in _PATH_SUFFIXES
            ):
                before = len(self.records)
                self._register_path(average_raster, producer=tool_name, role="intermediate")
                for record in self.records[before:]:
                    record.producer_step = step_id
                    added.append(record)
        for item in _walk(value):
            if isinstance(item, str) and Path(item).suffix.casefold() in _PATH_SUFFIXES:
                if tool_name in reference_only_tools:
                    continue
                before = len(self.records)
                self._register_path(item, producer=tool_name, role="intermediate")
                for record in self.records[before:]:
                    record.producer_step = step_id
                    added.append(record)
            elif isinstance(item, (int, float)) and (not isinstance(item, bool)):
                number = float(item)
                if math.isfinite(number):
                    added.append(
                        self.add(
                            number,
                            semantic_type="Scalar",
                            producer=tool_name,
                            producer_step=step_id,
                            metadata={},
                        )
                    )
        if isinstance(value, (dict, list)):
            added.append(
                self.add(
                    value,
                    semantic_type="StructuredResult",
                    producer=tool_name,
                    producer_step=step_id,
                )
            )
        if isinstance(rs_contract, dict):
            source_assets = {
                item["path"]: item
                for item in rs_contract.get("assets", [])
                if isinstance(item, dict) and item.get("path")
            }
            for record in added:
                if record.semantic_type != "Raster":
                    continue
                sources = []
                for node in _walk(value):
                    if not isinstance(node, dict) or str(
                        node.get("output_path", node.get("average_raster", ""))
                    ) != str(record.value):
                        continue
                    paths = [
                        p.get("source")
                        for p in node.get("input_interpretation", [])
                        if isinstance(p, dict)
                    ]
                    paths += [
                        node[k]
                        for k in ("source_path", "qa_pixel_path", "image_a_path", "image_b_path")
                        if k in node
                    ]
                    sources.extend((source_assets[p] for p in paths if p in source_assets))
                if not sources:
                    sources = list(source_assets.values())
                record.metadata["remote_sensing"] = {
                    "contract_version": rs_contract.get("version"),
                    "source_assets": sources,
                    "acquisition_dates": sorted(
                        {p["acquisition_date"] for p in sources if p.get("acquisition_date")}
                    ),
                    "producer_step": step_id,
                }
        return added

    def paths(
        self, *, semantic_types: Iterable[str] | None = None, producer: str | None = None
    ) -> list[str]:
        allowed = set(semantic_types or ())
        values = []
        for record in self.records:
            if allowed and record.semantic_type not in allowed:
                continue
            if producer and record.producer != producer:
                continue
            if isinstance(record.value, str) and record.semantic_type in {
                "Collection",
                "File",
                "Table",
                "Vector",
                "Raster",
                "Image",
                "Map",
            }:
                values.append(record.value)
        return list(dict.fromkeys(values))

    def available_types(self) -> list[str]:
        return sorted({record.semantic_type for record in self.records})

    def compact_context(self, max_records: int = 24) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.semantic_type] = counts.get(record.semantic_type, 0) + 1
        recent = []
        for record in self.records[-max_records:]:
            value = record.value
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False, default=str)
                value = rendered[:400] + ("…" if len(rendered) > 400 else "")
            recent.append(
                {
                    "artifact_id": record.artifact_id,
                    "semantic_type": record.semantic_type,
                    "producer": record.producer,
                    "producer_step": record.producer_step,
                    "value": value,
                    "metadata": {
                        key: record.metadata.get(key)
                        for key in (
                            "format",
                            "crs",
                            "spatial_extent",
                            "spatial_resolution",
                            "bands",
                            "nodata",
                        )
                        if key in record.metadata
                    },
                }
            )
        return _json_safe({"counts": counts, "available_types": sorted(counts), "recent": recent})

    def as_trace(self) -> dict[str, Any]:
        return {
            "artifact_count": len(self.records),
            "available_types": self.available_types(),
            "artifacts": [record.as_dict() for record in self.records],
        }
