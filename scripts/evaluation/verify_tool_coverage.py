"""Execute every scoped tool with reference/synthetic arguments, without model calls.

Reference parameters are evaluation fixtures. This is tool acceptance, not Agent
planning or a benchmark prediction. No reference answers enter the runtime.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

from agentgs.contracts.models import ToolCall
from agentgs.datasets.registry import DatasetRegistry
from agentgs.evaluation.benchmark import write_json
from agentgs.evaluation.datasets.geotooleval import load_tasks
from agentgs.evaluation.scientific import bind_inputs, create_inputs, observe_result
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.tools.executor import LocalToolExecutor
from agentgs.skills.definitions import load_tool_scope


def cases(root, inputs):
    result = {}
    for task in load_tasks(root / "data/geotooleval/GeoToolEval.xlsx"):
        for step in task.gold_steps:
            result.setdefault(
                step["tool_name"],
                {
                    "tool": step["tool_name"],
                    "arguments": step["arguments"],
                    "fixture": "reference_parameters",
                    "question_id": task.question_id,
                },
            )
    # Replace unresolved historical step placeholders with measured local fixtures.
    portable = json.loads((root / "src/tests/fixtures/scientific/synthetic.json").read_text())
    for case in portable["cases"]:
        if case["expected"]["result"].get("status") != "error":
            result[case["tool"]] = {
                "tool": case["tool"],
                "arguments": bind_inputs(case["arguments"], inputs),
                "fixture": "synthetic",
            }
    points = inputs / "points.geojson"
    points.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": coordinate},
                        "properties": {"value": value, "age": value, "case_id": str(i)},
                    }
                    for i, (coordinate, value) in enumerate(
                        [([0.5, 1.5], 1), ([1.5, 0.5], 3), ([2.5, 1.5], 5)]
                    )
                ],
            }
        )
    )
    zones = inputs / "zones.geojson"
    zones.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [3, 0], [3, 2], [0, 2], [0, 0]]],
                        },
                        "properties": {"zone_id": "A"},
                    }
                ],
            }
        )
    )
    first, second = str(inputs / "first.tif"), str(inputs / "second.tif")
    overrides = {
        "ClimateCompareEpochs": {"values": [1, 2, 4]},
        "RemoteSensingListFiles": {"dir_path": str(inputs)},
        "RemoteSensingCloudMask": {
            "sr_band_path": first,
            "qa_pixel_path": second,
            "output_path": "masked.tif",
        },
        "RemoteSensingExtremeSnowLossPercentage": {"binary_map_path": first, "threshold": 3},
        "RemoteSensingFireIncreaseMap": {
            "change_image_path": first,
            "threshold": 3,
            "output_path": "fire.tif",
        },
    }
    for prefix in ("DeepTime", "CCS"):
        for suffix in (
            "StandardizeSpatialData",
            "ValidateSpatialData",
            "NearestNeighbor",
            "Buffer",
            "BatchVector",
        ):
            overrides[prefix + suffix] = {"input_path": str(points)}
        overrides[prefix + "Overlay"] = {"paths": [str(points), str(zones)]}
        overrides[prefix + "InterpolateIDW"] = {
            "points_path": str(points),
            "value_field": "value",
            "grid_size": 5,
            "bbox": [0, 0, 3, 2],
        }
        overrides[prefix + "RasterAlgebra"] = {
            "raster_path": first,
            "operation": "mask",
            "minimum": 3,
        }
        overrides[prefix + "ZonalStatistics"] = {
            "raster_path": first,
            "zones_path": str(zones),
            "zone_field": "zone_id",
            "operation": "mean",
        }
        overrides[prefix + "RenderMap"] = {"raster_path": first}
    for name, arguments in overrides.items():
        result[name] = {"tool": name, "arguments": arguments, "fixture": "synthetic"}
    return [result[name] for name in load_tool_scope()["tool_ids"]]


async def verify(root, output, source=None):
    class Events:
        def emit(self, *args):
            pass

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "tool_acceptance_reference_and_synthetic_inputs",
        "model_calls": 0,
        "tools": [],
    }
    reference = None
    if source:
        contract = json.loads((root / "docs/algorithm_alignment.json").read_text())
        for relative, identity in contract["baseline"]["files"].items():
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != identity["sha256"]:
                raise ValueError(f"Original source differs: {relative}")
        path = root / "scripts/build/capture_scientific_baseline.py"
        spec = importlib.util.spec_from_file_location("scientific_capture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = module.original_runtime(source, output / "source_artifacts")
        report["source_tree_sha256"] = contract["baseline"]["tree_sha256"]
        report["source_io_adaptations"] = [
            "Outputs redirected to per-tool acceptance directories",
            "Vector output directory prepared as in the original Standardize -> Overlay workflow",
            "Input provenance compared by content SHA256; output paths compared by role",
            "Map data area compared separately from the path-dependent provenance footer",
        ]

    manifest = json.loads((root / "data/processed/resource_manifest.json").read_text())
    identities = {}
    for record in manifest["records"]:
        identities[str(root / record["path"])] = record["sha256"]
        identities[record["source"]] = record["sha256"]

    def canonical_resources(value):
        if isinstance(value, dict):
            return {k: canonical_resources(v) for k, v in value.items()}
        if isinstance(value, list):
            return [canonical_resources(v) for v in value]
        if isinstance(value, str):
            if value in identities:
                return "$RESOURCE/" + identities[value]
            if value.startswith("/") and Path(value).is_file():
                with Path(value).open("rb") as stream:
                    return "$RESOURCE/" + hashlib.file_digest(stream, "sha256").hexdigest()
        return value

    (root / "data").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "data", prefix="acceptance_") as directory:
        inputs = create_inputs(Path(directory))
        for index, case in enumerate(cases(root, inputs)):
            artifacts = LocalArtifactStore(output / "artifacts" / case["tool"])
            tools = LocalToolExecutor(DatasetRegistry(root), artifacts, Events())
            try:
                result = await tools.execute(
                    ToolCall(tool=case["tool"], arguments=case["arguments"])
                )
                row = {
                    "tool": case["tool"],
                    "fixture": case["fixture"],
                    "succeeded": result.succeeded,
                    "error": result.error.model_dump() if result.error else None,
                    "observation": observe_result(result.data, inputs, artifacts.root),
                    "artifacts": [a.model_dump() for a in result.artifacts],
                }
            except Exception as exc:
                row = {
                    "tool": case["tool"],
                    "fixture": case["fixture"],
                    "succeeded": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            if reference and case["tool"] != "GeoBigDataExecuteGeoSPARQL":
                original_outputs = output / "source_artifacts" / case["tool"]
                original_outputs.mkdir(parents=True, exist_ok=True)
                (original_outputs / "vector").mkdir(exist_ok=True)
                reference._RUNTIME_OUTPUTS = original_outputs
                expected = json.loads(reference._tool(case["tool"])(json.dumps(case["arguments"])))
                expected = canonical_resources(observe_result(expected, inputs, original_outputs))
                actual = canonical_resources(row.get("observation"))
                row["source_expected"] = expected
                row["source_matches"] = expected == actual
            elif reference:
                row["source_matches"] = None
                row["source_exception"] = (
                    "Authorized snapshot protocol; covered by 125 independent replay cases"
                )
            report["tools"].append(row)
            write_json(output / "coverage.json", report)
            print(
                json.dumps(
                    {
                        "index": index + 1,
                        "tool": case["tool"],
                        "succeeded": row["succeeded"],
                        "error": row["error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    report["passed"] = sum(row["succeeded"] for row in report["tools"])
    report["failed"] = len(report["tools"]) - report["passed"]
    if reference:
        report["source_compared"] = sum(
            row.get("source_matches") is not None for row in report["tools"]
        )
        report["source_differences"] = [
            row["tool"] for row in report["tools"] if row.get("source_matches") is False
        ]
    write_json(output / "coverage.json", report)
    return report["failed"] == 0


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "outputs/verification/tools")
    parser.add_argument(
        "--source", type=Path, help="Optional read-only source oracle; not a runtime dependency"
    )
    args = parser.parse_args()
    raise SystemExit(
        0
        if asyncio.run(
            verify(root, args.output.resolve(), args.source.resolve() if args.source else None)
        )
        else 1
    )


if __name__ == "__main__":
    main()
