"""Capture deterministic decisions from a hash-checked, read-only source baseline.

Only data-only declarations and pure planning/diagnostic functions are imported.
No harness, tools runtime, model, service or scientific backend is started.
Existing snapshots are never overwritten by this command.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def package(name, path):
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference(source):
    package("alignment_reference", source)
    package("alignment_reference.skills", source / "app/skills")
    mapping = load(
        "alignment_reference.skills.mapping_extensions", source / "app/skills/mapping_extensions.py"
    )
    # Freeze the import-time library profile explicitly, without changing os.environ.
    mapping.mapping_extensions_enabled = lambda: True
    library = load("alignment_reference.skills.library", source / "app/skills/library.py")
    load("alignment_reference.skills.models", source / "app/skills/models.py")
    load("alignment_reference.skills.hypergraph", source / "app/skills/hypergraph.py")
    for name, relative in [
        ("ToolsAgent", "ToolsAgent"),
        ("ToolsAgent.ToolsFuns", "ToolsAgent/ToolsFuns"),
        ("ToolsAgent.ToolsFuns.GeoEval", "ToolsAgent/ToolsFuns/GeoEval"),
    ]:
        package(name, source / relative)
    raster = load(
        "ToolsAgent.ToolsFuns.GeoEval.rs_contracts",
        source / "ToolsAgent/ToolsFuns/GeoEval/rs_contracts.py",
    )
    raster.enabled = lambda: True
    planner = load("alignment_reference.skills.planning", source / "app/skills/planning.py")
    feedback = load("alignment_reference.feedback", source / "app/tools/execution_feedback.py")
    validator = load("alignment_reference.validator", source / "app/llms/evidence_validator_v3.py")
    candidate_context = load(
        "alignment_reference.candidate_context", source / "app/llms/planning/candidate_context.py"
    )
    return library, mapping, planner, feedback, validator, candidate_context


def cases(definitions):
    marker = (
        "Evaluation input resource (authoritative benchmark locator; use it exactly "
        "when relevant, but retrieve and select the registered tools yourself):"
    )
    queries = [
        "Count earthquake events in the frozen USGS catalog.",
        "Estimate earthquake energy from moment magnitude Mw.",
        "Query LITHO1.0 Moho depth at this location.",
        "Sample the SEMUCB model at a USGS earthquake event.",
        "Find lakes inside a county using GeoSPARQL.",
        "Search critical mineral deposits in a country.",
        "Rank mineral commodity countries.",
        "Compare SSP2-4.5 warming with mean GISTEMP temperature.",
        "Query deep-time climate age in kyr.",
        "Compute renewable capacity growth between two years.",
        "Find the nearest mineral site to an energy facility.",
        "Read SOCAT monthly carbon grid values.",
        "Compute a GLODAP surface to deep vertical profile difference.",
        "Check SNAPO cruise quality and deduplicate stations.",
        "Convert BioDeepTime longitude coordinates to DMS.",
        "Interpolate a CCS spatial surface and render a map.",
        "Normalize the coordinate system, clip and resample a raster.",
        "Calculate NDVI from red and near infrared rasters.",
        "Calculate NBR and cloud-mask a Landsat image.",
        "Compare raster means across two time periods.",
        "Compute a Sen slope, Mann Kendall trend and kurtosis.",
        "Filter EM-DAT events and sum deaths by country.",
        "Select the latest natural disaster start date.",
        "Calculate WGS84 distance, bearing and bounding box area.",
        "完全没有相关信息 xyzunknown",
        "",
        "GeoHaversineDistance GeoBearing",
        "深时 坐标 地震 能量",
        "Inspect source data. " + marker + ' {"input_assets":["scene.tif"]}',
        "Inspect source data. " + marker + ' {"input_assets":["points.geojson"]}',
        "Inspect source data. " + marker + ' {"assets":["events.csv"]}',
        "Find lakes. " + marker + " county_lake_query",
        "Inspect data. " + marker + " input.tiff?download=1",
        "Query events. " + marker + ' {"source_locator":{"year":2011,"crs":"EPSG:4326"}}',
        "First resource. " + marker + ' "a.csv" ' + marker + ' "b.geojson"',
        "USGS 地震数据获取",
        "地震事件地图可视化",
    ]
    rows = [
        {"id": f"query-{i:02d}", "query": query, "inferred_tools": [], "mapping_extensions": True}
        for i, query in enumerate(queries, 1)
    ]
    for i, skill in enumerate(definitions, 1):
        rows.append(
            {
                "id": f"hint-{i:02d}",
                "query": "Process the supplied resource.",
                "inferred_tools": skill["tool_candidates"][:1],
                "mapping_extensions": True,
            }
        )
    rows.append(
        {
            "id": "hint-soft-cap",
            "query": "Process the supplied resource.",
            "inferred_tools": [skill["tool_candidates"][0] for skill in definitions[:5]],
            "mapping_extensions": True,
        }
    )
    for i, query in enumerate(queries[:6] + [queries[-2], queries[-1]], 1):
        rows.append(
            {
                "id": f"disabled-{i:02d}",
                "query": query,
                "inferred_tools": [],
                "mapping_extensions": False,
            }
        )
    return rows


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("Snapshot destination already exists; choose a fresh --output")
    if output.is_relative_to(source):
        parser.error("Snapshot destination must be outside the read-only legacy source")
    contract = json.loads((root / "docs/algorithm_alignment.json").read_text("utf-8"))
    for relative, identity in contract["baseline"]["files"].items():
        path = source / relative
        if not path.is_file() or digest(path) != identity["sha256"]:
            parser.error(f"Source differs from the approved baseline: {relative}")
    selection = contract["baseline"]["selection"]
    scope_path = root / selection["manifest"]
    if digest(scope_path) != selection["sha256"]:
        parser.error("Tool scope differs from the approved selection")
    scope = json.loads(scope_path.read_text("utf-8"))
    library, mapping, planner, feedback, validator, candidate_context = reference(source)
    # Change only the explicitly selected declaration corpus. All decisions below
    # are still executed by the hash-checked original algorithm, never new code.
    definitions = [
        skill for skill in library.load_skill_library() if skill["id"] in scope["skill_ids"]
    ]
    # Explicit user-authorized protocol changes can alter declaration text.
    # Execute the original planner over that documented corpus, not new outputs.
    for change in contract.get("authorized_algorithm_changes", []):
        for override in change.get("skill_overrides", []):
            skill = next(s for s in definitions if s["id"] == override["id"])
            skill["description"] = override["description"]
            skill["goal"]["description"] = override["description"]
    library.LEGACY_SKILL_DEFINITIONS[:] = [
        skill for skill in definitions if not skill.get("mapping_extension")
    ]
    library.SKILL_DEFINITIONS[:] = definitions
    captured = []
    for case in cases(definitions):
        enabled = case["mapping_extensions"]
        mapping.mapping_extensions_enabled = lambda enabled=enabled: enabled
        matches = planner.SkillRetriever(definitions).retrieve(
            case["query"], top_k=5, tool_names=case["inferred_tools"]
        )
        plan = planner.SkillPlanner(definitions).plan(
            case["query"], inferred_tools=case["inferred_tools"], top_k=5
        )
        captured.append(
            {
                **case,
                "matches": [
                    {
                        "skill_id": match.skill_id,
                        "score": match.score,
                        "matched_terms": list(match.matched_terms),
                        "score_components": match.score_components,
                    }
                    for match in matches
                ],
                "selected_skill_ids": [step["skill_id"] for step in plan.steps],
                "capability_expansion": plan.capability_expansion,
                "constraints": plan.constraints,
                "plan_sha256": decision_digest(plan.as_dict()),
            }
        )
    graph = planner.apply_extension_graph_semantics(planner.SkillHypergraph(definitions))
    tool_graph = planner.ToolGraph(graph)
    available = list(tool_graph.tool_capabilities)
    if set(available) != set(scope["tool_ids"]):
        parser.error("Original Skill graph does not exactly cover the selected tools")
    candidate_cases = []
    for name, supplied, inferred, schemas, artifacts in [
        ("full-registry", available, [], {}, []),
        ("unavailable", [], [], {}, []),
        ("query-hint", available, ["DeepEarthQueryUSGSCatalog"], {}, []),
        (
            "schema-and-type",
            available,
            [],
            {
                "DeepTimeHaversineDistance": {
                    "Input": "Spatial coordinates",
                    "Output": "Numeric distance",
                },
            },
            ["Spatial", "Numeric"],
        ),
    ]:
        candidate_cases.append(
            {
                "id": name,
                "available_tools": supplied,
                "inferred_tools": inferred,
                "schemas": schemas,
                "artifact_types": artifacts,
                "expected": {
                    cid: [row.as_dict() for row in rows]
                    for cid, rows in tool_graph.candidates(
                        graph.capabilities, supplied, inferred, schemas, artifacts
                    ).items()
                },
            }
        )
    grounding_cases = []
    schemas = {
        "DeepEarthQueryUSGSCatalog": {"Input": "Catalog filter", "Output": "Table events"},
        "DeepTimeHaversineDistance": {"Input": "Spatial coordinates", "Output": "Numeric distance"},
    }
    for name, query, supplied, inferred, proposed in [
        (
            "repeat-instances",
            "Query earthquake events.",
            available,
            [],
            ["DeepEarthQueryUSGSCatalog"] * 2,
        ),
        ("cross-skill-tools", "Query earthquake events.", available, [], ["DeepTimeHaversineDistance"]),
        ("unknown-name", "Query earthquake events.", available, [], ["NotRegistered"]),
        (
            "partial-registry",
            "Query earthquake events.",
            ["DeepEarthQueryUSGSCatalog"],
            [],
            ["DeepEarthGetUSGSEvent"],
        ),
        (
            "inferred-priority",
            "Query earthquake events.",
            available,
            ["DeepEarthQueryUSGSCatalog"],
            [],
        ),
        ("semantic-candidates", "Calculate distance and bearing.", available, [], []),
        ("no-backends", "Query earthquake events.", [], [], []),
        (
            "mixed-repeated",
            "Calculate distance and query earthquake events.",
            available,
            [],
            ["DeepTimeHaversineDistance", "DeepEarthQueryUSGSCatalog", "DeepEarthQueryUSGSCatalog"],
        ),
    ]:
        mapping.mapping_extensions_enabled = lambda: True
        plan = planner.SkillPlanner(definitions).plan(query, inferred_tools=inferred)
        expected = (
            planner.ToolGrounder(definitions)
            .ground(
                plan,
                supplied,
                inferred_tools=inferred,
                proposed_chain=proposed,
                tool_schemas=schemas,
                tool_next={"DeepTimeHaversineDistance": ["DeepEarthQueryUSGSCatalog"]},
                artifact_types=["Spatial", "Numeric"],
            )
            .as_dict()
        )
        # The complete lower graph is already captured once above.
        expected.pop("tool_graph")
        grounding_cases.append(
            {
                "id": name,
                "query": query,
                "available_tools": supplied,
                "inferred_tools": inferred,
                "proposed_chain": proposed,
                "schemas": schemas,
                "tool_next": {"DeepTimeHaversineDistance": ["DeepEarthQueryUSGSCatalog"]},
                "artifact_types": ["Spatial", "Numeric"],
                "expected": expected,
            }
        )
    feedback_cases = []
    for i, value in enumerate(
        [
            "",
            "Input error: missing coordinates",
            "CRS check failed: EPSG:4326 and EPSG:3857",
            {"error": 0.02, "value": 3.0},
            {"success": False},
            {"status_code": 500},
            "Tool function execution error, ConnectionError: connection refused",
            "Tool function execution error, ValueError: unknown event",
            "Traceback (most recent call last): failure",
            "Runtime error: invalid operation",
            "a normal result",
            {"status": "success", "error": 0.1},
        ],
        1,
    ):
        for attempt in [1, 3]:
            result = feedback.assess_tool_execution(
                value, input_format="JSON", attempt=attempt, max_attempts=3
            )
            feedback_cases.append(
                {
                    "id": f"feedback-{i:02d}-{attempt}",
                    "output": value,
                    "attempt": attempt,
                    "max_attempts": 3,
                    "input_format": "JSON",
                    "expected": result.__dict__,
                }
            )
    validation_cases = []
    for i, (question, answer, outputs, completed) in enumerate(
        [
            ("Report the measured value.", "3.0", [{"value": 3.0}], True),
            ("Report the measured value.", "30.0", [{"value": 3.0}], True),
            ("Report the measured value.", "3.0", [{"value": 3.0}], False),
            ("Report the measured value.", "", [{"value": 3.0}], True),
            ("Report the measured value.", "3.0", [], True),
            ("Report the measured value.", "0", [{"value": 0.001}], True),
            ("Report the measured value.", "1.23", [{"value": 1.234}], True),
            ("Report the measured value.", "1.23", [{"value": 1.239}], True),
            ("Report the measured value.", "1.2e3", [{"value": 1234.0}], True),
            ("Report the observation.", "event_id: 42", [{"event_id": 42}], True),
            ("Report the observation.", "42", [{"event_id": 42}], True),
            ("Report the date.", "2011-03-11", [{"date": "2011-03-11"}], True),
            ("Report the value. Options: A. 1.0; B. 2.0", "B. 2.0", [{"value": 2.0}], True),
            ("Report the value. Options: A. 1.0; B. 2.0", "A. 1.0", [{"value": 2.0}], True),
            (
                "Report the value. Options: A. 1.0; B. 2.0",
                "None of the options match. Value: 3.0",
                [{"value": 3.0}],
                True,
            ),
            (
                "Report the value. Options: A. 1.0; B. 2.0",
                "A. 1.0",
                [{"value": 1.0}, {"value": 2.0}],
                True,
            ),
        ],
        1,
    ):
        preserved, event = validator.validate_returned_evidence(
            question, answer, outputs, completed=completed
        )
        validation_cases.append(
            {
                "id": f"validation-{i:02d}",
                "question": question,
                "answer": answer,
                "outputs": outputs,
                "completed": completed,
                "expected_answer": preserved,
                "expected_event": event,
            }
        )
    context_cases = []
    references = [
        ["early", "inputs", "Required input " * 20],
        ["early", "documentation url", "https://example.invalid/long-documentation"],
        ["late", "inputs", "Spatial inputs"],
        ["late", "outputs", {"value": "Numeric result"}],
        ["late", "recommended next tool", "final"],
        ["late", "inputs", "Spatial inputs"],
        {"subject": "final", "predicate": "is sourced from", "object": "Original source"},
        {"unrecognized_reference": ["preserve", "this"]},
    ]
    for budget in [0, 120, 900, 48000]:
        context_cases.append(
            {
                "id": f"context-{budget}",
                "references": references,
                "max_chars": budget,
                "expected": json.loads(
                    candidate_context.build_planning_candidate_context(references, budget)
                ),
            }
        )
    # Also characterize the startup flag, distinct from toggling admission after
    # importing the selected library. Scope selection removes both extension
    # Skills, so either startup flag must now preserve the same 24 declarations.
    library_profiles = []
    for enabled in [True, False]:
        mapping.mapping_extensions_enabled = lambda enabled=enabled: enabled
        library.SKILL_DEFINITIONS[:] = definitions if enabled else library.LEGACY_SKILL_DEFINITIONS
        original = planner.SkillPlanner()
        query = "Count earthquake events in the frozen USGS catalog."
        library_profiles.append(
            {
                "id": "extensions-on" if enabled else "extensions-off",
                "mapping_extensions": enabled,
                "query": query,
                "skill_ids": list(original.skills),
                "graph_sha256": decision_digest(original.hypergraph.as_dict()),
                "tool_graph_sha256": decision_digest(
                    planner.ToolGraph(original.hypergraph).as_dict()
                ),
                "plan_sha256": decision_digest(original.plan(query).as_dict()),
            }
        )
    snapshot = {
        "schema_version": 1,
        "baseline_id": contract["baseline"]["id"],
        "source_tree_sha256": contract["baseline"]["tree_sha256"],
        "selection_sha256": selection["sha256"],
        "oracle": "executed original data-only SkillRetriever and SkillPlanner; no models or backends",
        "library": definitions,
        "graph": graph.as_dict(),
        "tool_graph": tool_graph.as_dict(),
        "cases": captured,
        "candidate_cases": candidate_cases,
        "grounding_cases": grounding_cases,
        "feedback_cases": feedback_cases,
        "validation_cases": validation_cases,
        "context_cases": context_cases,
        "library_profiles": library_profiles,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"snapshot": str(output), "cases": len(captured), "skills": len(definitions)}))


if __name__ == "__main__":
    main()
