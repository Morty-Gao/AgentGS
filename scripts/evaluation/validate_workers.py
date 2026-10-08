"""Exercise real workers, bindings, artifacts and recovery with a local model fixture."""

import argparse
import asyncio
import json
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agentgs.contracts.models import TaskInput
from agentgs.evaluation.benchmark import write_json
from agentgs.evaluation.scientific import create_inputs
from agentgs.runtime.service import TaskService


async def validate(root, folder):
    project = folder / "project"
    if project.exists():
        raise FileExistsError("Preserve previous acceptance runs; choose a fresh output directory")
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname="worker-validation"\nversion="0.0.0"\n'
    )
    (project / "data").symlink_to(root / "data", target_is_directory=True)
    (project / "tools").symlink_to(root / "tools", target_is_directory=True)
    shutil.copy2(root / "llm.toml", project / "llm.toml")
    shutil.copy2(root / "agent.toml", project / "agent.toml")
    declared = create_inputs(root / "data/worker_fixtures" / folder.name)
    state = {"mode": "recovery", "division": 0, "means": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["messages"][-1]["content"]
            if "You are a professional planner" in prompt:
                chain = (
                    [
                        "RemoteSensingListFiles",
                        "RemoteSensingBatchImageMean",
                        "RemoteSensingDivision",
                    ]
                    if state["mode"] == "recovery"
                    else [
                        "DeepTimeStandardizeSpatialData",
                        "DeepTimeValidateSpatialData",
                        "DeepTimeInterpolateIDW",
                        "DeepTimeRasterAlgebra",
                        "DeepTimeZonalStatistics",
                        "DeepTimeRenderMap",
                    ]
                )
                if state["mode"] == "cross_domain":
                    chain = [
                        "DeepTimeInterpolateIDW",
                        "RemoteSensingBatchImageMean",
                        "RemoteSensingMean",
                    ]
                text = "Plan Chain:\n" + repr(chain)
                stage = "planning"
            elif "tool_name:" in prompt:
                name = re.search(r"tool_name:\s*([^\n]+)", prompt).group(1).strip()
                stage = name
                params = {"input_path": str(declared / "points.geojson")}
                if name == "RemoteSensingListFiles":
                    params = {"dir_path": str(declared)}
                elif name == "RemoteSensingBatchImageMean":
                    state["means"] += 1
                    params = {
                        "file_list": [str(declared / "first.tif")],
                        "rs_statistic_mode": "pixelwise_composite"
                        if state["means"] <= 2
                        else "per_scene_statistics",
                    }
                    if state["mode"] == "cross_domain":
                        products = re.findall(r'"([^"\n]+\.tif)"', prompt)
                        params = {
                            "file_list": [products[-1]],
                            "rs_statistic_mode": "per_scene_statistics",
                        }
                elif name == "RemoteSensingMean":
                    params = {"x": [3]}
                elif name == "RemoteSensingDivision":
                    state["division"] += 1
                    params = {"a": 6, "b": 0 if state["division"] <= 3 else 2}
                elif name == "DeepTimeInterpolateIDW":
                    params = {
                        "points_path": str(declared / "points.geojson"),
                        "value_field": "value",
                        "grid_size": 5,
                        "bbox": [0, 0, 3, 2],
                    }
                elif name in {
                    "DeepTimeRasterAlgebra",
                    "DeepTimeZonalStatistics",
                    "DeepTimeRenderMap",
                }:
                    products = re.findall(r'"([^"\n]+\.tif)"', prompt)
                    params = {"raster_path": products[-1], "operation": "mask", "minimum": 0}
                    if name == "DeepTimeZonalStatistics":
                        params.update(
                            zones_path=str(declared / "zones.geojson"),
                            zone_field="zone_id",
                            operation="mean",
                        )
                    if name == "DeepTimeRenderMap":
                        params.pop("operation")
                text = "Input:" + json.dumps(params)
            else:
                text = (
                    "IsCompleted:Completed\nFinalAnswer:0.9142857142857144"
                    if state["mode"] == "recovery"
                    else "IsCompleted:Completed\nFinalAnswer:The measured spatial workflow and map were produced."
                )
                if state["mode"] == "cross_domain":
                    text = "IsCompleted:Completed\nFinalAnswer:3"
                stage = "answer"
            state["requests"].append(stage)
            response = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": text}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (project / ".env").write_text(
        f"LLM_BASE_URL=http://127.0.0.1:{server.server_port}/v1\nLLM_API_KEY=local-fixture\nLLM_MODEL=local-fixture\n"
    )
    service = TaskService(project)
    report = {
        "protocol": "real_taskservice_local_model_fixture",
        "paid_model_calls": 0,
        "results": [],
    }
    try:
        for mode in ("recovery", "spatial", "cross_domain"):
            state["mode"] = mode
            request = TaskInput(
                question="List remote sensing rasters, calculate their per-image means, then report the ratio of the last two means."
                if mode == "recovery"
                else "Standardize and validate the measured points, interpolate value with IDW, mask the raster, calculate zone mean and render the map.",
                input_resource=str(declared)
                if mode == "recovery"
                else {
                    "points_path": str(declared / "points.geojson"),
                    "zones_path": str(declared / "zones.geojson"),
                },
                planning_mode="skill_guided",
                require_llm=False if mode == "recovery" else True,
                max_model_calls=12,
            )
            if mode == "cross_domain":
                request.question = "Use deep-time IDW on the measured points, then remote sensing per-image means and scalar mean of the resulting GeoTIFF."
            submitted = service.submit(request, timeout_seconds=120, rss_limit_mb=2048)
            result = await service.wait(submitted.spec.id)
            report["results"].append(
                {
                    "mode": mode,
                    "task": result.model_dump(mode="json"),
                    "provider_requests": list(state["requests"]),
                }
            )
            state["requests"].clear()
            checks = {"task_succeeded": result.status.value == "succeeded"}
            if result.result:
                trace = result.result.trace
                if mode == "recovery":
                    checks["reused_successful_prefix"] = any(
                        row["type"] == "plan_resume"
                        and row["skipped_successful_prefix"] == ["RemoteSensingListFiles"]
                        for row in trace
                    )
                    checks["listing_executed_once"] = (
                        sum(
                            row.tool == "RemoteSensingListFiles"
                            for row in result.result.tool_results
                        )
                        == 1
                    )
                    checks["failed_attempts_preserved"] = (
                        sum(not row.succeeded for row in result.result.tool_results) >= 3
                    )
                elif mode == "spatial":
                    checks["six_scientific_steps"] = len(result.result.tool_results) == 6 and all(
                        row.succeeded for row in result.result.tool_results
                    )
                    checks["rendered_files_registered"] = any(
                        a.media_type == "image/png"
                        for row in result.result.tool_results
                        for a in row.artifacts
                    )
                else:
                    checks["cross_domain_chain"] = [
                        row.tool for row in result.result.tool_results
                    ] == [
                        "DeepTimeInterpolateIDW",
                        "RemoteSensingBatchImageMean",
                        "RemoteSensingMean",
                    ]
                    checks["measured_result"] = (
                        abs(result.result.tool_results[-1].data["value"] - 3) < 1e-12
                    )
            report["results"][-1]["checks"] = checks
            write_json(folder / "report.json", report)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "status": result.status.value,
                        "tool_calls": len(result.result.tool_results) if result.result else 0,
                    }
                ),
                flush=True,
            )
    finally:
        await service.close()
        server.shutdown()
        server.server_close()
    return report


if __name__ == "__main__":
    project_root = next(
        p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=project_root / "outputs/verification/workers"
    )
    args = parser.parse_args()
    report = asyncio.run(validate(project_root, args.output.resolve()))
    raise SystemExit(0 if all(all(row["checks"].values()) for row in report["results"]) else 1)
