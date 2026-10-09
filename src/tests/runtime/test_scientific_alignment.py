import importlib.util
import json
from pathlib import Path

import pytest

from agentgs.runtime.tools.context import tool_context
from agentgs.runtime.tools.dispatch import execute_domain

ROOT = Path(__file__).parents[3]
CASES = json.loads((ROOT / "src/tests/fixtures/scientific/oracle.json").read_text("utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[f"science-{i}" for i in range(len(CASES))])
def test_actual_scientific_results_match_original_source(case, tmp_path):
    if (
        importlib.util.find_spec("pandas") is None
        or not (ROOT / "data/processed/resource_manifest.json").exists()
    ):
        pytest.skip("Original scientific dependency/data profile is not installed")
    with tool_context(ROOT, tmp_path):
        actual = json.loads(execute_domain(case["tool"], case["arguments"]))
    assert actual == case["expected"]


SYNTHETIC = json.loads((ROOT / "src/tests/fixtures/scientific/synthetic.json").read_text("utf-8"))


@pytest.mark.parametrize("case", SYNTHETIC["cases"], ids=lambda c: c["tool"])
def test_portable_raster_and_spatial_source_decisions(case, tmp_path):
    if importlib.util.find_spec("rasterio") is None:
        pytest.skip("Scientific raster extra is required")
    from agentgs.evaluation.analysis.scientific import bind_inputs, create_inputs, observe_result

    inputs = create_inputs(tmp_path)
    outputs = tmp_path / "outputs"
    with tool_context(tmp_path, outputs):
        actual = json.loads(execute_domain(case["tool"], bind_inputs(case["arguments"], inputs)))
    expected = {
        **case["expected"],
        "images": {
            name: {
                key: value
                for key, value in image.items()
                if key != "full_image_sha256_diagnostic_only"
            }
            for name, image in case["expected"]["images"].items()
        },
    }
    assert observe_result(actual, inputs, outputs) == expected
