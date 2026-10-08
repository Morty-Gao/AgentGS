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
