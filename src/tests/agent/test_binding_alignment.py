import json
from pathlib import Path

import pytest

from agentgs.agent.execution.bindings import prepare_tool_input

CASES = json.loads((Path(__file__).parents[1] / "fixtures/bindings/oracle.json").read_text("utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[f"case-{i}" for i in range(len(CASES))])
def test_scoped_domain_binder_preserves_original_decision(case):
    result = prepare_tool_input(
        case["tool"],
        case["generated"],
        case["question"],
        current_output=case["current"],
        execution_context=case["context"],
    )
    assert list(result) == case["expected"]
