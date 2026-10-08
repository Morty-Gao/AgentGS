import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentgs.agent.retrieval.tools import KGToolRetriever

CASES = json.loads((Path(__file__).parents[1] / "fixtures/alignment/kg.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[f"kg-{i}" for i in range(len(CASES))])
async def test_source_sdk_and_kg_decisions(case):
    class Embed:
        def get_text_embedding(self, text):
            return [1.0, 2.0] if case["dimension"] == 2 else [1.0, 2.0, 3.0]

    retriever = KGToolRetriever(case["vectors"], case["graph"], Embed())
    assert await retriever.retrieve(SimpleNamespace(query_str=case["query"])) == case["expected"]
