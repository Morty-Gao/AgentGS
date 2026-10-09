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


async def test_routing_hints_control_domain_seeds_per_request():
    import asyncio

    class Embed:
        def get_text_embedding(self, text):
            return [1.0, 0.0]

    descriptions = {
        "RemoteSensingMean": "Compute a scalar mean",
        "NaturalHazardLoadEventTable": "Load natural hazard records",
    }
    vectors = {
        str((name, "has the functionality that", description)): vector
        for (name, description), vector in zip(descriptions.items(), [[1.0, 0.0], [0.0, 1.0]])
    }
    graph = {
        name: [["has the functionality that", description]]
        for name, description in descriptions.items()
    }
    retriever = KGToolRetriever(vectors, graph, Embed(), similarity_top_k=1, max_tools=1)
    enabled, disabled = await asyncio.gather(
        *(
            retriever.retrieve(
                SimpleNamespace(query_str="Read the natural hazard event table", routing_hints=flag)
            )
            for flag in (True, False)
        )
    )
    assert enabled[0][0] == "NaturalHazardLoadEventTable"
    assert disabled[0][0] == "RemoteSensingMean"
