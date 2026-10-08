import json

from agentgs.agent.execution.bindings import prepare_tool_input, tool_input_is_ready
from agentgs.agent.retrieval.scored_skills import RESOURCE_MARKER
from agentgs.runtime.tools.domains.geosparql import SnapshotQuery


def test_symbolic_topic_is_provenance_not_a_snapshot_identifier():
    question = "Report forest areas.\n" + RESOURCE_MARKER + " forest_area_statistics"
    initial, _ = prepare_tool_input("GeoBigDataExecuteGeoSPARQL", "{}", question)
    assert not tool_input_is_ready("GeoBigDataExecuteGeoSPARQL", initial)
    query = "SELECT ?forest WHERE { ?forest rdf:type y2geoo:OSM_forest }"
    prepared, _ = prepare_tool_input(
        "GeoBigDataExecuteGeoSPARQL",
        json.dumps({"query": query, "endpoint": "unused", "input_resource": "topic"}),
        question,
    )
    params = SnapshotQuery.model_validate_json(prepared)
    assert params.query == query and params.source_id is None
    assert params.dataset == "forest_area_statistics"


def test_numeric_public_source_id_is_retained_without_private_fields():
    question = (
        "Retrieve the source.\n"
        + RESOURCE_MARKER
        + ' {"source_locator":{"source_id":591},"gold_answer":"private"}'
    )
    prepared, _ = prepare_tool_input("GeoBigDataExecuteGeoSPARQL", "{}", question)
    params = SnapshotQuery.model_validate_json(prepared)
    assert params.source_id == 591 and "private" not in prepared
