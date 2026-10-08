"""Offline GeoQuestions replay by source identifier or historical query structure."""

from copy import deepcopy

from pydantic import AliasChoices, Field, model_validator

from agentgs.contracts.models import Model
from agentgs.datasets.geoquestions import _match_materialized_query, _normalize_query, load_snapshot


class SnapshotQuery(Model):
    source_id: int | None = Field(default=None, ge=1, description="Original GeoQuestions source ID")
    query: str = Field(
        default="",
        validation_alias=AliasChoices("query", "sparql"),
        description="SPARQL query matched against frozen structures; used if source_id does not match",
    )
    dataset: str | None = Field(default=None, description="Provenance label; does not select data")

    @model_validator(mode="after")
    def has_locator(self):
        if self.source_id is None and not self.query.strip():
            raise ValueError("Supply source_id or a non-empty SPARQL query")
        return self


def replay_query(params: SnapshotQuery, datasets):
    rows, snapshot_hash = load_snapshot(datasets)
    matched = next((row for row in rows if row["source_id"] == params.source_id), None)
    match_kind = "source_id"
    if matched is None:
        matched = _match_materialized_query(params.query, rows)
        match_kind = (
            "exact"
            if matched
            and _normalize_query(matched["gold_sparql"]) == _normalize_query(params.query)
            else "semantic_equivalent"
        )
    if matched is None:
        raise ValueError("GeoSPARQL source_id/query was not found uniquely in the pinned snapshot")
    return {
        "mode": "pinned_yago2geo_snapshot",
        "source_id": matched["source_id"],
        "query": matched["gold_sparql"],
        "result": deepcopy(matched["result"]),
        "query_match": match_kind,
        "provenance": {
            "execution": "materialized snapshot lookup",
            "snapshot_sha256": snapshot_hash,
            "snapshot_records": len(rows),
            "dataset_label": params.dataset,
            "answer_lookup": True,
        },
    }
