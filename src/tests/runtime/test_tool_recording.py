import pytest

from agentgs.contracts.models import ToolCall
from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.tools.executor import LocalToolExecutor


async def test_auxiliary_storage_failures_do_not_erase_success_or_repeat_calculation(project):
    class BrokenArtifacts(LocalArtifactStore):
        def write_json(self, *args):
            raise OSError("disk full")

        def register(self, *args):
            raise OSError("cannot register file")

    class Events:
        def emit(self, *args):
            raise OSError("event sink unavailable")

    artifacts = BrokenArtifacts(project / "artifacts")
    tools = LocalToolExecutor(DatasetRegistry(project), artifacts, Events())
    calls = []

    def compute(name, args, **kwargs):
        calls.append(name)
        (artifacts.root / "product.txt").write_text("scientific output")
        return '{"value":2}'

    tools._domain = compute
    result = await tools.execute(ToolCall(tool="RemoteSensingMean", arguments={"x": [1, 3]}))
    assert result.succeeded and result.data == {"value": 2}
    assert result.raw_output == '{"value":2}' and result.artifacts == []
    assert calls == ["RemoteSensingMean"]


async def test_empty_catalog_arguments_use_the_legacy_count_contract(project):
    class Events:
        def emit(self, *args):
            pass

    tools = LocalToolExecutor(
        DatasetRegistry(project), LocalArtifactStore(project / "artifacts"), Events()
    )
    calls = []

    def domain(name, arguments, **kwargs):
        calls.append((name, arguments))
        return '{"operation":"count","count":3,"event_ids":["a","b","c"]}'

    tools._domain = domain
    result = await tools.execute(ToolCall(tool="DeepEarthQueryUSGSCatalog", arguments={}))
    assert calls == [("DeepEarthQueryUSGSCatalog", {})]
    assert result.succeeded and result.data["event_ids"] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "relative",
    [
        "geotooleval/GeoToolEval.xlsx",
        "geotooleval/GeoToolEval.jsonl",
        "geotooleval/release_v1_1/rescore_and_replay.json",
        "answer_formats/format.jsonl",
        "geoquestions1089/GeoQuestions1089_answers.json",
    ],
)
def test_generic_tools_cannot_resolve_private_or_replay_data(project, relative):
    from agentgs.runtime.tools.context import ToolPaths

    path = project / "data" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("private benchmark data")
    context = ToolPaths.create(project, project / "artifacts")
    with pytest.raises(ValueError):
        context.resolve(path)


def test_scientific_natural_hazard_workbook_remains_accessible(project):
    from agentgs.runtime.tools.context import ToolPaths

    path = project / "data/geotooleval/10.natural-hazards-and-extreme-earthquakes.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("scientific observations")
    assert ToolPaths.create(project, project / "artifacts").resolve(path) == path
