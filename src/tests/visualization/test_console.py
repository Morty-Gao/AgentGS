"""Console read contracts and failure boundaries, independent of model services."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import (
    AgentResult,
    ArtifactRef,
    TaskInput,
    TaskRecord,
    TaskSpec,
    TaskStatus,
    ToolCall,
    ToolError,
    ToolResult,
)
from agentgs.messages import Conversation, Message
from agentgs.runtime.storage.store import QueueFull, TaskStore
from agentgs.visualization.backend.app import create_app


def save(project, *, task_id="sample", status=TaskStatus.SUCCEEDED, result=None, **inputs):
    record = TaskRecord(
        spec=TaskSpec(id=task_id, input=TaskInput(**inputs)), status=status, result=result
    )
    TaskStore(project / "outputs/runtime").save_sync(record)
    return record


def evidence():
    conversation = Conversation()
    conversation.append(Message(role="user", content="A geoscience question"))
    conversation.record_event("model.started", {"request_id": "request-1", "index": 1})
    call = conversation.begin_tool("catalog", '{"value":1}', call_id="call-1")
    conversation.finish_tool(call, "result", status="success", runtime_arguments={"value": 2})
    return AgentResult(
        answer="An answer",
        messages=conversation.messages,
        usage=[{"call": 1, "stage": "planning", "status": "completed", "usage": None}],
    )


def test_paginated_summaries_filter_search_and_counts(project):
    save(project, task_id="first", question="Arctic temperature", result=evidence())
    save(project, task_id="second", status=TaskStatus.FAILED, calls=[ToolCall(tool="Catalog")])
    save(project, task_id="third", question="Literal % marker", status=TaskStatus.CANCELLED)
    with TestClient(create_app(project)) as client:
        page = client.get("/api/tasks?limit=1").json()
        assert page["total"] == 3
        assert page["items"][0]["id"] == "third"
        assert "result" not in page["items"][0]
        assert page["counts"] == {"succeeded": 1, "failed": 1, "cancelled": 1}
        assert client.get("/api/tasks?limit=1&offset=1").json()["items"][0]["id"] == "second"
        for search, task_id in [("arctic", "first"), ("catalog", "second"), ("%", "third")]:
            result = client.get("/api/tasks", params={"search": search}).json()
            assert result["total"] == 1
            assert result["items"][0]["id"] == task_id
        assert client.get("/api/tasks?status=failed").json()["total"] == 1
        assert client.get("/api/tasks?status=failed&search=Arctic").json()["items"] == []
        assert client.get("/api/tasks?offset=100").json()["items"] == []


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1", "status=bogus"])
def test_list_rejects_invalid_query(project, query):
    with TestClient(create_app(project)) as client:
        assert client.get("/api/tasks?" + query).status_code == 422


def test_full_conversation_export_and_partial_checkpoint(project):
    record = save(project, status=TaskStatus.RUNNING, question="Question")
    work = project / "outputs/tasks" / record.spec.id
    work.mkdir(parents=True)
    result = evidence()
    (work / "result.json").write_text(result.model_dump_json())
    with TestClient(create_app(project)) as client:
        live = client.get("/api/tasks/sample").json()
        assert live["status"] == "running"
        assert live["result"]["answer"] == "An answer"
        entries = client.get("/api/tasks/sample/messages").json()
        assert any(row["role"] == "event" for row in entries)
        assert client.get("/api/tasks/sample/conversation").json() == {
            "schema_version": 2,
            "messages": entries,
        }
        export = client.get("/api/tasks/sample/export")
        assert export.json() == live
        assert "attachment" in export.headers["content-disposition"]
        assert TaskStore(project / "outputs/runtime").get_sync("sample").result is None
        (work / "result.json").write_text("unfinished")
        assert client.get("/api/tasks/sample").json()["result"] is None


def test_terminal_record_is_authoritative_and_preserves_incomplete_result(project):
    result = evidence().model_copy(update={"completion_status": "Not Completed"})
    record = save(project, status=TaskStatus.FAILED, result=result)
    record.error = ToolError(code="task_incomplete", message="incomplete")
    TaskStore(project / "outputs/runtime").save_sync(record)
    work = project / "outputs/tasks/sample"
    work.mkdir(parents=True)
    (work / "result.json").write_text(evidence().model_dump_json())
    with TestClient(create_app(project)) as client:
        response = client.get("/api/tasks/sample").json()
        assert response["result"]["completion_status"] == "Not Completed"
        assert response["error"]["code"] == "task_incomplete"
        assert response["result"]["messages"]


def test_input_conversation_visible_before_first_checkpoint(project):
    save(project, status=TaskStatus.QUEUED, messages=[Message(role="user", content="Question")])
    with TestClient(create_app(project)) as client:
        assert (
            client.get("/api/tasks/sample/conversation").json()["messages"][0]["content"]
            == "Question"
        )


def test_submission_validation_and_queue_backpressure(project):
    app = create_app(project)
    with TestClient(app) as client:
        assert client.post("/api/tasks", json={}).status_code == 422
        assert client.post("/api/tasks", json={"question": "   "}).status_code == 422
        assert (
            client.post("/api/tasks", json={"question": "test", "planning_mode": "old"}).status_code
            == 422
        )
        assert (
            client.post("/api/tasks", json={"question": "test", "obsolete": True}).status_code
            == 422
        )

        async def full(_):
            raise QueueFull("queue full")

        app.state.tasks.submit = full
        response = client.post("/api/tasks", json={"question": "Question"})
        assert response.status_code == 429
        assert response.json()["detail"] == "queue full"


def test_sse_resume_terminal_and_empty_cursor(project):
    save(project)
    store = TaskStore(project / "outputs/runtime")
    for kind in ["task.queued", "tool.completed", "task.succeeded"]:
        store.emit_sync("sample", kind, {})
    with TestClient(create_app(project)) as client:
        response = client.get("/api/tasks/sample/events?after=1", headers={"Last-Event-ID": "2"})
        assert "id: 3\n" in response.text
        assert "id: 2\n" not in response.text
        assert "event: end" in response.text
        assert response.headers["x-accel-buffering"] == "no"
        assert "data:" in client.get("/api/tasks/sample/events?after=99").text
        assert client.get("/api/tasks/sample/events?after=-1").status_code == 422
        assert client.get("/api/tasks/missing/events").status_code == 404
        assert (
            "id: 1" in client.get("/api/tasks/sample/events", headers={"Last-Event-ID": "bad"}).text
        )


@pytest.mark.parametrize("path_kind", ["valid", "missing", "escape", "symlink"])
def test_registered_artifact_access_is_confined(project, path_kind):
    root = project / "outputs/tasks/sample/artifacts"
    root.mkdir(parents=True)
    (project / "private.txt").write_text("private")
    (root / "safe.txt").write_text("safe")
    (root / "link.txt").symlink_to(project / "private.txt")
    path = {
        "valid": "safe.txt",
        "missing": "missing.txt",
        "escape": "../../../../private.txt",
        "symlink": "link.txt",
    }[path_kind]
    ref = ArtifactRef(
        id="artifact", path=path, media_type="text/plain", sha256="abc", bytes=4, source="tool"
    )
    result = AgentResult(
        answer="",
        tool_results=[ToolResult(call_id="call", tool="tool", succeeded=True, artifacts=[ref])],
    )
    save(project, result=result)
    with TestClient(create_app(project)) as client:
        response = client.get("/api/tasks/sample/artifacts/artifact")
        assert response.status_code == (200 if path_kind == "valid" else 404)
        if path_kind == "valid":
            assert response.text == "safe"
            assert "attachment" in response.headers["content-disposition"]
            assert response.headers["x-content-type-options"] == "nosniff"
        assert client.get("/api/tasks/sample/artifacts/unregistered").status_code == 404


def test_catalog_config_and_openapi_are_current(project):
    (project / "config.toml").write_text('[agent]\nplanning_mode="data_only"\ncommand_attempts=7\n')
    with TestClient(create_app(project, RuntimeSettings(default_timeout_seconds=42))) as client:
        config = client.get("/api/config").json()
        assert config["algorithm"]["planning_mode"] == "data_only"
        assert config["algorithm"]["command_attempts"] == 7
        assert config["runtime"]["default_timeout_seconds"] == 42
        assert client.get("/api/tools").json()[0]["input_schema"]
        assert client.get("/api/skills").json()[0]["tools"]
        spec = client.get("/openapi.json").json()
        assert spec["paths"]["/api/tasks"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("TaskPage")
        input_ref = spec["paths"]["/api/tasks"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]["$ref"].rsplit("/", 1)[-1]
        assert "model" in json.dumps(spec["components"]["schemas"][input_ref])


@pytest.mark.parametrize("mode", ["tool_only", "skill_guided", "data_only"])
def test_submission_preserves_current_task_options(project, mode):
    payload = {
        "messages": [{"role": "user", "content": "Question from conversation"}],
        "planning_mode": mode,
        "model": "configured-model",
        "effort": "high",
        "max_model_calls": 12,
        "input_resource": {"source": "data/example.csv"},
        "file_path_list": ["data/example.csv"],
        "output_contract": {"format": "json"},
        "allow_rule_planning": False,
        "plan_attempts": 2,
        "tool_attempts": 4,
        "command_attempts": 7,
        "skill_ids": ["catalog"],
        "input_domain": "earth",
    }
    app = create_app(project)
    with TestClient(app) as client:
        captured = []

        async def capture(task):
            captured.append(task)
            return TaskRecord(spec=TaskSpec(input=task))

        app.state.tasks.submit = capture
        response = client.post("/api/tasks", json=payload)
        assert response.status_code == 202
        assert captured[0].question == "Question from conversation"
        for name, value in payload.items():
            if name != "messages":
                assert getattr(captured[0], name) == value


@pytest.mark.parametrize(
    "media_type,content,expected",
    [
        ("application/json", b'{"value": "<script>alert(1)</script>"}', 200),
        ("image/png", b"\x89PNG\r\n\x1a\n", 200),
        ("text/html", b"<script>alert(1)</script>", 415),
        ("image/svg+xml", b"<svg></svg>", 415),
        ("text/plain", b"x" * (256 * 1024 + 1), 413),
    ],
)
def test_artifact_preview_is_bounded_and_inert(project, media_type, content, expected):
    root = project / "outputs/tasks/sample/artifacts"
    root.mkdir(parents=True)
    (root / "file").write_bytes(content)
    ref = ArtifactRef(
        id="file",
        path="file",
        media_type=media_type,
        sha256="abc",
        bytes=len(content),
        source="tool",
    )
    save(
        project,
        result=AgentResult(
            answer="",
            tool_results=[ToolResult(call_id="call", tool="tool", succeeded=True, artifacts=[ref])],
        ),
    )
    with TestClient(create_app(project)) as client:
        response = client.get("/api/tasks/sample/artifacts/file/preview")
        assert response.status_code == expected
        if expected == 200:
            assert response.content == content
            assert response.headers["content-security-policy"] == "default-src 'none'"
            assert response.headers["content-type"].startswith(
                "image/png" if media_type == "image/png" else "text/plain"
            )


def test_event_cursor_cannot_overflow_sqlite(project):
    save(project)
    with TestClient(create_app(project)) as client:
        assert client.get("/api/tasks/sample/events?after=9223372036854775808").status_code == 422
        assert (
            client.get("/api/tasks/sample/events", headers={"Last-Event-ID": "9" * 30}).status_code
            == 400
        )


def test_terminal_stream_waits_for_supervisor_final_event(project):
    save(project)
    app = create_app(project)
    with TestClient(app) as client:
        async def schedule_finish():
            async def finish():
                await asyncio.sleep(0.05)
                await app.state.tasks.store.emit("sample", "task.succeeded", {})

            app.state.tasks.jobs["sample"] = asyncio.create_task(finish())

        client.portal.call(schedule_finish)
        response = client.get("/api/tasks/sample/events")
        assert response.text.index("task.succeeded") < response.text.index("event: end")
