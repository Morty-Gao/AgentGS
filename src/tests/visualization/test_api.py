import time

from fastapi.testclient import TestClient

from agentgs.visualization.backend.app import create_app


def test_submit_events_artifact_and_errors(project):
    with TestClient(create_app(project)) as client:
        assert client.get("/api/tasks/missing").status_code == 404
        response = client.post(
            "/api/tasks",
            json={"calls": [{"tool": "DeepEarthQueryUSGSCatalog", "arguments": {"min_magnitude": 8}}]},
        )
        assert response.status_code == 202
        task_id = response.json()["spec"]["id"]
        for _ in range(100):
            record = client.get("/api/tasks/" + task_id).json()
            if record["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.03)
        assert record["status"] == "succeeded"
        events = client.get("/api/tasks/" + task_id + "/events").text
        assert "task.succeeded" in events
        ref = record["result"]["tool_results"][0]["artifacts"][0]
        assert client.get(f"/api/tasks/{task_id}/artifacts/{ref['id']}").status_code == 200
        assert client.get(f"/api/tasks/{task_id}/artifacts/not-an-id").status_code == 404
