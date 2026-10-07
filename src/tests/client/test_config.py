import os

import pytest

from agentgs.client.config import load_llm
from agentgs.config.env import interpolate
from agentgs.config.paths import project_root

CONFIG = """schema_version=1
[providers.p]
base_url="${TEST_LLM_URL}"
api_key="${TEST_LLM_KEY}"
[models.m]
provider="p"
model="test"
context_window=100
max_output_tokens=10
supported_efforts=["low"]
[roles]
planner="m"
"""


def test_config_does_not_mutate_environment(project, monkeypatch):
    (project / "llm.toml").write_text(CONFIG)
    (project / ".env").write_text("TEST_LLM_URL=https://fixture/v1\nTEST_LLM_KEY=local-secret\n")
    monkeypatch.setenv("TEST_LLM_KEY", "process-secret")
    config = load_llm(project)
    assert config.providers["p"].api_key.get_secret_value() == "process-secret"
    assert os.environ.get("TEST_LLM_URL") is None
    assert "process-secret" not in repr(config)


def test_missing_env_and_root_discovery(project):
    with pytest.raises(ValueError, match="Missing environment"):
        interpolate("${MISSING_123}", {})
    child = project / "nested"
    child.mkdir()
    assert project_root(child) == project
