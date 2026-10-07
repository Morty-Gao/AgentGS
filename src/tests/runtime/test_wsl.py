from agentgs.runtime.backends.wsl import run


def test_wsl_bridge_uses_argv_not_shell(monkeypatch):
    captured = []
    monkeypatch.setattr("subprocess.call", lambda argv: captured.append(argv) or 0)
    arguments = ["run", "--tool", "GeoMagnitudeToEnergy", "--arguments", '{"magnitude":7}']
    assert run("/home/user/project", arguments) == 0
    assert captured[0][-len(arguments) :] == arguments
    assert "--exec" in captured[0] and "bash" not in captured[0]
