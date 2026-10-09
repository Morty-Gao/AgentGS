"""Allowlisted data mounts stay read-only and exclude answer files."""

import pytest

from agentgs.config.settings import RuntimeSettings
from agentgs.runtime.sandbox.base import CodeExecutor, SandboxUnavailable, build_shell_command
from agentgs.runtime.sandbox.mounts import DATA_DIRECTORIES, DATA_FILES, data_mounts, mount_manifest


def test_mount_manifest_keeps_stored_names_and_omits_answer_keys(tmp_path):
    data = tmp_path / "data"
    (data / "climate").mkdir(parents=True)
    (data / "climate" / "note.txt").write_text("42\n")
    (data / "earthbench").mkdir()
    (data / "mineral.csv").write_text("a,b\n")
    for name in (
        "geotooleval.jsonl",
        "metadata",
        "geoquestions",
        "raw",
        "geotoolkg",
    ):
        path = data / name
        if name.endswith(".jsonl"):
            path.write_text('{"expected_answer":"secret"}\n')
        else:
            path.mkdir()
            (path / "secret.txt").write_text("secret")
    guests = mount_manifest(data)
    assert guests == ["/data/climate", "/data/earthbench", "/data/mineral.csv"]
    assert "geotooleval.jsonl" not in DATA_FILES
    assert {"metadata", "geoquestions", "raw", "geotoolkg"}.isdisjoint(DATA_DIRECTORIES)
    assert all("secret" not in guest for guest in guests)
    assert data_mounts(data / "missing") == []


def test_shell_argv_binds_data_without_copying_or_exposing_answers(tmp_path):
    climate = tmp_path / "climate"
    climate.mkdir()
    work = tmp_path / "work"
    mounts = [(climate, "/data/climate")]
    bwrap = build_shell_command("bwrap", "bwrap", "ls /data", work, mounts)
    guest = bwrap.index("/data/climate")
    assert bwrap[guest - 2 : guest] == ["--ro-bind", str(climate)]
    assert "--unshare-all" in bwrap
    bind = bwrap.index("--bind")
    assert bwrap[bind : bind + 3] == ["--bind", str(work), "/work"]
    assert "geotooleval" not in " ".join(bwrap)
    with pytest.raises(SandboxUnavailable):
        build_shell_command("proot", "proot", "ls /data", work, mounts)


async def test_bwrap_shows_mounted_data_and_hides_answers(sandbox_project):
    root = sandbox_project
    data = root / "sandbox-data"
    (data / "climate").mkdir(parents=True)
    (data / "climate" / "note.txt").write_text("kept\n")
    (data / "geotooleval.jsonl").write_text('{"expected_answer":"secret"}\n')
    executor = CodeExecutor(root, root / "code", RuntimeSettings(sandbox="bwrap"))
    work = root / "scratch"
    result = await executor.run_shell(
        "cat /data/climate/note.txt; "
        "if [ -e /data/geotooleval.jsonl ]; then echo LEAK; exit 3; fi; "
        "if echo x > /data/climate/note.txt 2>/dev/null; then echo WROTE; exit 4; fi; "
        "echo OK",
        work=work,
        data_root=data,
        timeout=30,
    )
    assert result["exit_code"] == 0, result
    assert result["stdout"] == "kept\nOK\n"
    assert result["data_mount"] == "bwrap_ro"
    assert result["network_isolated"] is True
    assert "secret" not in result["stdout"]
    assert (data / "climate" / "note.txt").read_text() == "kept\n"
