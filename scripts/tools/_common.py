"""Versioned tool installation. No global PATH or system package modifications."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("No project root")


def platform_id():
    arch = {"x86_64": "x64", "AMD64": "x64", "aarch64": "arm64", "ARM64": "arm64"}.get(
        platform.machine()
    )
    return ("win" if os.name == "nt" else "linux") + "-" + str(arch)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def install(name):
    parser = argparse.ArgumentParser(description=f"Install versioned {name} under tools/")
    parser.add_argument("--output", default="tools", help="Relative to project root")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Installation output must stay inside project")
    manifest = json.loads((Path(__file__).parent / "manifests" / f"{name}.json").read_text())
    target_platform = platform_id()
    item = manifest["platforms"].get(target_platform)
    if not item:
        raise RuntimeError(f"{name} has no installer for {target_platform}; use WSL on Windows")
    destination = output / "packages" / name / manifest["version"] / target_platform
    receipt = output / "state" / f"{name}.json"
    binary = output / "bin" / (name + ".exe" if os.name == "nt" else name)
    if args.verify:
        if not receipt.exists() or not binary.exists():
            raise RuntimeError(f"{name} is not installed")
        recorded = json.loads(receipt.read_text())
        if recorded["version"] != manifest["version"] or sha(binary) != recorded["binary_sha256"]:
            raise RuntimeError("Installed binary differs from receipt")
        subprocess.run([str(binary), "--version"], check=True)
        return
    if receipt.exists() and binary.exists():
        recorded = json.loads(receipt.read_text())
        if recorded["version"] == manifest["version"] and sha(binary) == recorded["binary_sha256"]:
            print(f"{name}: already installed")
            return
    temporary = root / "tmp" / "tools" / name
    temporary.mkdir(parents=True, exist_ok=True)
    archive = temporary / item["archive"]
    if not archive.exists() or sha(archive) != item["sha256"]:
        part = archive.with_suffix(".part")
        with urllib.request.urlopen(item["url"], timeout=120) as src, part.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        if sha(part) != item["sha256"]:
            raise RuntimeError("Download checksum mismatch")
        part.replace(archive)
    staging = temporary / "staging"
    staging.mkdir(exist_ok=True)
    if archive.name.endswith(".deb"):
        if shutil.which("dpkg-deb") is None:
            raise RuntimeError("dpkg-deb is required on Ubuntu")
        subprocess.run(["dpkg-deb", "-x", str(archive), str(staging)], check=True)
        for dependency in manifest.get("dependencies", []):
            dependency_path = temporary / dependency["archive"]
            if not dependency_path.exists() or sha(dependency_path) != dependency["sha256"]:
                with (
                    urllib.request.urlopen(dependency["url"], timeout=120) as src,
                    dependency_path.open("wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
            if sha(dependency_path) != dependency["sha256"]:
                raise RuntimeError("Dependency checksum mismatch")
            subprocess.run(["dpkg-deb", "-x", str(dependency_path), str(staging)], check=True)
        executable = staging / "usr" / "bin" / name
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                if not (staging / member.filename).resolve().is_relative_to(staging.resolve()):
                    raise ValueError("Unsafe archive member")
            source.extractall(staging)
        executable = next(staging.glob("node-*/node.exe"))
    else:
        with tarfile.open(archive) as source:
            source.extractall(staging, filter="data")
        executable = next(staging.glob("node-*/bin/node"))
    verify_env = dict(os.environ)
    if name == "proot":
        verify_env["LD_LIBRARY_PATH"] = str(staging / "usr/lib/x86_64-linux-gnu")
    subprocess.run([str(executable), "--version"], check=True, env=verify_env)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(
            "Existing installation is damaged; preserve it and choose a fresh --output"
        )
    staging.replace(destination)
    relative = executable.relative_to(staging)
    (output / "bin").mkdir(parents=True, exist_ok=True)
    temporary_bin = output / "bin" / ("." + name + ".new")
    if temporary_bin.exists() or temporary_bin.is_symlink():
        temporary_bin.unlink()
    if name == "proot":
        import shlex

        library = shlex.quote(str(destination / "usr/lib/x86_64-linux-gnu"))
        real_binary = shlex.quote(str(destination / relative))
        temporary_bin.write_text(
            f'#!/bin/sh\nexec /lib64/ld-linux-x86-64.so.2 --library-path {library} {real_binary} "$@"\n'
        )
        temporary_bin.chmod(0o755)
        rootfs = output / "share" / "rootfs"
        for directory in ["usr", "lib", "lib64", "work", "tmp"]:
            (rootfs / directory).mkdir(parents=True, exist_ok=True)
    elif os.name == "nt":
        shutil.copy2(destination / relative, temporary_bin)
    else:
        temporary_bin.symlink_to(destination / relative)
    temporary_bin.replace(binary)
    if name == "node":
        package = (
            (destination / relative).parent.parent
            if os.name != "nt"
            else (destination / relative).parent
        )
        npm = package / (
            "lib/node_modules/npm/bin/npm-cli.js"
            if os.name != "nt"
            else "node_modules/npm/bin/npm-cli.js"
        )
        npm_path = output / "state" / "npm-path.txt"
        npm_path.parent.mkdir(parents=True, exist_ok=True)
        npm_path.write_text(str(npm))
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "name": name,
                "version": manifest["version"],
                "platform": target_platform,
                "source": item["url"],
                "archive_sha256": item["sha256"],
                "binary_sha256": sha(binary),
            },
            indent=2,
        )
    )
    print(f"Installed {name} at {binary}")
