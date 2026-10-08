"""Package prepared, checksum-verified runtime data for the server."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "dist/agentgs-data.tar")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Preserve previous bundles; choose a fresh output")
    if not output.is_relative_to(root / "dist"):
        parser.error("Bundle output must be inside dist/")
    resources = json.loads((root / "data/processed/resource_manifest.json").read_text())
    version = json.loads((root / "data/geotooleval/version_manifest.json").read_text())
    records = {row["path"]: row["sha256"] for row in resources["records"]}
    records.update({row["path"]: row["sha256"] for row in version["records"]})
    for relative in (
        "data/processed/resource_manifest.json",
        "data/geotooleval/version_manifest.json",
        "data/geotooleval/GeoToolEval.xlsx",
        "data/geotooleval/10.自然灾害与极端地震.xlsx",
        "data/geoquestions1089/processed/geobigdata_snapshot.json",
        "data/usgs/usgs_2010_2020_m7.csv",
        "data/snapo/113597.xlsx",
    ):
        records[relative] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    exported = []
    try:
        with tarfile.open(temporary, "w") as bundle:
            for relative, expected in sorted(records.items()):
                if (
                    Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                    or not relative.startswith("data/")
                ):
                    raise ValueError(f"Invalid bundle member: {relative}")
                path = root / relative
                with path.open("rb") as stream:
                    fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
                if expected and fingerprint != expected:
                    raise ValueError(f"Prepared data differs: {relative}")
                bundle.add(path, arcname=relative, recursive=False)
                exported.append(
                    {"path": relative, "sha256": fingerprint, "bytes": path.stat().st_size}
                )
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with output.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {
        "bundle": str(output.relative_to(root)),
        "sha256": fingerprint,
        "bytes": output.stat().st_size,
        "files": exported,
        "note": "Prepared runtime/evaluator data only; no secrets, old source code, local smoke fixtures or downloaded archives.",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "bundle": str(output),
                "files": len(exported),
                "bytes": receipt["bytes"],
                "sha256": fingerprint,
            }
        )
    )


if __name__ == "__main__":
    main()
