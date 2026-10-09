"""Sample a GeoToolEval subset with the same count from each field.

A field is ``category_id``, one of the ten scientific domains. Inside a field
the draw is uniform. Fisher-Yates uses only ``random.Random.random`` so the
same seed yields the same question ids on different Python versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def _repository_root(start: Path) -> Path:
    candidate = start.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"No pyproject.toml found above {candidate}")


REPO = _repository_root(Path(__file__))

FIELD_NAMES = {
    1: "DeepEarthTools",
    2: "GeoBigDataTools",
    3: "MineralResourcesTools",
    4: "ClimateChangeTools",
    5: "EnergyResourcesTools",
    6: "OceanCarbonTools",
    7: "DeepTimeBiosphereTools",
    8: "CCSTools",
    9: "RemoteSensingTools",
    10: "NaturalHazardTools",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def load_rows(path: Path) -> list[tuple[dict, str]]:
    rows: list[tuple[dict, str]] = []
    seen: set[int] = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or "question_id" not in row or "category_id" not in row:
                raise SystemExit(f"{path}:{line_number}: missing question_id or category_id")
            question_id = int(row["question_id"])
            category_id = int(row["category_id"])
            if question_id in seen:
                raise SystemExit(f"duplicate question_id {question_id}")
            seen.add(question_id)
            row["question_id"] = question_id
            row["category_id"] = category_id
            rows.append((row, text))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return rows


def allocate(available: dict[int, int], size: int) -> dict[int, int]:
    """Give every field the same count, then one extra to the smallest fields."""
    if size < 1:
        raise SystemExit("--size must be at least 1")
    total = sum(available.values())
    if size > total:
        raise SystemExit(f"--size {size} exceeds {total} rows")
    fields = sorted(available)
    assigned = {field: 0 for field in fields}
    for _ in range(size):
        field = min(
            (item for item in fields if assigned[item] < available[item]),
            key=lambda item: (assigned[item], item),
        )
        assigned[field] += 1
    return assigned


def choose(population: list[tuple[dict, str]], count: int, rng: random.Random):
    """Uniform subset. Population order must already be stable."""
    pool = list(population)
    for index in range(len(pool) - 1, 0, -1):
        swap = int(rng.random() * (index + 1))
        pool[index], pool[swap] = pool[swap], pool[index]
    return pool[:count]


def sample(rows: list[tuple[dict, str]], size: int, seed: int) -> list[tuple[dict, str]]:
    grouped: dict[int, list[tuple[dict, str]]] = {}
    for item in rows:
        grouped.setdefault(item[0]["category_id"], []).append(item)
    for field_rows in grouped.values():
        field_rows.sort(key=lambda item: item[0]["question_id"])
    quotas = allocate({field: len(field_rows) for field, field_rows in grouped.items()}, size)
    rng = random.Random(seed)
    selected: list[tuple[dict, str]] = []
    for field in sorted(grouped):
        selected.extend(choose(grouped[field], quotas[field], rng))
    selected.sort(key=lambda item: item[0]["question_id"])
    return selected


def write_subset(dataset: Path, output: Path, manifest_path: Path, size: int, seed: int) -> None:
    rows = load_rows(dataset)
    selected = sample(rows, size, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{text}\n" for _, text in selected), encoding="utf-8")

    available: dict[int, int] = {}
    for row, _ in rows:
        available[row["category_id"]] = available.get(row["category_id"], 0) + 1
    by_field: dict[int, list[int]] = {}
    for row, _ in selected:
        by_field.setdefault(row["category_id"], []).append(row["question_id"])
    fields = []
    for field in sorted(available):
        question_ids = by_field.get(field, [])
        fields.append(
            {
                "category_id": field,
                "name": FIELD_NAMES.get(field, f"Category{field}"),
                "available": available[field],
                "selected": len(question_ids),
                "question_ids": question_ids,
            }
        )
    manifest = {
        "dataset": "geotooleval",
        "field": "category_id",
        "size": len(selected),
        "seed": seed,
        "source": repo_relative(dataset),
        "source_sha256": sha256(dataset),
        "output": repo_relative(output),
        "output_sha256": sha256(output),
        "fields": fields,
        "question_ids": [row["question_id"] for row, _ in selected],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = ", ".join(f"{item['category_id']}:{item['selected']}" for item in fields)
    print(f"wrote {len(selected)} rows to {repo_relative(output)}")
    print(f"manifest {repo_relative(manifest_path)}")
    print(f"per field [{counts}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "data/geotooleval.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "data/ablation/geotooleval_60.jsonl",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    write_subset(args.dataset, args.output, manifest, args.size, args.seed)


if __name__ == "__main__":
    main()
