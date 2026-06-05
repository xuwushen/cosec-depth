from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def scene_group(sequence_name: str) -> str:
    basename = Path(sequence_name).name
    return re.sub(r"_\d+$", "", basename)


def build_split_manifest(
    records: list[dict[str, Any]],
    subset: str,
) -> dict[str, Any]:
    return {
        "subset": subset,
        "total_sequences": len(records),
        "total_frames": sum(int(record["rgb_count"]) for record in records),
        "sequences": [
            {"sequence": record["sequence"], "status": "ok"}
            for record in records
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic sequence-level CoSEC train/validation splits.")
    parser.add_argument("--manifest", default="metadata/train_manifest.json")
    parser.add_argument("--train-output", default="metadata/train_split.json")
    parser.add_argument("--val-output", default="metadata/val_split.json")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be between 0 and 1, got {args.val_ratio}.")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid_records = [
        record
        for record in manifest.get("sequences", [])
        if record.get("status") == "ok"
        and record.get("rgb_count") == record.get("depth_count") == record.get("timestamp_count")
    ]
    if not valid_records:
        raise ValueError(f"No valid count-consistent sequences found in {manifest_path}.")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid_records:
        groups[scene_group(record["sequence"])].append(record)

    rng = random.Random(args.seed)
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    for group_name in sorted(groups):
        group_records = sorted(groups[group_name], key=lambda record: record["sequence"])
        rng.shuffle(group_records)

        if len(group_records) == 1:
            val_count = 0
        else:
            val_count = max(1, round(len(group_records) * args.val_ratio))
            val_count = min(val_count, len(group_records) - 1)

        val_records.extend(group_records[:val_count])
        train_records.extend(group_records[val_count:])

    train_records.sort(key=lambda record: record["sequence"])
    val_records.sort(key=lambda record: record["sequence"])

    train_manifest = build_split_manifest(train_records, "train")
    val_manifest = build_split_manifest(val_records, "val")

    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.write_text(json.dumps(train_manifest, indent=2), encoding="utf-8")
    val_output.write_text(json.dumps(val_manifest, indent=2), encoding="utf-8")

    print(f"Source manifest: {manifest_path}")
    print(f"Seed: {args.seed}")
    print(f"Validation ratio target: {args.val_ratio:.3f}")
    print(f"Train sequences: {len(train_records)}")
    print(f"Train frames: {train_manifest['total_frames']}")
    print(f"Validation sequences: {len(val_records)}")
    print(f"Validation frames: {val_manifest['total_frames']}")
    print(f"Train manifest: {train_output}")
    print(f"Validation manifest: {val_output}")
    print("\nValidation sequences by scene group:")
    for record in val_records:
        print(f"  {scene_group(record['sequence'])}: {record['sequence']} ({record['rgb_count']} frames)")


if __name__ == "__main__":
    main()
