from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py


REQUIRED_ITEMS = {
    "img_co_left": "dir",
    "depth_co": "dir",
    "events_co_left.h5": "file",
    "timestamps.txt": "file",
    "intrinsics.json": "file",
}


def count_images(directory: Path) -> int:
    return sum(1 for path in directory.glob("*.png") if path.is_file())


def count_timestamps(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def missing_items(sequence_dir: Path) -> list[str]:
    missing: list[str] = []
    for name, kind in REQUIRED_ITEMS.items():
        path = sequence_dir / name
        exists = path.is_dir() if kind == "dir" else path.is_file()
        if not exists:
            missing.append(name)
    return missing


def inspect_event_file(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        info: dict[str, Any] = {
            "event_keys": keys,
            "event_datasets": {},
            "event_count": None,
            "event_t_min": None,
            "event_t_max": None,
        }

        for key in keys:
            obj = handle[key]
            if isinstance(obj, h5py.Dataset):
                info["event_datasets"][key] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }

        if "t" in handle and isinstance(handle["t"], h5py.Dataset):
            t_dataset = handle["t"]
            event_count = int(t_dataset.shape[0])
            info["event_count"] = event_count
            if event_count > 0:
                info["event_t_min"] = float(t_dataset[0])
                info["event_t_max"] = float(t_dataset[event_count - 1])

    return info


def inspect_sequence(split: str, sequence_dir: Path) -> dict[str, Any]:
    sequence_name = f"{split}/{sequence_dir.name}"
    record: dict[str, Any] = {
        "sequence": sequence_name,
        "sequence_dir": str(sequence_dir),
        "status": "ok",
        "missing_items": [],
        "rgb_count": None,
        "depth_count": None,
        "timestamp_count": None,
        "event_keys": [],
        "event_count": None,
        "event_t_min": None,
        "event_t_max": None,
    }

    missing = missing_items(sequence_dir)
    if missing:
        record["status"] = "missing_files"
        record["missing_items"] = missing
        return record

    record["rgb_count"] = count_images(sequence_dir / "img_co_left")
    record["depth_count"] = count_images(sequence_dir / "depth_co")
    record["timestamp_count"] = count_timestamps(sequence_dir / "timestamps.txt")

    event_info = inspect_event_file(sequence_dir / "events_co_left.h5")
    record.update(
        {
            "event_keys": event_info["event_keys"],
            "event_count": event_info["event_count"],
            "event_t_min": event_info["event_t_min"],
            "event_t_max": event_info["event_t_max"],
            "event_datasets": event_info["event_datasets"],
        }
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CoSEC training split manifest.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="metadata/train_manifest.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    sequence_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    records = [inspect_sequence(args.split, sequence_dir) for sequence_dir in sequence_dirs]

    ok_count = sum(1 for record in records if record["status"] == "ok")
    missing_count = sum(1 for record in records if record["status"] == "missing_files")

    manifest = {
        "data_root": str(data_root),
        "split": args.split,
        "total_sequences": len(records),
        "ok_sequences": ok_count,
        "missing_sequences": missing_count,
        "sequences": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Data root: {data_root}")
    print(f"Split: {args.split}")
    print(f"Total sequences: {len(records)}")
    print(f"OK sequences: {ok_count}")
    print(f"Missing sequences: {missing_count}")
    print(f"Manifest written to: {output_path}")


if __name__ == "__main__":
    main()
