from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def count_images(directory: Path) -> int:
    return sum(1 for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def print_h5_info(path: Path) -> None:
    print(f"Event file: {path}")
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        print(f"H5 keys: {keys}")
        for key in keys:
            obj = handle[key]
            if isinstance(obj, h5py.Dataset):
                print(f"  {key}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"  {key}: group")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a CoSEC sequence tree without reading full event arrays.")
    parser.add_argument("--config", default="configs/cosec.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sequence", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(args.data_root or cfg["data_root"])
    sequence = args.sequence or cfg["sequence"]

    sequence_dir = data_root / sequence
    image_dir = sequence_dir / cfg.get("image_folder", "img_co_left")
    depth_dir = sequence_dir / cfg.get("depth_folder", "depth_co")
    event_path = sequence_dir / cfg.get("event_file", "events_co_left.h5")
    timestamp_path = sequence_dir / cfg.get("timestamp_file", "timestamps.txt")
    intrinsics_path = sequence_dir / cfg.get("intrinsics_file", "intrinsics.json")

    require_dir(sequence_dir, "sequence directory")
    require_dir(image_dir, "RGB image folder")
    require_dir(depth_dir, "depth folder")
    require_file(event_path, "event h5 file")
    require_file(timestamp_path, "timestamp file")
    require_file(intrinsics_path, "intrinsics file")

    timestamp_count = sum(1 for line in timestamp_path.read_text().splitlines() if line.strip())

    print(f"Data root: {data_root}")
    print(f"Sequence: {sequence}")
    print(f"Sequence directory: {sequence_dir}")
    print(f"RGB directory: {image_dir}")
    print(f"RGB image count: {count_images(image_dir)}")
    print(f"Depth directory: {depth_dir}")
    print(f"Depth image count: {count_images(depth_dir)}")
    print(f"Timestamp file: {timestamp_path}")
    print(f"Timestamp rows: {timestamp_count}")
    print(f"Intrinsics file: {intrinsics_path}")
    print_h5_info(event_path)


if __name__ == "__main__":
    main()
