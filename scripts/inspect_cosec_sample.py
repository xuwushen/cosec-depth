from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import CoSECDataset


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_dataset(cfg: dict) -> CoSECDataset:
    return CoSECDataset(
        data_root=cfg["data_root"],
        sequence=cfg["sequence"],
        view=cfg.get("view", "left"),
        image_folder=cfg.get("image_folder", "img_co_left"),
        depth_folder=cfg.get("depth_folder", "depth_co"),
        event_file=cfg.get("event_file", "events_co_left.h5"),
        timestamp_file=cfg.get("timestamp_file", "timestamps.txt"),
        intrinsics_file=cfg.get("intrinsics_file", "intrinsics.json"),
        intrinsics_key=cfg.get("intrinsics_key", "Co_Rect_L"),
        image_height=cfg.get("image_height", 624),
        image_width=cfg.get("image_width", 1200),
        depth_scale=cfg.get("depth_scale", 256.0),
        event_window_mode=cfg.get("event_window_mode", "fixed"),
        event_window_ms=cfg.get("event_window_ms", 50),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one CoSEC RGB/depth/event sample.")
    parser.add_argument("--config", default="configs/cosec.yaml")
    parser.add_argument("--frame-index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = build_dataset(cfg)
    sample = dataset[args.frame_index]

    rgb = sample["rgb"]
    depth = sample["depth"]
    events = sample["events"]
    event_count = len(events["t"])
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]

    print(f"Dataset length: {len(dataset)}")
    print(f"frame_id: {sample['frame_id']}")
    print(f"timestamp: {sample['timestamp']}")
    print(f"rgb shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"depth shape: {depth.shape}, dtype: {depth.dtype}")
    if valid_depth.size > 0:
        print(f"depth min: {float(valid_depth.min()):.6f}")
        print(f"depth max: {float(valid_depth.max()):.6f}")
        print(f"depth mean: {float(valid_depth.mean()):.6f}")
    else:
        print("depth stats: no valid depth values greater than 0")
    print(f"event number: {event_count}")
    if event_count > 0:
        print(f"event x range: [{int(events['x'].min())}, {int(events['x'].max())}]")
        print(f"event y range: [{int(events['y'].min())}, {int(events['y'].max())}]")
        print(f"event p unique: {np.unique(events['p']).tolist()}")
    else:
        print("event stats: empty event window")
    print(f"rgb_path: {sample['rgb_path']}")
    print(f"depth_path: {sample['depth_path']}")
    print(f"sequence_name: {sample['sequence_name']}")
    print(f"intrinsics:\n{sample['intrinsics']}")


if __name__ == "__main__":
    main()
