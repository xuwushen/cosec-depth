from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
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
        image_folder=cfg.get("image_folder", "img_co_left"),
        depth_folder=cfg.get("depth_folder", "depth_co"),
        event_file=cfg.get("event_file", "events_co_left.h5"),
        timestamp_file=cfg.get("timestamp_file", "timestamps.txt"),
        depth_scale=cfg.get("depth_scale", 256.0),
        event_window_mode=cfg.get("event_window_mode", "fixed"),
        event_window_ms=cfg.get("event_window_ms", 50),
    )


def depth_to_color(depth: np.ndarray) -> np.ndarray:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        hi = lo + 1.0

    normalized = np.zeros_like(depth, dtype=np.float32)
    valid_mask = np.isfinite(depth) & (depth > 0)
    normalized[valid_mask] = np.clip((depth[valid_mask] - lo) / (hi - lo), 0.0, 1.0)
    depth_u8 = (normalized * 255).astype(np.uint8)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    color[~valid_mask] = 0
    return color


def events_to_image(events: dict[str, np.ndarray], height: int, width: int) -> np.ndarray:
    event_image = np.zeros((height, width, 3), dtype=np.uint8)
    if len(events["t"]) == 0:
        cv2.putText(
            event_image,
            "empty event window",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return event_image

    x = np.clip(events["x"].astype(np.int64), 0, width - 1)
    y = np.clip(events["y"].astype(np.int64), 0, height - 1)
    p = events["p"]

    pos = p == 1
    neg = p == 0
    event_image[y[pos], x[pos], 2] = 255
    event_image[y[neg], x[neg], 0] = 255
    return event_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one CoSEC event window.")
    parser.add_argument("--config", default="configs/cosec.yaml")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = build_dataset(cfg)
    sample = dataset[args.frame_index]

    rgb = sample["rgb"]
    depth = sample["depth"]
    events = sample["events"]
    output_root = Path(args.output_dir or cfg.get("output_dir", "outputs/debug_vis"))
    frame_dir = output_root / f"frame_{args.frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_vis = depth_to_color(depth)
    event_vis = events_to_image(events, rgb.shape[0], rgb.shape[1])
    overlay = cv2.addWeighted(rgb_bgr, 0.85, event_vis, 0.15, 0.0)

    paths = {
        "rgb_frame": frame_dir / "rgb_frame.png",
        "depth_vis": frame_dir / "depth_vis.png",
        "event_window": frame_dir / "event_window.png",
        "overlay_event_on_rgb": frame_dir / "overlay_event_on_rgb.png",
    }

    cv2.imwrite(str(paths["rgb_frame"]), rgb_bgr)
    cv2.imwrite(str(paths["depth_vis"]), depth_vis)
    cv2.imwrite(str(paths["event_window"]), event_vis)
    cv2.imwrite(str(paths["overlay_event_on_rgb"]), overlay)

    print(f"frame_id: {sample['frame_id']}")
    print(f"timestamp: {sample['timestamp']}")
    print(f"event number: {len(events['t'])}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
