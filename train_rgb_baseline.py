from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datasets import CoSECDataset


class TinyRGBDepthNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        features = self.encoder(rgb)
        pred = self.decoder(features)
        return F.interpolate(pred, size=rgb.shape[-2:], mode="bilinear", align_corners=False)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def collate_batch(batch: list[dict]) -> dict:
    rgb_tensors = []
    depth_tensors = []

    for item in batch:
        rgb = np.ascontiguousarray(item["rgb"])
        depth = np.ascontiguousarray(item["depth"])

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB shape HWC with 3 channels, got {rgb.shape}.")
        if depth.ndim != 2:
            raise ValueError(f"Expected depth shape HW, got {depth.shape}.")

        rgb_tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
        depth_tensors.append(torch.from_numpy(depth).unsqueeze(0).float())

    return {
        "rgb": torch.stack(rgb_tensors, dim=0),
        "depth": torch.stack(depth_tensors, dim=0),
        "frame_id": [item["frame_id"] for item in batch],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RGB-only CoSEC depth baseline.")
    parser.add_argument("--config", default="configs/cosec.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Read one batch, run one forward pass, then exit.")
    args = parser.parse_args()

    if not args.dry_run:
        raise SystemExit("Refusing to start formal training. Pass --dry-run for the initial one-batch check.")

    cfg = load_config(args.config)
    dataset = CoSECDataset(
        data_root=args.data_root or cfg["data_root"],
        sequence=args.sequence or cfg["sequence"],
        event_window_ms=cfg.get("event_window_ms", 50),
        event_window_mode=cfg.get("event_window_mode", "fixed"),
        view=cfg.get("view", "left"),
        depth_scale=cfg.get("depth_scale", 256.0),
        image_folder=cfg.get("image_folder", "img_co_left"),
        event_file=cfg.get("event_file", "events_co_left.h5"),
        depth_folder=cfg.get("depth_folder", "depth_co"),
        timestamp_file=cfg.get("timestamp_file", "timestamps.txt"),
        intrinsics_file=cfg.get("intrinsics_file", "intrinsics.json"),
        intrinsics_key=cfg.get("intrinsics_key", "Co_Rect_L"),
        image_height=cfg.get("image_height", 624),
        image_width=cfg.get("image_width", 1200),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or cfg.get("batch_size", 1),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0),
        collate_fn=collate_batch,
    )

    model = TinyRGBDepthNet()
    batch = next(iter(loader))
    rgb = batch["rgb"]
    depth = batch["depth"]
    pred = model(rgb)

    valid = torch.isfinite(depth) & (depth > 0)
    if valid.any():
        loss = F.l1_loss(pred[valid], depth[valid])
    else:
        loss = F.l1_loss(pred, depth)

    print(f"rgb batch shape: {tuple(rgb.shape)}")
    print(f"depth batch shape: {tuple(depth.shape)}")
    print(f"prediction shape: {tuple(pred.shape)}")
    print(f"loss value: {float(loss.detach()):.6f}")


if __name__ == "__main__":
    main()
