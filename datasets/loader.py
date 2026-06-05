from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .cosec_multi_sequence_dataset import CoSECMultiSequenceDataset


class DistributedEvalSampler(Sampler[int]):
    """Split evaluation indices across ranks without padding duplicate samples."""

    def __init__(self, dataset: CoSECMultiSequenceDataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return max(0, (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size)


def dataset_options(config: dict[str, Any], load_events: bool = False) -> dict[str, Any]:
    return {
        "view": config.get("view", "left"),
        "image_folder": config.get("image_folder", "img_co_left"),
        "depth_folder": config.get("depth_folder", "depth_co"),
        "event_file": config.get("event_file", "events_co_left.h5"),
        "timestamp_file": config.get("timestamp_file", "timestamps.txt"),
        "intrinsics_file": config.get("intrinsics_file", "intrinsics.json"),
        "intrinsics_key": config.get("intrinsics_key", "Co_Rect_L"),
        "image_height": config.get("image_height", 624),
        "image_width": config.get("image_width", 1200),
        "depth_scale": config.get("depth_scale", 256.0),
        "event_window_mode": config.get("event_window_mode", "fixed"),
        "event_window_ms": config.get("event_window_ms", 50),
        "load_events": load_events,
    }


def collate_rgb_depth(batch: list[dict[str, Any]]) -> dict[str, Any]:
    rgb = [
        torch.from_numpy(np.array(item["rgb"], copy=True, order="C"))
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        for item in batch
    ]
    depth = [
        torch.from_numpy(np.array(item["depth"], copy=True, order="C")).unsqueeze(0).float()
        for item in batch
    ]
    return {
        "rgb": torch.stack(rgb),
        "depth": torch.stack(depth),
        "frame_id": [item["frame_id"] for item in batch],
        "sequence_name": [item["sequence_name"] for item in batch],
    }


def create_rgb_depth_loader(
    config: dict[str, Any],
    manifest_path: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[CoSECMultiSequenceDataset, DataLoader]:
    dataset = CoSECMultiSequenceDataset(
        data_root=config["data_root"],
        manifest_path=manifest_path,
        **dataset_options(config, load_events=False),
    )
    sampler: Sampler[int] | None = None
    if world_size > 1:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=int(config.get("seed", 42)),
            )
            if shuffle
            else DistributedEvalSampler(dataset, rank=rank, world_size=world_size)
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_rgb_depth,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    return dataset, loader
