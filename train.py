import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from datasets import create_rgb_depth_loader
from losses import masked_l1_loss
from metrics import DepthMetricAccumulator
from models import RGBUNetDepth
from utils import (
    cleanup_distributed,
    is_main_process,
    launch_with_torchrun,
    load_checkpoint,
    load_config,
    reduce_sum,
    save_checkpoint,
    setup_distributed,
    unwrap_model,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_value(command_line_value, config, config_key, default_value):
    if command_line_value is not None:
        return command_line_value

    if config_key in config:
        return config[config_key]

    return default_value


@torch.no_grad()
def validate(model, loader, device, max_batches=None):
    model.eval()
    metrics = DepthMetricAccumulator()
    progress = tqdm(loader, desc="Validation", leave=False, disable=not is_main_process())
    inference_model = unwrap_model(model)

    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = inference_model(rgb)

        metrics.update(prediction, depth)

    metrics.synchronize(device)
    return metrics.compute()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CoSEC RGB-only depth baseline.")
    parser.add_argument("--config", default="configs/cosec.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers per GPU.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpus < 1:
        raise ValueError("--gpus must be at least 1.")

    launch_with_torchrun(args.gpus, __file__)

    config = load_config(args.config)

    if args.data_root is not None:
        config["data_root"] = args.data_root

    train_manifest = choose_value(args.train_manifest, config, "train_manifest_path", "metadata/train_split.json")
    val_manifest = choose_value(args.val_manifest, config, "val_manifest_path", "metadata/val_split.json")
    epochs = int(choose_value(args.epochs, config, "epochs", 20))
    batch_size = int(choose_value(args.batch_size, config, "batch_size", 2))
    num_workers = int(choose_value(args.num_workers, config, "num_workers", 2))
    learning_rate = float(choose_value(args.learning_rate, config, "learning_rate", 1e-4))
    checkpoint_dir = choose_value(args.checkpoint_dir, config, "checkpoint_dir", "checkpoints/rgb_unet_baseline")
    checkpoint_dir = Path(checkpoint_dir)

    device, rank, world_size = setup_distributed(args.device)
    set_seed(int(config.get("seed", 42)) + rank)

    train_dataset, train_loader = create_rgb_depth_loader(
        config=config,
        manifest_path=train_manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    val_dataset, val_loader = create_rgb_depth_loader(
        config=config,
        manifest_path=val_manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    model = RGBUNetDepth(base_channels=int(config.get("base_channels", 32))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 1
    best_rmse = math.inf

    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rmse = float(checkpoint["best_rmse"])

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)

    if is_main_process():
        print(f"device: {device}")
        print(f"world size: {world_size}")
        print(f"batch size per GPU: {batch_size}")
        print(f"global batch size: {batch_size * world_size}")
        print(f"train sequences/samples: {train_dataset.sequence_count}/{len(train_dataset)}")
        print(f"val sequences/samples: {val_dataset.sequence_count}/{len(val_dataset)}")

    try:
        for epoch in range(start_epoch, epochs + 1):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            loss_sum = 0.0
            batch_count = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", disable=not is_main_process())

            for batch_index, batch in enumerate(progress):
                if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                    break

                rgb = batch["rgb"].to(device, non_blocking=True)
                depth = batch["depth"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    prediction = model(rgb)
                    loss = masked_l1_loss(prediction, depth)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                loss_sum += float(loss.detach())
                batch_count += 1

            total_loss = reduce_sum(loss_sum, device)
            total_batches = int(reduce_sum(batch_count, device))

            if total_batches == 0:
                raise RuntimeError("No training batches were processed.")

            val_metrics = validate(model, val_loader, device, max_batches=args.max_val_batches)
            val_rmse = float(val_metrics["rmse"])

            is_best = val_rmse < best_rmse
            if is_best:
                best_rmse = val_rmse

            if is_main_process():
                checkpoint = {
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best_rmse": best_rmse,
                    "config": config,
                }
                save_checkpoint(checkpoint_dir / "last.pt", checkpoint)

                if is_best:
                    save_checkpoint(checkpoint_dir / "best.pt", checkpoint)

                train_loss = total_loss / total_batches
                print(
                    f"epoch {epoch}: "
                    f"train_loss={train_loss:.6f}, "
                    f"val_RMSE={val_rmse:.6f}, "
                    f"val_AbsRel={val_metrics['abs_rel']:.6f}, "
                    f"val_delta1={val_metrics['delta1']:.6f}, "
                    f"best_RMSE={best_rmse:.6f}"
                )

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
