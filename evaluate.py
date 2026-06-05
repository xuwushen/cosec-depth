import argparse

import torch
from tqdm import tqdm

from datasets import create_rgb_depth_loader
from metrics import DepthMetricAccumulator
from models import RGBUNetDepth
from utils import (
    cleanup_distributed,
    is_main_process,
    launch_with_torchrun,
    load_checkpoint,
    load_config,
    reduce_sum,
    setup_distributed,
)


def choose_value(command_line_value, config, config_key, default_value):
    if command_line_value is not None:
        return command_line_value

    if config_key in config:
        return config[config_key]

    return default_value


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the CoSEC RGB-only depth baseline.")
    parser.add_argument("--config", default="configs/cosec.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers per GPU.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpus < 1:
        raise ValueError("--gpus must be at least 1.")

    launch_with_torchrun(args.gpus, __file__)

    config = load_config(args.config)

    if args.data_root is not None:
        config["data_root"] = args.data_root

    manifest = choose_value(args.manifest, config, "val_manifest_path", "metadata/val_split.json")
    batch_size = int(choose_value(args.batch_size, config, "batch_size", 2))
    num_workers = int(choose_value(args.num_workers, config, "num_workers", 2))

    device, rank, world_size = setup_distributed(args.device)

    dataset, loader = create_rgb_depth_loader(
        config=config,
        manifest_path=manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    checkpoint = load_checkpoint(args.checkpoint, device)
    model = RGBUNetDepth(base_channels=int(config.get("base_channels", 32))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metrics = DepthMetricAccumulator()
    sample_count = 0

    with torch.inference_mode():
        progress = tqdm(loader, desc="Evaluating", disable=not is_main_process())

        for batch_index, batch in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(rgb)

            metrics.update(prediction, depth)
            sample_count += int(rgb.shape[0])

    metrics.synchronize(device)
    sample_count = int(reduce_sum(sample_count, device))
    result = metrics.compute()

    if is_main_process():
        print(f"device: {device}")
        print(f"world size: {world_size}")
        print(f"manifest: {manifest}")
        print(f"dataset sequences/samples: {dataset.sequence_count}/{len(dataset)}")
        print(f"evaluated samples: {sample_count}")
        print(f"valid pixels: {result['valid_pixels']}")
        print(f"RMSE: {result['rmse']:.6f}")
        print(f"AbsRel: {result['abs_rel']:.6f}")
        print(f"delta1: {result['delta1']:.6f}")
        print(f"delta2: {result['delta2']:.6f}")
        print(f"delta3: {result['delta3']:.6f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
