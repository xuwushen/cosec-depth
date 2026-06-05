from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def launch_with_torchrun(gpus: int, script_path: str | Path) -> None:
    if gpus <= 1 or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return
    if gpus > torch.cuda.device_count():
        raise ValueError(f"Requested {gpus} GPUs, but only {torch.cuda.device_count()} are visible.")

    child_args: list[str] = []
    skip_next = False
    for argument in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument == "--gpus":
            skip_next = True
            continue
        if argument.startswith("--gpus="):
            continue
        child_args.append(argument)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        str(Path(script_path).resolve()),
        "--gpus",
        str(gpus),
        *child_args,
    ]
    raise SystemExit(subprocess.call(command))


def setup_distributed(requested_device: str | None = None) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA with the NCCL backend.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), dist.get_rank(), world_size

    device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return device, 0, 1


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())
