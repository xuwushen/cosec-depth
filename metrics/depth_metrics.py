from __future__ import annotations

import torch
import torch.distributed as dist


class DepthMetricAccumulator:
    """按有效像素累计深度指标，适用于完整验证集。"""

    def __init__(
        self,
        min_depth: float = 0.0,
        max_depth: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.squared_error_sum = 0.0
        self.abs_rel_sum = 0.0
        self.delta1_count = 0
        self.delta2_count = 0
        self.delta3_count = 0
        self.valid_pixels = 0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(prediction.shape)} does not match target shape {tuple(target.shape)}."
            )

        valid = torch.isfinite(target) & (target > self.min_depth)
        if self.max_depth is not None:
            valid &= target <= self.max_depth

        valid_count = int(valid.sum().item())
        if valid_count == 0:
            return

        pred = prediction[valid].double().clamp_min(self.eps)
        gt = target[valid].double()
        error = pred - gt
        ratio = torch.maximum(pred / gt.clamp_min(self.eps), gt / pred)

        self.squared_error_sum += float(error.square().sum().item())
        self.abs_rel_sum += float((torch.abs(error) / gt.clamp_min(self.eps)).sum().item())
        self.delta1_count += int((ratio < 1.25).sum().item())
        self.delta2_count += int((ratio < 1.25**2).sum().item())
        self.delta3_count += int((ratio < 1.25**3).sum().item())
        self.valid_pixels += valid_count

    def compute(self) -> dict[str, float | int]:
        if self.valid_pixels == 0:
            raise ValueError("No valid target depth pixels were accumulated.")

        return {
            "rmse": (self.squared_error_sum / self.valid_pixels) ** 0.5,
            "abs_rel": self.abs_rel_sum / self.valid_pixels,
            "delta1": self.delta1_count / self.valid_pixels,
            "delta2": self.delta2_count / self.valid_pixels,
            "delta3": self.delta3_count / self.valid_pixels,
            "valid_pixels": self.valid_pixels,
        }

    def synchronize(self, device: torch.device) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        values = torch.tensor(
            [
                self.squared_error_sum,
                self.abs_rel_sum,
                self.delta1_count,
                self.delta2_count,
                self.delta3_count,
                self.valid_pixels,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        (
            self.squared_error_sum,
            self.abs_rel_sum,
            delta1_count,
            delta2_count,
            delta3_count,
            valid_pixels,
        ) = values.tolist()
        self.delta1_count = int(delta1_count)
        self.delta2_count = int(delta2_count)
        self.delta3_count = int(delta3_count)
        self.valid_pixels = int(valid_pixels)


def compute_depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    min_depth: float = 0.0,
    max_depth: float | None = None,
    eps: float = 1e-6,
) -> dict[str, float | int]:
    """计算 CoSEC 官方深度指标，仅统计有效 GT 深度像素。"""
    accumulator = DepthMetricAccumulator(min_depth=min_depth, max_depth=max_depth, eps=eps)
    accumulator.update(prediction, target)
    return accumulator.compute()
