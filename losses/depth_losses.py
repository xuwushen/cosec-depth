import torch
import torch.nn.functional as F


def masked_l1_loss(prediction, target):
    """只在有效深度像素上计算 L1 loss。

    CoSEC 的深度图中，0 表示无效深度，所以训练时需要跳过这些像素。
    """
    valid = torch.isfinite(target) & (target > 0)

    if valid.any():
        return F.l1_loss(prediction[valid], target[valid])

    return prediction.sum() * 0.0
