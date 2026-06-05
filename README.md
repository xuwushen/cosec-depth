# CoSEC 单目深度估计 Baseline

这是一个干净的 CoSEC RGB-only 深度估计 baseline。它的目标是先把数据读取、训练、验证和 checkpoint 流程跑通，方便后续替换更强模型或加入 event 分支。

当前版本默认只使用左视图 RGB 和深度监督。event 文件的读取函数保留在 dataset 中，后续做 RGB + event 融合时可以继续用。

## 项目结构

```text
configs/
  cosec.yaml                  # 默认配置
datasets/
  cosec_dataset.py            # 读取单个 sequence
  cosec_multi_sequence_dataset.py
                              # 读取多个 sequence
  loader.py                   # DataLoader 和 batch 拼接
models/
  rgb_unet.py                 # 简单 RGB U-Net baseline
losses/
  depth_losses.py             # 深度估计训练损失
metrics/
  depth_metrics.py            # RMSE、AbsRel、delta 指标
utils/
  config.py                   # 读取 yaml
  checkpoint.py               # 保存/加载 checkpoint
  distributed.py              # 单卡/多卡训练工具
scripts/
  build_train_manifest.py     # 扫描数据集，生成 sequence 清单
  create_sequence_splits.py   # 划分 train/val sequence
metadata/
  train_split.json            # 默认训练划分
  val_split.json              # 默认验证划分
train.py                      # 训练入口
evaluate.py                   # 验证入口
requirements.txt
```

## 数据目录

代码仓库不保存 CoSEC 数据集。推荐结构：

```text
/path/to/code/cosec-depth      # 本代码仓库
/path/to/CoSEC                 # CoSEC 数据集
```

CoSEC 训练集 sequence 示例：

```text
/path/to/CoSEC/train/Day_Campus_000/
  img_co_left/
  depth_co/
  events_co_left.h5
  timestamps.txt
  intrinsics.json
```

## 数据格式

RGB 图像：

```text
img_co_left/000000.png
shape: 624 x 1200 x 3
dtype: uint8
```

深度图：

```text
depth_co/000000.png
shape: 624 x 1200
dtype: uint16
```

深度值需要转换成米：

```text
depth_m = depth_raw / 256.0
```

深度为 `0` 的像素视为无效像素，训练 loss 和验证指标都会忽略这些位置。

event 文件：

```text
events_co_left.h5
keys: ms_to_idx, x, y, t, p
```

`timestamps.txt` 中第 `i` 行对应 `img_co_left/{i:06d}.png`。时间戳单位是微秒。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 生成数据清单

如果第一次在服务器上使用，可以先扫描训练集：

```bash
python scripts/build_train_manifest.py \
  --data-root /path/to/CoSEC \
  --split train \
  --output metadata/train_manifest.json
```

然后划分训练集和验证集：

```bash
python scripts/create_sequence_splits.py \
  --manifest metadata/train_manifest.json \
  --train-output metadata/train_split.json \
  --val-output metadata/val_split.json
```

仓库中已经提供了一份默认 `train_split.json` 和 `val_split.json`，方便直接复现实验。

## 快速训练测试

先跑一个很小的 smoke test，确认数据、模型、loss、验证都能工作：

```bash
python train.py \
  --data-root /path/to/CoSEC \
  --gpus 1 \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 2 \
  --max-train-batches 2 \
  --max-val-batches 2
```

单卡正式训练示例：

```bash
python train.py \
  --data-root /path/to/CoSEC \
  --gpus 1 \
  --epochs 20 \
  --batch-size 2 \
  --num-workers 4
```

多卡训练示例：

```bash
python train.py \
  --data-root /path/to/CoSEC \
  --gpus 8 \
  --epochs 20 \
  --batch-size 2 \
  --num-workers 2
```

注意：`batch-size` 是每张 GPU 的 batch size。`--gpus 8 --batch-size 2` 表示总 batch size 为 16。

## 验证模型

```bash
python evaluate.py \
  --data-root /path/to/CoSEC \
  --checkpoint checkpoints/rgb_unet_baseline/best.pt \
  --manifest metadata/val_split.json \
  --gpus 1 \
  --batch-size 2 \
  --num-workers 4
```

验证会输出：

```text
RMSE
AbsRel
delta1
delta2
delta3
```

比赛主要看 RMSE，越低越好。

## 后续改进方向

建议按顺序改：

1. 替换 `models/rgb_unet.py`，换成更强 encoder-decoder。
2. 调整 loss，例如 L1 + RMSE/MSE 或深度梯度约束。
3. 加入数据增强。
4. 最后再加入 event 分支，做 RGB + event 融合。

## Git 注意事项

不要提交：

```text
CoSEC 数据集
events_co_left.h5
zip 文件
checkpoint
outputs
logs
submissions
```

`.gitignore` 已经默认忽略这些大文件和输出目录。
