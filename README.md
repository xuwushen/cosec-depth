# CoSEC Monocular Depth Estimation

This repository is the local code workspace for the CoSEC RGB + event stream monocular depth estimation competition. Code is maintained on a local Mac, pushed with Git, then pulled and run on the server.

The dataset must not be stored in this Git repository.

## Server Layout

Recommended layout:

```text
/home/yzx/code/cosec-depth        # Git code repository
/data_nvme_4tb/yzx/cosec          # CoSEC dataset directory
```

Current debug sequence:

```text
train/Day_Campus_000
```

Expected sequence tree:

```text
/data_nvme_4tb/yzx/cosec/train/Day_Campus_000/
├── img_co_left/
├── depth_co/
├── events_co_left.h5
├── timestamps.txt
└── intrinsics.json
```

## Data Conventions

RGB frames are read from `img_co_left/` and have shape `624 x 1200 x 3`.

Depth frames are read from `depth_co/`. The raw depth files are `uint16` with shape `624 x 1200`; metric depth is:

```text
metric_depth = raw_depth / 256.0
```

The event file `events_co_left.h5` contains:

```text
ms_to_idx
p
t
x
y
```

Each event is represented by `x`, `y`, `t`, and `p`. The `timestamps.txt` file maps line `i` to RGB frame `img_co_left/00000i.png`. In fixed-window mode, frame `i` uses events in:

```text
[frame_t - 50000, frame_t]
```

for `event_window_ms: 50`, assuming event timestamps are in microseconds.

## Setup

On the server:

```bash
cd /home/yzx/code/cosec-depth
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Check Dataset Tree

```bash
python scripts/check_dataset_tree.py --config configs/cosec.yaml
```

This checks the sequence directory, RGB folder, depth folder, event h5 file, timestamps, and intrinsics. It prints image counts, timestamp rows, and h5 keys/shapes/dtypes without reading all events into memory.

## Build Train Manifest

After the full training split is available on the server, scan all sequences and write a small JSON manifest:

```bash
python scripts/build_train_manifest.py --data-root /data_nvme_4tb/yzx/cosec --split train --output metadata/train_manifest.json
```

Then summarize it:

```bash
python scripts/check_manifest_summary.py --manifest metadata/train_manifest.json
```

The manifest records sequence status, missing files, RGB/depth/timestamp counts, event keys, event count, and event timestamp range. It reads h5 metadata and only the first/last event timestamps, not the full event arrays.

## Inspect One Sample

```bash
python scripts/inspect_cosec_sample.py --config configs/cosec.yaml --frame-index 10
```

This prints frame id, timestamp, RGB shape, depth shape, valid depth statistics, event count, event coordinate ranges, and polarity values.

## Visualize One Event Window

```bash
python scripts/visualize_event_window.py --config configs/cosec.yaml --frame-index 10
```

Outputs are saved under:

```text
outputs/debug_vis/frame_000010/
├── rgb_frame.png
├── depth_vis.png
├── event_window.png
└── overlay_event_on_rgb.png
```

Event visualization uses red for `p = 1` and blue for `p = 0`. Zero depth is treated as invalid when choosing the visualization range.

## Git Rules

Only commit code, configs, scripts, documentation, and small text metadata. Do not commit datasets, h5 event files, zip archives, checkpoints, submissions, logs, or debug outputs.
