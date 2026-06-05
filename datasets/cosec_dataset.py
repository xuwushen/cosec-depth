import json
import warnings
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


class CoSECDataset:
    """读取一个 CoSEC sequence。

    返回的数据保持简单：
    - rgb: numpy array, HWC, uint8
    - depth: numpy array, HW, float32, 单位为米
    - events: dict 或 None
    """

    def __init__(
        self,
        data_root,
        sequence,
        view="left",
        image_folder="img_co_left",
        depth_folder="depth_co",
        event_file="events_co_left.h5",
        timestamp_file="timestamps.txt",
        intrinsics_file="intrinsics.json",
        intrinsics_key="Co_Rect_L",
        image_height=624,
        image_width=1200,
        depth_scale=256.0,
        event_window_mode="fixed",
        event_window_ms=50,
        load_events=True,
    ):
        if view != "left":
            raise ValueError("This baseline only supports left view.")

        self.data_root = Path(data_root)
        self.sequence = sequence
        self.sequence_dir = self.data_root / sequence

        self.image_dir = self.sequence_dir / image_folder
        self.depth_dir = self.sequence_dir / depth_folder
        self.event_path = self.sequence_dir / event_file
        self.timestamp_path = self.sequence_dir / timestamp_file
        self.intrinsics_path = self.sequence_dir / intrinsics_file

        self.intrinsics_key = intrinsics_key
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.depth_scale = float(depth_scale)
        self.event_window_mode = event_window_mode
        self.event_window_ms = float(event_window_ms)
        self.load_events = bool(load_events)

        self._check_required_paths()

        self.rgb_paths = sorted(self.image_dir.glob("*.png"))
        self.depth_paths = self._make_depth_paths()
        self.timestamps = self._load_timestamps()
        self.intrinsics = self._load_intrinsics()

        if len(self.rgb_paths) == 0:
            raise FileNotFoundError(f"No RGB png files found in {self.image_dir}")
        if len(self.timestamps) != len(self.rgb_paths):
            raise ValueError(
                f"Timestamp count {len(self.timestamps)} does not match RGB count {len(self.rgb_paths)}."
            )

        self.ms_to_idx = None
        if self.load_events:
            self.ms_to_idx = self._load_ms_to_idx()

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of range. Dataset length is {len(self)}.")

        rgb_path = self.rgb_paths[index]
        depth_path = self.depth_paths[index]
        timestamp = int(self.timestamps[index])

        sample = {
            "rgb": self._read_rgb(rgb_path),
            "depth": self._read_depth(depth_path),
            "events": None,
            "timestamp": timestamp,
            "frame_id": rgb_path.stem,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "sequence_name": self.sequence,
            "intrinsics": self.intrinsics.copy(),
        }

        if self.load_events:
            start_t, end_t = self._event_window(index)
            sample["events"] = self._read_events(start_t, end_t)
            if len(sample["events"]["t"]) == 0:
                warnings.warn(
                    f"No events in sequence={self.sequence}, frame={rgb_path.stem}, "
                    f"window=[{start_t}, {end_t}].",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return sample

    def _check_required_paths(self):
        if not self.sequence_dir.is_dir():
            raise FileNotFoundError(f"Missing sequence directory: {self.sequence_dir}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Missing RGB directory: {self.image_dir}")
        if not self.depth_dir.is_dir():
            raise FileNotFoundError(f"Missing depth directory: {self.depth_dir}")
        if not self.timestamp_path.is_file():
            raise FileNotFoundError(f"Missing timestamp file: {self.timestamp_path}")
        if not self.intrinsics_path.is_file():
            raise FileNotFoundError(f"Missing intrinsics file: {self.intrinsics_path}")
        if self.load_events and not self.event_path.is_file():
            raise FileNotFoundError(f"Missing event file: {self.event_path}")

    def _make_depth_paths(self):
        depth_paths = []
        for rgb_path in self.rgb_paths:
            depth_path = self.depth_dir / rgb_path.name
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing depth file for {rgb_path.name}: {depth_path}")
            depth_paths.append(depth_path)
        return depth_paths

    def _load_timestamps(self):
        try:
            timestamps = np.loadtxt(self.timestamp_path, dtype=np.int64)
        except ValueError as exc:
            raise ValueError(f"Expected one integer timestamp per line: {self.timestamp_path}") from exc

        timestamps = np.atleast_1d(timestamps)
        if timestamps.ndim != 1:
            raise ValueError(f"Expected 1D timestamps in {self.timestamp_path}")
        return timestamps

    def _load_intrinsics(self):
        data = json.loads(self.intrinsics_path.read_text(encoding="utf-8"))
        if self.intrinsics_key not in data:
            raise KeyError(f"Missing intrinsics key {self.intrinsics_key} in {self.intrinsics_path}")
        return np.asarray(data[self.intrinsics_key]["K"], dtype=np.float32)

    def _load_ms_to_idx(self):
        with h5py.File(self.event_path, "r") as events:
            for key in ["ms_to_idx", "x", "y", "t", "p"]:
                if key not in events:
                    raise KeyError(f"Missing key {key} in {self.event_path}")
            return np.asarray(events["ms_to_idx"][:], dtype=np.int64)

    def _read_rgb(self, path):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))

        expected_shape = (self.image_height, self.image_width, 3)
        if rgb.shape != expected_shape:
            raise ValueError(f"RGB shape is {rgb.shape}, expected {expected_shape}: {path}")
        return rgb

    def _read_depth(self, path):
        with Image.open(path) as image:
            depth_raw = np.asarray(image)

        expected_shape = (self.image_height, self.image_width)
        if depth_raw.shape != expected_shape:
            raise ValueError(f"Depth shape is {depth_raw.shape}, expected {expected_shape}: {path}")

        depth = depth_raw.astype(np.float32) / self.depth_scale
        return depth

    def _event_window(self, index):
        end_t = int(self.timestamps[index])

        if self.event_window_mode == "prev_frame" and index > 0:
            start_t = int(self.timestamps[index - 1])
        else:
            start_t = int(round(end_t - self.event_window_ms * 1000.0))

        return start_t, end_t

    def _read_events(self, start_t, end_t):
        if self.ms_to_idx is None:
            raise RuntimeError("load_events=False, event index is not available.")

        if start_t < 0:
            start_t = 0

        start_ms = int(round(start_t / 1000.0))
        end_ms = int(round(end_t / 1000.0))

        if start_ms < 0:
            start_ms = 0
        if end_ms < start_ms:
            end_ms = start_ms

        with h5py.File(self.event_path, "r") as events:
            event_count = int(events["t"].shape[0])

            if start_ms >= len(self.ms_to_idx):
                start_idx = event_count
            else:
                start_idx = int(self.ms_to_idx[start_ms])

            if end_ms >= len(self.ms_to_idx):
                end_idx = event_count
            else:
                end_idx = int(self.ms_to_idx[end_ms])

            return {
                "x": np.asarray(events["x"][start_idx:end_idx]),
                "y": np.asarray(events["y"][start_idx:end_idx]),
                "t": np.asarray(events["t"][start_idx:end_idx]),
                "p": np.asarray(events["p"][start_idx:end_idx]),
            }
