from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


class CoSECDataset:
    """Minimal CoSEC left-view RGB/depth/event reader for data debugging."""

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        data_root: str | Path,
        sequence: str,
        image_folder: str = "img_co_left",
        depth_folder: str = "depth_co",
        event_file: str = "events_co_left.h5",
        timestamp_file: str = "timestamps.txt",
        depth_scale: float = 256.0,
        event_window_mode: str = "fixed",
        event_window_ms: float = 50.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.sequence = sequence
        self.sequence_dir = self.data_root / sequence
        self.image_folder = image_folder
        self.depth_folder = depth_folder
        self.event_file = event_file
        self.timestamp_file = timestamp_file
        self.depth_scale = float(depth_scale)
        self.event_window_mode = event_window_mode
        self.event_window_ms = float(event_window_ms)

        if self.event_window_mode not in {"fixed", "prev_frame"}:
            raise ValueError(
                "event_window_mode must be either 'fixed' or 'prev_frame', "
                f"got {self.event_window_mode!r}."
            )

        self.image_dir = self.sequence_dir / self.image_folder
        self.depth_dir = self.sequence_dir / self.depth_folder
        self.event_path = self.sequence_dir / self.event_file
        self.timestamp_path = self.sequence_dir / self.timestamp_file

        self._validate_paths()
        self.rgb_paths = self._list_images(self.image_dir)
        self.depth_paths = self._list_images(self.depth_dir)
        self.timestamps = self._load_timestamps(self.timestamp_path)
        self._validate_lengths()
        self._validate_event_file()

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of range for {len(self)} RGB frames.")

        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]
        frame_t = float(self.timestamps[idx])
        start_t, end_t = self._event_window_for_index(idx)

        rgb = self._read_rgb(rgb_path)
        depth = self._read_depth(depth_path)
        events = self._read_events(start_t, end_t)

        if len(events["t"]) == 0:
            warnings.warn(
                f"No events found for frame index {idx} in time window [{start_t}, {end_t}].",
                RuntimeWarning,
                stacklevel=2,
            )

        return {
            "rgb": rgb,
            "depth": depth,
            "events": events,
            "timestamp": frame_t,
            "frame_id": rgb_path.stem,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "sequence_name": self.sequence,
        }

    def _validate_paths(self) -> None:
        required_dirs = {
            "sequence directory": self.sequence_dir,
            "RGB image folder": self.image_dir,
            "depth folder": self.depth_dir,
        }
        required_files = {
            "event h5 file": self.event_path,
            "timestamp file": self.timestamp_path,
        }

        for label, path in required_dirs.items():
            if not path.is_dir():
                raise FileNotFoundError(f"Missing {label}: {path}")
        for label, path in required_files.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")

    @classmethod
    def _list_images(cls, directory: Path) -> list[Path]:
        return sorted(path for path in directory.iterdir() if path.suffix.lower() in cls.IMAGE_EXTENSIONS)

    @staticmethod
    def _load_timestamps(path: Path) -> np.ndarray:
        timestamps: list[float] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            token = stripped.replace(",", " ").split()[0]
            try:
                timestamps.append(float(token))
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp at {path}:{line_number}: {stripped!r}") from exc

        if not timestamps:
            raise ValueError(f"No valid timestamps found in {path}.")
        return np.asarray(timestamps, dtype=np.float64)

    def _validate_lengths(self) -> None:
        if not self.rgb_paths:
            raise FileNotFoundError(f"No RGB images found in {self.image_dir}.")
        if not self.depth_paths:
            raise FileNotFoundError(f"No depth images found in {self.depth_dir}.")
        if len(self.depth_paths) < len(self.rgb_paths):
            raise ValueError(
                f"Depth image count ({len(self.depth_paths)}) is smaller than RGB image count "
                f"({len(self.rgb_paths)})."
            )
        if len(self.timestamps) < len(self.rgb_paths):
            raise ValueError(
                f"Timestamp rows ({len(self.timestamps)}) are fewer than RGB frames ({len(self.rgb_paths)})."
            )

    def _validate_event_file(self) -> None:
        with h5py.File(self.event_path, "r") as handle:
            missing = [key for key in ("x", "y", "t", "p") if key not in handle]
            if missing:
                raise KeyError(f"Missing event datasets in {self.event_path}: {missing}")

            lengths = {key: int(handle[key].shape[0]) for key in ("x", "y", "t", "p")}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Event x/y/t/p lengths do not match: {lengths}")

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        try:
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise OSError(f"Failed to read RGB image: {path}") from exc

    def _read_depth(self, path: Path) -> np.ndarray:
        try:
            with Image.open(path) as image:
                depth_raw = np.asarray(image)
        except Exception as exc:
            raise OSError(f"Failed to read depth image: {path}") from exc

        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        if depth_raw.dtype != np.uint16:
            warnings.warn(
                f"Depth image {path} has dtype {depth_raw.dtype}, expected uint16.",
                RuntimeWarning,
                stacklevel=2,
            )
        return depth_raw.astype(np.float32) / self.depth_scale

    def _event_window_for_index(self, idx: int) -> tuple[float, float]:
        end_t = float(self.timestamps[idx])
        if self.event_window_mode == "prev_frame" and idx > 0:
            start_t = float(self.timestamps[idx - 1])
        else:
            start_t = end_t - self.event_window_ms * 1000.0
        return start_t, end_t

    def _read_events(self, start_t: float, end_t: float) -> dict[str, np.ndarray]:
        if start_t > end_t:
            raise ValueError(f"Invalid event window: start_t={start_t} is greater than end_t={end_t}.")

        with h5py.File(self.event_path, "r") as handle:
            t_dataset = handle["t"]
            start_idx = _searchsorted_h5(t_dataset, start_t, side="left")
            end_idx = _searchsorted_h5(t_dataset, end_t, side="right")

            if start_idx > end_idx:
                raise IndexError(f"Invalid event slice [{start_idx}:{end_idx}] for window [{start_t}, {end_t}].")

            return {
                "x": np.asarray(handle["x"][start_idx:end_idx]),
                "y": np.asarray(handle["y"][start_idx:end_idx]),
                "t": np.asarray(handle["t"][start_idx:end_idx]),
                "p": np.asarray(handle["p"][start_idx:end_idx]),
            }


def _searchsorted_h5(dataset: h5py.Dataset, value: float, side: str = "left") -> int:
    """Binary search a sorted 1D h5py dataset without loading it all."""
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}.")

    left = 0
    right = int(dataset.shape[0])
    while left < right:
        mid = (left + right) // 2
        mid_value = float(dataset[mid])
        if mid_value < value or (side == "right" and mid_value == value):
            left = mid + 1
        else:
            right = mid
    return left
