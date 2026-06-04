from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


class CoSECDataset:
    """Minimal CoSEC left-view RGB/depth/event reader for data debugging."""

    def __init__(
        self,
        data_root: str | Path,
        sequence: str,
        view: str = "left",
        image_folder: str = "img_co_left",
        depth_folder: str = "depth_co",
        event_file: str = "events_co_left.h5",
        timestamp_file: str = "timestamps.txt",
        intrinsics_file: str = "intrinsics.json",
        intrinsics_key: str = "Co_Rect_L",
        image_height: int = 624,
        image_width: int = 1200,
        depth_scale: float = 256.0,
        event_window_mode: str = "fixed",
        event_window_ms: float = 50.0,
        **kwargs: Any,
    ) -> None:
        self.data_root = Path(data_root)
        self.sequence = sequence
        self.sequence_dir = self.data_root / sequence
        self.view = view
        self.image_folder = image_folder
        self.depth_folder = depth_folder
        self.event_file = event_file
        self.timestamp_file = timestamp_file
        self.intrinsics_file = intrinsics_file
        self.intrinsics_key = intrinsics_key
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.depth_scale = float(depth_scale)
        self.event_window_mode = event_window_mode
        self.event_window_ms = float(event_window_ms)
        self.extra_options = kwargs

        if self.view != "left":
            raise ValueError(f"Only left view is supported for now, got view={self.view!r}.")
        if self.event_window_mode not in {"fixed", "prev_frame"}:
            raise ValueError(
                "event_window_mode must be either 'fixed' or 'prev_frame', "
                f"got {self.event_window_mode!r}."
            )

        self.image_dir = self.sequence_dir / self.image_folder
        self.depth_dir = self.sequence_dir / self.depth_folder
        self.event_path = self.sequence_dir / self.event_file
        self.timestamp_path = self.sequence_dir / self.timestamp_file
        self.intrinsics_path = self.sequence_dir / self.intrinsics_file

        self._validate_base_paths()
        self.rgb_paths = self._list_pngs(self.image_dir)
        self.depth_paths = self._match_depth_paths(self.rgb_paths)
        self.timestamps = self._load_timestamps(self.timestamp_path)
        self.intrinsics = self._load_intrinsics(self.intrinsics_path)
        self._validate_lengths()
        self.ms_to_idx = self._load_event_index()

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of range for {len(self)} RGB frames.")

        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]
        frame_t = int(self.timestamps[idx])
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
            "intrinsics": self.intrinsics.copy(),
        }

    def _validate_base_paths(self) -> None:
        required_dirs = {
            "sequence directory": self.sequence_dir,
            "RGB image folder": self.image_dir,
            "depth folder": self.depth_dir,
        }
        required_files = {
            "event h5 file": self.event_path,
            "timestamp file": self.timestamp_path,
            "intrinsics file": self.intrinsics_path,
        }

        for label, path in required_dirs.items():
            if not path.is_dir():
                raise FileNotFoundError(f"Missing {label}: {path}")
        for label, path in required_files.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")

    @staticmethod
    def _list_pngs(directory: Path) -> list[Path]:
        return sorted(directory.glob("*.png"))

    def _match_depth_paths(self, rgb_paths: list[Path]) -> list[Path]:
        depth_paths = [self.depth_dir / rgb_path.name for rgb_path in rgb_paths]
        missing = [path for path in depth_paths if not path.is_file()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            suffix = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise FileNotFoundError(f"Missing depth files matching RGB names: {preview}{suffix}")
        return depth_paths

    @staticmethod
    def _load_timestamps(path: Path) -> np.ndarray:
        try:
            timestamps = np.loadtxt(path, dtype=np.int64)
        except ValueError as exc:
            raise ValueError(f"Expected one integer timestamp per line in {path}.") from exc

        timestamps = np.atleast_1d(timestamps)
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise ValueError(f"Expected a non-empty 1D timestamp array in {path}, got {timestamps.shape}.")
        if np.any(timestamps[1:] <= timestamps[:-1]):
            raise ValueError(f"Timestamps are not strictly increasing in {path}.")
        return timestamps

    def _load_intrinsics(self, path: Path) -> np.ndarray:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            camera = data[self.intrinsics_key]
            intrinsics = np.asarray(camera["K"], dtype=np.float32)
            resolution = tuple(camera["resolution"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Failed to read intrinsics key {self.intrinsics_key!r} from {path}."
            ) from exc

        if intrinsics.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 intrinsics matrix, got shape {intrinsics.shape}.")
        expected_resolution = (self.image_height, self.image_width)
        if resolution != expected_resolution:
            raise ValueError(
                f"Intrinsics resolution {resolution} does not match expected {expected_resolution}."
            )
        return intrinsics

    def _validate_lengths(self) -> None:
        if not self.rgb_paths:
            raise FileNotFoundError(f"No RGB images found in {self.image_dir}.")
        if not self.depth_paths:
            raise FileNotFoundError(f"No depth images found in {self.depth_dir}.")
        if len(self.depth_paths) != len(self.rgb_paths):
            raise ValueError(
                f"Depth image count ({len(self.depth_paths)}) does not match RGB image count "
                f"({len(self.rgb_paths)})."
            )
        if len(self.timestamps) != len(self.rgb_paths):
            raise ValueError(
                f"Timestamp rows ({len(self.timestamps)}) do not match RGB frames ({len(self.rgb_paths)})."
            )

    def _load_event_index(self) -> np.ndarray:
        with h5py.File(self.event_path, "r") as handle:
            missing = [key for key in ("ms_to_idx", "x", "y", "t", "p") if key not in handle]
            if missing:
                raise KeyError(f"Missing event datasets in {self.event_path}: {missing}")

            lengths = {key: int(handle[key].shape[0]) for key in ("x", "y", "t", "p")}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Event x/y/t/p lengths do not match: {lengths}")

            ms_to_idx = np.asarray(handle["ms_to_idx"][:], dtype=np.int64)
            if ms_to_idx.ndim != 1 or ms_to_idx.size == 0:
                raise ValueError(f"ms_to_idx must be a non-empty 1D array in {self.event_path}.")
            if np.any(ms_to_idx[1:] < ms_to_idx[:-1]):
                raise ValueError(f"ms_to_idx is not non-decreasing in {self.event_path}.")
            if ms_to_idx[0] < 0 or ms_to_idx[-1] > lengths["t"]:
                raise ValueError(
                    f"ms_to_idx contains event indices outside [0, {lengths['t']}] in {self.event_path}."
                )
            return ms_to_idx

    def _read_rgb(self, path: Path) -> np.ndarray:
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise OSError(f"Failed to read RGB image: {path}") from exc
        expected_shape = (self.image_height, self.image_width, 3)
        if rgb.shape != expected_shape:
            raise ValueError(f"RGB image {path} has shape {rgb.shape}, expected {expected_shape}.")
        return rgb

    def _read_depth(self, path: Path) -> np.ndarray:
        try:
            with Image.open(path) as image:
                depth_raw = np.asarray(image)
        except Exception as exc:
            raise OSError(f"Failed to read depth image: {path}") from exc

        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        expected_shape = (self.image_height, self.image_width)
        if depth_raw.shape != expected_shape:
            raise ValueError(f"Depth image {path} has shape {depth_raw.shape}, expected {expected_shape}.")
        if depth_raw.dtype != np.uint16:
            warnings.warn(
                f"Depth image {path} has dtype {depth_raw.dtype}, expected uint16.",
                RuntimeWarning,
                stacklevel=2,
            )
        return depth_raw.astype(np.float32) / self.depth_scale

    def _event_window_for_index(self, idx: int) -> tuple[int, int]:
        end_t = int(self.timestamps[idx])
        if self.event_window_mode == "prev_frame" and idx > 0:
            start_t = int(self.timestamps[idx - 1])
        else:
            start_t = int(round(end_t - self.event_window_ms * 1000.0))
        return start_t, end_t

    def _read_events(self, start_t: int, end_t: int) -> dict[str, np.ndarray]:
        if start_t > end_t:
            raise ValueError(f"Invalid event window: start_t={start_t} is greater than end_t={end_t}.")

        with h5py.File(self.event_path, "r") as handle:
            t_dataset = handle["t"]
            event_count = int(t_dataset.shape[0])
            coarse_start_idx, coarse_end_idx = self._coarse_event_indices(start_t, end_t, event_count)
            local_t = np.asarray(t_dataset[coarse_start_idx:coarse_end_idx])

            local_start_pos = int(np.searchsorted(local_t, start_t, side="left"))
            local_end_pos = int(np.searchsorted(local_t, end_t, side="right"))

            start_idx = coarse_start_idx + local_start_pos
            end_idx = coarse_start_idx + local_end_pos

            return {
                "x": np.asarray(handle["x"][start_idx:end_idx]),
                "y": np.asarray(handle["y"][start_idx:end_idx]),
                "t": np.asarray(handle["t"][start_idx:end_idx]),
                "p": np.asarray(handle["p"][start_idx:end_idx]),
            }

    def _coarse_event_indices(self, start_t: int, end_t: int, event_count: int) -> tuple[int, int]:
        # Expand by one millisecond on each side, then crop precisely using event t.
        start_ms = max(0, start_t // 1000 - 1)
        end_ms = end_t // 1000 + 2

        if start_ms >= len(self.ms_to_idx):
            return event_count, event_count

        coarse_start_idx = int(self.ms_to_idx[start_ms])
        coarse_end_idx = (
            int(self.ms_to_idx[end_ms])
            if end_ms < len(self.ms_to_idx)
            else event_count
        )
        return coarse_start_idx, coarse_end_idx
