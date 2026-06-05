from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset

from .cosec_dataset import CoSECDataset


class CoSECMultiSequenceDataset(ConcatDataset):
    """Combine all valid CoSEC sequences listed in a manifest."""

    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        **sequence_dataset_kwargs: Any,
    ) -> None:
        self.data_root = Path(data_root)
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("sequences")
        if not isinstance(records, list):
            raise ValueError(f"Manifest has no valid 'sequences' list: {self.manifest_path}")

        self.sequence_names = [
            record["sequence"]
            for record in records
            if record.get("status") == "ok"
        ]
        if not self.sequence_names:
            raise ValueError(f"Manifest contains no sequences with status='ok': {self.manifest_path}")

        datasets = [
            CoSECDataset(
                data_root=self.data_root,
                sequence=sequence_name,
                **sequence_dataset_kwargs,
            )
            for sequence_name in self.sequence_names
        ]
        super().__init__(datasets)

    @property
    def sequence_count(self) -> int:
        return len(self.sequence_names)
