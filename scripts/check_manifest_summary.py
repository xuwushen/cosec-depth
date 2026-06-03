from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a CoSEC train manifest.")
    parser.add_argument("--manifest", default="metadata/train_manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("sequences", [])
    ok_records = [record for record in records if record.get("status") == "ok"]
    abnormal_records = [record for record in records if record.get("status") != "ok"]

    consistent_records = []
    inconsistent_records = []
    total_train_frames = 0

    for record in ok_records:
        rgb_count = record.get("rgb_count")
        depth_count = record.get("depth_count")
        timestamp_count = record.get("timestamp_count")
        counts_match = rgb_count == depth_count == timestamp_count

        if counts_match:
            consistent_records.append(record)
            total_train_frames += int(rgb_count or 0)
        else:
            inconsistent_records.append(record)
            counts = [count for count in (rgb_count, depth_count, timestamp_count) if count is not None]
            if counts:
                total_train_frames += int(min(counts))

    print(f"Manifest: {manifest_path}")
    print(f"Data root: {manifest.get('data_root')}")
    print(f"Split: {manifest.get('split')}")
    print(f"Total sequences: {len(records)}")
    print(f"OK sequences: {len(ok_records)}")
    print(f"Abnormal sequences: {len(abnormal_records)}")
    print(f"Count-consistent OK sequences: {len(consistent_records)}")
    print(f"Count-inconsistent OK sequences: {len(inconsistent_records)}")
    print(f"Estimated total training frames: {total_train_frames}")

    if abnormal_records:
        print("\nAbnormal sequences:")
        for record in abnormal_records:
            print(
                f"  {record.get('sequence')}: status={record.get('status')}, "
                f"missing_items={record.get('missing_items', [])}"
            )

    if inconsistent_records:
        print("\nSequences with inconsistent rgb/depth/timestamp counts:")
        for record in inconsistent_records:
            print(
                f"  {record.get('sequence')}: "
                f"rgb_count={record.get('rgb_count')}, "
                f"depth_count={record.get('depth_count')}, "
                f"timestamp_count={record.get('timestamp_count')}"
            )


if __name__ == "__main__":
    main()
