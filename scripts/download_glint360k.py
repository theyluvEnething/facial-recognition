"""Download all Glint360K WebDataset shards from HuggingFace to $SCRATCH.

Usage:
    pixi run python scripts/download_glint360k.py
    pixi run python scripts/download_glint360k.py --start 0 --end 100  # subset

The dataset is 1385 shards (~130 GB). Downloads go to
$SCRATCH/datasets/glint360k-wds/ with the format:
    glint360k-0000.tar.gz
    glint360k-0001.tar.gz
    ...
    glint360k-1384.tar.gz

Each shard is a tar.gz containing jpg images + cls labels, ready for
WebDataset streaming during training.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ID = "gaunernst/glint360k-wds-gz"
NUM_SHARDS = 1385


def main():
    parser = argparse.ArgumentParser(description="Download Glint360K WebDataset shards")
    parser.add_argument("--start", type=int, default=0, help="First shard (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="Last shard (default: 1384)")
    parser.add_argument("--target", default=None, help="Target dir (default: $SCRATCH/datasets/glint360k-wds)")
    args = parser.parse_args()

    end = args.end if args.end is not None else NUM_SHARDS - 1

    scratch = os.environ.get("CINECA_SCRATCH", os.environ.get("SCRATCH", "."))
    target = Path(args.target or f"{scratch}/datasets/glint360k-wds")
    target.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    total = (end - args.start + 1)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"Repo:     {REPO_ID}")
    print(f"Target:   {target}")
    print(f"Shards:   {args.start} → {end} ({total:,} files)")
    print(f"Est size: ~{total * 0.094:.0f} GB")
    print("=" * 60)

    t0 = time.time()
    for i in range(args.start, end + 1):
        filename = f"glint360k-{i:04d}.tar.gz"
        local_path = target / filename

        if local_path.exists():
            skipped += 1
            if skipped % 100 == 0:
                print(f"  [{i:04d}] skipped ({skipped} already exist)")
            continue

        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(target),
                local_dir_use_symlinks=False,
            )
            downloaded += 1
        except Exception as e:
            print(f"  [{i:04d}] FAILED: {e}")
            failed += 1
            if failed > 5:
                print("Too many failures, stopping.")
                break
            continue

        if downloaded % 10 == 0:
            elapsed = time.time() - t0
            rate = downloaded / max(elapsed, 1)
            print(f"  [{i:04d}] {downloaded}/{total} downloaded  "
                  f"({rate:.1f} files/s)  {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped:    {skipped}")
    print(f"  Failed:     {failed}")
    print(f"  Target:     {target}")
    print("=" * 60)


if __name__ == "__main__":
    main()
