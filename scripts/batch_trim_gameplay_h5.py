from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.trim_gameplay_h5 import copy_dataset, load_ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-create cleaned HDF5 files from .bad_ranges.txt markers.")
    parser.add_argument("--ranges-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--pattern", default="*.bad_ranges.txt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-empty", action="store_true", help="Also copy files whose ranges file has no drops.")
    return parser.parse_args()


def source_from_ranges_file(ranges_file: Path) -> Path:
    stem = ranges_file.name[: -len(".bad_ranges.txt")]
    return ranges_file.with_name(f"{stem}.h5")


def trim_one(source: Path, ranges_file: Path, out: Path, overwrite: bool) -> tuple[int, int]:
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists. Pass --overwrite to replace it.")

    with h5py.File(source, "r") as src:
        total = len(src["frames"])
        drop_ranges = load_ranges(ranges_file, total)
        drop_mask = np.zeros(total, dtype=bool)
        for start, end in drop_ranges:
            drop_mask[start:end] = True
        keep_indices = np.flatnonzero(~drop_mask)

        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["source_file"] = str(source)
            dst.attrs["source_samples"] = int(total)
            dst.attrs["samples"] = int(len(keep_indices))
            dst.attrs["dropped_samples"] = int(total - len(keep_indices))
            dst.attrs["drop_ranges_file"] = str(ranges_file)
            dst.attrs["drop_ranges"] = ",".join(f"{start}:{end}" for start, end in drop_ranges)

            for name, item in src.items():
                if isinstance(item, h5py.Dataset):
                    copy_dataset(item, dst, name, keep_indices, total)

    return total, len(keep_indices)


def main() -> None:
    args = parse_args()
    ranges_files = sorted(args.ranges_dir.glob(args.pattern))
    if not ranges_files:
        raise FileNotFoundError(f"No ranges files matched {args.ranges_dir / args.pattern}")

    made = 0
    skipped = 0
    for ranges_file in ranges_files:
        source = source_from_ranges_file(ranges_file)
        if not source.exists():
            print(f"SKIP missing source: {source}")
            skipped += 1
            continue

        with h5py.File(source, "r") as h5:
            total = len(h5["frames"])
        drop_ranges = load_ranges(ranges_file, total)
        dropped = sum(max(0, end - start) for start, end in drop_ranges)
        if dropped == 0 and not args.include_empty:
            print(f"SKIP empty ranges: {ranges_file.name}")
            skipped += 1
            continue

        out = args.out_dir / source.name
        total, kept = trim_one(source, ranges_file, out, args.overwrite)
        print(f"OK {source.name}: {total} -> {kept}, dropped={total - kept}, out={out}")
        made += 1

    print(f"done made={made} skipped={skipped}")


if __name__ == "__main__":
    main()
