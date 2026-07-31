from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a cleaned HDF5 by dropping or keeping frame ranges.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--drop-ranges", type=str, default="")
    parser.add_argument("--drop-ranges-file", type=Path, default=None)
    parser.add_argument("--keep-ranges", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_range_token(token: str, total: int) -> tuple[int, int]:
    if ":" not in token:
        index = int(token)
        return max(0, index), min(total, index + 1)
    left, right = token.split(":", 1)
    start = int(left) if left else 0
    end = int(right) if right else total
    start = max(0, min(total, start))
    end = max(0, min(total, end))
    if end < start:
        start, end = end, start
    return start, end


def parse_ranges(text: str, total: int) -> list[tuple[int, int]]:
    ranges = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        for raw in line.split(","):
            token = raw.strip()
            if not token:
                continue
            ranges.append(parse_range_token(token, total))
    return ranges


def load_ranges(path: Path, total: int) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return parse_ranges(path.read_text(encoding="utf-8"), total)


def mask_from_ranges(total: int, ranges: list[tuple[int, int]], value: bool) -> np.ndarray:
    mask = np.zeros(total, dtype=bool) if value else np.ones(total, dtype=bool)
    for start, end in ranges:
        mask[start:end] = value
    return mask


def copy_dataset(src: h5py.Dataset, dst_file: h5py.File, name: str, keep_indices: np.ndarray, source_total: int) -> None:
    if src.shape and src.shape[0] == source_total:
        data = src[keep_indices]
        maxshape = (None, *src.shape[1:])
        chunks = src.chunks
        dst = dst_file.create_dataset(
            name,
            data=data,
            maxshape=maxshape,
            chunks=chunks,
            compression=src.compression,
            compression_opts=src.compression_opts,
        )
    else:
        dst = dst_file.create_dataset(name, data=src[()])
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def main() -> None:
    args = parse_args()
    if args.out is None:
        args.out = args.input.with_name(f"{args.input.stem}_clean{args.input.suffix}")
    if args.drop_ranges_file is None and not args.drop_ranges and not args.keep_ranges:
        default_ranges = args.input.with_suffix(".bad_ranges.txt")
        if default_ranges.exists():
            args.drop_ranges_file = default_ranges

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"{args.out} already exists. Pass --overwrite to replace it.")

    with h5py.File(args.input, "r") as src:
        total = len(src["frames"])
        keep_mask = np.ones(total, dtype=bool)

        keep_ranges = parse_ranges(args.keep_ranges, total)
        if keep_ranges:
            keep_mask = mask_from_ranges(total, keep_ranges, True)

        drop_ranges = parse_ranges(args.drop_ranges, total)
        if args.drop_ranges_file is not None:
            drop_ranges.extend(load_ranges(args.drop_ranges_file, total))
        if drop_ranges:
            drop_mask = mask_from_ranges(total, drop_ranges, True)
            keep_mask &= ~drop_mask

        keep_indices = np.flatnonzero(keep_mask)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.out, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["source_file"] = str(args.input)
            dst.attrs["source_samples"] = int(total)
            dst.attrs["samples"] = int(len(keep_indices))
            dst.attrs["dropped_samples"] = int(total - len(keep_indices))
            dst.attrs["drop_ranges"] = ",".join(f"{start}:{end}" for start, end in drop_ranges)
            dst.attrs["keep_ranges"] = ",".join(f"{start}:{end}" for start, end in keep_ranges)

            for name, item in src.items():
                if isinstance(item, h5py.Dataset):
                    copy_dataset(item, dst, name, keep_indices, total)

    print(f"input={args.input} samples={total}")
    print(f"output={args.out} samples={len(keep_indices)} dropped={total - len(keep_indices)}")


if __name__ == "__main__":
    main()
