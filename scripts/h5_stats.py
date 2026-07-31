from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print frame and key-label stats for gameplay HDF5 files.")
    parser.add_argument("paths", type=Path, nargs="+")
    return parser.parse_args()


def expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        text = str(path)
        if any(char in text for char in "*?[]"):
            expanded.extend(sorted(path.parent.glob(path.name)))
        else:
            expanded.append(path)
    return expanded


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.paths)
    if not paths:
        raise FileNotFoundError("No files matched.")

    for path in paths:
        with h5py.File(path, "r") as h5:
            if "frames" not in h5 or "key_labels" not in h5:
                print(f"{path}: skipped, not a real-game gameplay H5")
                continue
            labels = h5["key_labels"][:]
            names = str(h5.attrs.get("key_labels", "")).split(",")
            counts = labels.sum(axis=0)
            total = max(1, len(labels))
            print(f"{path} frames={len(labels)}")
            for name, count in zip(names, counts):
                print(f"  {name}: count={int(count)} rate={count / total:.3f}")


if __name__ == "__main__":
    main()
