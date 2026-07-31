from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_game_bc import resolve_data_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine multiple real-game gameplay HDF5 files into one file.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("data/combined.h5"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_dataset(h5: h5py.File, name: str) -> h5py.Dataset:
    if name not in h5:
        raise KeyError(f"{h5.filename} is missing dataset {name!r}")
    return h5[name]


def read_key_labels(h5: h5py.File) -> str:
    return str(h5.attrs.get("key_labels", "w_forward,a_left,d_right,s_reverse,space_drift,j_item"))


def validate_inputs(paths: list[Path]) -> tuple[tuple[int, int, int], int, str]:
    if not paths:
        raise FileNotFoundError("No input files matched.")

    frame_shape: tuple[int, int, int] | None = None
    label_dim: int | None = None
    key_labels: str | None = None

    for path in paths:
        with h5py.File(path, "r") as h5:
            frames = require_dataset(h5, "frames")
            labels = require_dataset(h5, "key_labels")
            if frames.ndim != 4:
                raise ValueError(f"{path}: frames must be NHWC, got shape {frames.shape}")
            if labels.ndim != 2:
                raise ValueError(f"{path}: key_labels must be 2D, got shape {labels.shape}")
            if len(frames) != len(labels):
                raise ValueError(f"{path}: frames/key_labels length mismatch {len(frames)} != {len(labels)}")

            current_frame_shape = tuple(frames.shape[1:])
            current_label_dim = labels.shape[1]
            current_key_labels = read_key_labels(h5)

            if frame_shape is None:
                frame_shape = current_frame_shape
                label_dim = current_label_dim
                key_labels = current_key_labels
            elif current_frame_shape != frame_shape:
                raise ValueError(f"{path}: frame shape {current_frame_shape} does not match {frame_shape}")
            elif current_label_dim != label_dim:
                raise ValueError(f"{path}: label dim {current_label_dim} does not match {label_dim}")
            elif current_key_labels != key_labels:
                raise ValueError(f"{path}: key_labels attr {current_key_labels!r} does not match {key_labels!r}")

    assert frame_shape is not None
    assert label_dim is not None
    assert key_labels is not None
    return frame_shape, label_dim, key_labels


def main() -> None:
    args = parse_args()
    paths = resolve_data_paths(args.inputs)
    paths = [path for path in paths if path.resolve() != args.out.resolve()]
    frame_shape, label_dim, key_labels = validate_inputs(paths)

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"{args.out} already exists. Pass --overwrite to replace it.")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    for path in paths:
        with h5py.File(path, "r") as h5:
            total += len(h5["frames"])

    with h5py.File(args.out, "w") as out:
        frames_out = out.create_dataset(
            "frames",
            shape=(total, *frame_shape),
            maxshape=(None, *frame_shape),
            chunks=(64, *frame_shape),
            dtype=np.uint8,
            compression="lzf",
        )
        labels_out = out.create_dataset(
            "key_labels",
            shape=(total, label_dim),
            maxshape=(None, label_dim),
            chunks=(512, label_dim),
            dtype=np.float32,
            compression="lzf",
        )
        timestamps_out = out.create_dataset(
            "timestamps",
            shape=(total,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=np.float64,
            compression="lzf",
        )

        offset = 0
        source_names = []
        source_lengths = []
        next_time = 0.0
        for path in paths:
            with h5py.File(path, "r") as h5:
                n = len(h5["frames"])
                end = offset + n
                frames_out[offset:end] = h5["frames"][:]
                labels_out[offset:end] = h5["key_labels"][:]
                if "timestamps" in h5 and len(h5["timestamps"]) == n:
                    ts = np.asarray(h5["timestamps"][:], dtype=np.float64)
                    if n > 1:
                        step = float(np.median(np.diff(ts)))
                        if step <= 0:
                            step = 1.0 / float(h5.attrs.get("fps", 20.0))
                    else:
                        step = 1.0 / float(h5.attrs.get("fps", 20.0))
                    ts = ts - ts[0] + next_time
                    next_time = float(ts[-1] + step) if n else next_time
                    timestamps_out[offset:end] = ts
                else:
                    fps = float(h5.attrs.get("fps", 20.0))
                    timestamps_out[offset:end] = next_time + np.arange(n, dtype=np.float64) / fps
                    next_time += n / fps

                source_names.append(str(path))
                source_lengths.append(str(n))
                print(f"added {path} frames={n}")
                offset = end

        out.attrs["key_labels"] = key_labels
        out.attrs["samples"] = int(total)
        out.attrs["source_files"] = "\n".join(source_names)
        out.attrs["source_lengths"] = ",".join(source_lengths)
        out.attrs["combined_file_count"] = int(len(paths))

        with h5py.File(paths[0], "r") as first:
            for key, value in first.attrs.items():
                if key.startswith("crop_") or key == "fps":
                    out.attrs[key] = value

    print(f"combined_files={len(paths)} samples={total} -> {args.out}")


if __name__ == "__main__":
    main()
