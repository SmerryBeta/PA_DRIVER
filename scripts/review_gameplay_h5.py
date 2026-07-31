from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review a gameplay HDF5 file and mark bad frame ranges.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ranges-out", type=Path, default=None)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def compact_keys(label_names: list[str], labels: np.ndarray) -> str:
    active = [name.split("_", 1)[0].upper() for name, value in zip(label_names, labels) if value > 0.5]
    return "+".join(active) or "-"


def in_ranges(index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in ranges)


def save_ranges(path: Path, ranges: list[tuple[int, int]], source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# source={source}\n")
        handle.write("# Ranges are start:end, end exclusive.\n")
        for start, end in ranges:
            handle.write(f"{start}:{end}\n")


def draw_frame(
    frame: np.ndarray,
    index: int,
    total: int,
    label_names: list[str],
    labels: np.ndarray,
    ranges: list[tuple[int, int]],
    pending_start: int | None,
    playing: bool,
    scale: int,
) -> np.ndarray:
    display = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    marked = in_ranges(index, ranges)
    if marked:
        overlay = display.copy()
        overlay[:, :] = (0, 0, 160)
        display = cv2.addWeighted(overlay, 0.28, display, 0.72, 0)

    status = "PLAY" if playing else "PAUSE"
    if marked:
        status += " BAD"
    if pending_start is not None:
        status += f" MARK {pending_start}"

    lines = [
        f"{index}/{total - 1} {status}",
        f"keys={compact_keys(label_names, labels)}",
        "A/D prev/next  W/S +/-100  Space play  B begin  E end  U undo  Q quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 24
    return display


def main() -> None:
    args = parse_args()
    ranges_out = args.ranges_out or args.input.with_suffix(".bad_ranges.txt")
    ranges: list[tuple[int, int]] = []
    pending_start: int | None = None
    playing = False

    with h5py.File(args.input, "r") as h5:
        frames = h5["frames"]
        labels = h5["key_labels"]
        label_names = str(h5.attrs.get("key_labels", "")).split(",")
        total = len(frames)
        index = max(0, min(args.start, total - 1))
        delay = max(1, int(1000 / args.fps))

        cv2.namedWindow("PA_DRIVER H5 reviewer", cv2.WINDOW_NORMAL)
        try:
            while True:
                display = draw_frame(
                    frames[index],
                    index,
                    total,
                    label_names,
                    labels[index],
                    ranges,
                    pending_start,
                    playing,
                    args.scale,
                )
                cv2.imshow("PA_DRIVER H5 reviewer", display)
                key = cv2.waitKey(delay if playing else 0) & 0xFF

                if playing:
                    index = min(total - 1, index + 1)

                if key in (ord("q"), 27):
                    break
                if key in (ord(" "),):
                    playing = not playing
                elif key in (ord("d"), ord("n")):
                    playing = False
                    index = min(total - 1, index + 1)
                elif key in (ord("a"), ord("p")):
                    playing = False
                    index = max(0, index - 1)
                elif key == ord("w"):
                    playing = False
                    index = min(total - 1, index + 100)
                elif key == ord("s"):
                    playing = False
                    index = max(0, index - 100)
                elif key == ord("b"):
                    pending_start = index
                    playing = False
                elif key == ord("e") and pending_start is not None:
                    start, end = sorted((pending_start, index))
                    ranges.append((start, end + 1))
                    ranges.sort()
                    pending_start = None
                    playing = False
                elif key == ord("u") and ranges:
                    ranges.pop()
                    playing = False
                elif key == ord("c"):
                    pending_start = None
                    playing = False
                elif key == ord("r"):
                    save_ranges(ranges_out, ranges, args.input)
                    print(f"saved ranges -> {ranges_out}")
        finally:
            cv2.destroyAllWindows()

    save_ranges(ranges_out, ranges, args.input)
    print(f"saved ranges -> {ranges_out}")
    print(f"ranges={ranges}")


if __name__ == "__main__":
    main()
