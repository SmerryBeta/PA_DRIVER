from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import h5py
import mss
import numpy as np
from mss.exception import ScreenShotError
from PIL import ImageGrab
from pynput import keyboard


KEY_LABELS = ("w_forward", "a_left", "d_right", "s_reverse", "space_drift", "j_item")
DRIVE_KEYS = {"w", "a", "d", "s", "space", "j"}


class KeyState:
    def __init__(self) -> None:
        self.pressed: set[str] = set()
        self.recording = False
        self.stop = False
        self._last_toggle = 0.0

    def labels(self) -> np.ndarray:
        return np.array(
            [
                "w" in self.pressed,
                "a" in self.pressed,
                "d" in self.pressed,
                "s" in self.pressed,
                "space" in self.pressed,
                "j" in self.pressed,
            ],
            dtype=np.float32,
        )

    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        now = time.perf_counter()
        if key == keyboard.Key.f8 and now - self._last_toggle > 0.35:
            self.recording = not self.recording
            self._last_toggle = now
            print("Recording ON" if self.recording else "Recording PAUSED")
            return
        if key == keyboard.Key.f9:
            self.stop = True
            return

        normalized = normalize_key(key)
        if normalized in DRIVE_KEYS:
            self.pressed.add(normalized)

    def on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        normalized = normalize_key(key)
        if normalized in DRIVE_KEYS:
            self.pressed.discard(normalized)


def normalize_key(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    if key == keyboard.Key.space:
        return "space"
    if isinstance(key, keyboard.KeyCode) and key.char:
        return key.char.lower()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect screenshot + keyboard data from a real kart game.")
    parser.add_argument("--out", type=Path, default=Path("data/gameplay_party_kart_01.h5"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--top", type=int, default=170)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--output-width", type=int, default=224)
    parser.add_argument("--output-height", type=int, default=96)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-recording", action="store_true")
    parser.add_argument("--backend", choices=("auto", "mss", "pil"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class H5Writer:
    def __init__(self, path: Path, frame_shape: tuple[int, int, int], crop: dict[str, int], fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.count = 0
        self.frames = self.file.create_dataset(
            "frames",
            shape=(0, *frame_shape),
            maxshape=(None, *frame_shape),
            chunks=(64, *frame_shape),
            dtype=np.uint8,
            compression="lzf",
        )
        self.key_labels = self.file.create_dataset(
            "key_labels",
            shape=(0, len(KEY_LABELS)),
            maxshape=(None, len(KEY_LABELS)),
            chunks=(512, len(KEY_LABELS)),
            dtype=np.float32,
            compression="lzf",
        )
        self.timestamps = self.file.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=np.float64,
            compression="lzf",
        )
        self.file.attrs["key_labels"] = ",".join(KEY_LABELS)
        self.file.attrs["fps"] = float(fps)
        for key, value in crop.items():
            self.file.attrs[f"crop_{key}"] = int(value)

    def append(self, frames: list[np.ndarray], labels: list[np.ndarray], timestamps: list[float]) -> None:
        if not frames:
            return
        n = len(frames)
        start = self.count
        end = start + n
        self.frames.resize((end, *self.frames.shape[1:]))
        self.key_labels.resize((end, self.key_labels.shape[1]))
        self.timestamps.resize((end,))
        self.frames[start:end] = np.asarray(frames, dtype=np.uint8)
        self.key_labels[start:end] = np.asarray(labels, dtype=np.float32)
        self.timestamps[start:end] = np.asarray(timestamps, dtype=np.float64)
        self.count = end

    def close(self) -> None:
        self.file.attrs["samples"] = int(self.count)
        self.file.flush()
        self.file.close()


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(1, 10_000):
        candidate = parent / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find available output path for {path}")


def draw_preview(frame: np.ndarray, labels: np.ndarray, recording: bool, count: int) -> None:
    preview = cv2.resize(frame, (672, 288), interpolation=cv2.INTER_NEAREST)
    active = [name.split("_", 1)[0].upper() for name, value in zip(KEY_LABELS, labels) if value > 0.5]
    status = "REC" if recording else "PAUSED"
    color = (0, 0, 255) if recording else (80, 80, 80)
    cv2.putText(preview, f"{status} samples={count}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(preview, "keys=" + "+".join(active), (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("PA_DRIVER gameplay collector", preview)
    cv2.waitKey(1)


def capture_with_pil(crop: dict[str, int]) -> np.ndarray:
    bbox = (
        crop["left"],
        crop["top"],
        crop["left"] + crop["width"],
        crop["top"] + crop["height"],
    )
    image = ImageGrab.grab(bbox=bbox)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def capture_with_mss(sct: mss.MSS, crop: dict[str, int]) -> np.ndarray:
    shot = np.asarray(sct.grab(crop), dtype=np.uint8)
    return cv2.cvtColor(shot[:, :, :3], cv2.COLOR_BGR2RGB)


def main() -> None:
    args = parse_args()
    if not args.overwrite:
        args.out = next_available_path(args.out)
    frame_shape = (args.output_height, args.output_width, 3)
    crop = {"left": args.left, "top": args.top, "width": args.width, "height": args.height}
    monitor = dict(crop)
    interval = 1.0 / args.fps

    key_state = KeyState()
    key_state.recording = args.start_recording
    listener = keyboard.Listener(on_press=key_state.on_press, on_release=key_state.on_release)
    listener.start()

    writer = H5Writer(args.out, frame_shape=frame_shape, crop=crop, fps=args.fps)
    frame_buffer: list[np.ndarray] = []
    label_buffer: list[np.ndarray] = []
    time_buffer: list[float] = []

    print("Gameplay collector ready.")
    print("F8 start/pause recording, F9 save and quit.")
    print(f"Crop: left={args.left}, top={args.top}, width={args.width}, height={args.height}")
    print(f"Output: {args.output_width}x{args.output_height} RGB at {args.fps:.1f} FPS -> {args.out}")

    next_tick = time.perf_counter()
    start_time = time.time()
    try:
        with mss.MSS() as sct:
            while not key_state.stop:
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(min(0.002, next_tick - now))
                    continue
                next_tick += interval

                try:
                    if args.backend == "pil":
                        frame = capture_with_pil(crop)
                    else:
                        frame = capture_with_mss(sct, monitor)
                except ScreenShotError:
                    if args.backend == "mss":
                        raise
                    frame = capture_with_pil(crop)

                frame = cv2.resize(frame, (args.output_width, args.output_height), interpolation=cv2.INTER_AREA)
                labels = key_state.labels()

                if key_state.recording:
                    frame_buffer.append(frame)
                    label_buffer.append(labels)
                    time_buffer.append(time.time() - start_time)

                    if len(frame_buffer) >= args.buffer_size:
                        writer.append(frame_buffer, label_buffer, time_buffer)
                        frame_buffer.clear()
                        label_buffer.clear()
                        time_buffer.clear()
                        print(f"samples={writer.count}", end="\r")

                    if args.max_samples is not None and writer.count + len(frame_buffer) >= args.max_samples:
                        key_state.stop = True

                if args.preview:
                    draw_preview(frame, labels, key_state.recording, writer.count + len(frame_buffer))
    finally:
        writer.append(frame_buffer, label_buffer, time_buffer)
        writer.close()
        key_state.stop = True
        listener.stop()
        if args.preview:
            cv2.destroyAllWindows()

    print()
    print(f"Saved {writer.count} samples to {args.out}")


if __name__ == "__main__":
    main()
