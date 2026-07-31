from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mss
import numpy as np
import torch
from mss.exception import ScreenShotError
from PIL import ImageGrab
from pynput import keyboard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa_driver.game_model import GameDrivingCNN
from scripts.collect_gameplay import H5Writer, KEY_LABELS, normalize_key
from scripts.run_game_policy import KeyActuator, choose_labels


DRIVE_KEYS = {"w", "a", "d", "s", "space"}
LABEL_TO_KEY = {
    "w_forward": "w",
    "a_left": "a",
    "d_right": "d",
    "s_reverse": "s",
    "space_drift": "space",
}
CTRL_CHAR_TO_KEY = {
    "\x01": "a",
    "\x04": "d",
    "\x13": "s",
    "\x17": "w",
}
VK_TO_KEY = {
    0x41: "a",
    0x44: "d",
    0x53: "s",
    0x57: "w",
    0x20: "space",
}


def normalize_drive_key(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    normalized = normalize_key(key)
    if normalized in DRIVE_KEYS:
        return normalized

    char = getattr(key, "char", None)
    if char in CTRL_CHAR_TO_KEY:
        return CTRL_CHAR_TO_KEY[char]

    vk = getattr(key, "vk", None)
    if vk in VK_TO_KEY:
        return VK_TO_KEY[vk]

    return None


class CorrectionKeyState:
    def __init__(self) -> None:
        self.pressed: set[str] = set()
        self.model_keys: set[str] = set()
        self.human_control = False
        self.recording = False
        self.stop = False
        self._last_toggle = 0.0

    def set_model_labels(self, labels: set[str]) -> None:
        self.model_keys = {LABEL_TO_KEY[label] for label in labels if label in LABEL_TO_KEY}

    def labels(self) -> np.ndarray:
        return np.array(
            [
                "w" in self.pressed,
                "a" in self.pressed,
                "d" in self.pressed,
                "s" in self.pressed,
                "space" in self.pressed,
                False,
            ],
            dtype=np.float32,
        )

    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        now = time.perf_counter()
        if key == keyboard.Key.ctrl_l:
            self.human_control = True
            self.pressed.clear()
            return
        if key == keyboard.Key.f8 and now - self._last_toggle > 0.35:
            self.recording = not self.recording
            self._last_toggle = now
            print("CONTROL ON" if self.recording else "CONTROL PAUSED")
            return
        if key == keyboard.Key.f9:
            self.stop = True
            return

        normalized = normalize_drive_key(key)
        if self.human_control and normalized in DRIVE_KEYS:
            self.pressed.add(normalized)

    def on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key == keyboard.Key.ctrl_l:
            self.human_control = False
            self.pressed.clear()
            return

        normalized = normalize_drive_key(key)
        if normalized in DRIVE_KEYS:
            self.pressed.discard(normalized)


def labels_to_names(labels: np.ndarray) -> set[str]:
    return {name for name, value in zip(KEY_LABELS, labels) if value > 0.5}


def compact_keys(labels: set[str] | np.ndarray) -> str:
    if isinstance(labels, np.ndarray):
        names = [name for name, value in zip(KEY_LABELS, labels) if value > 0.5]
    else:
        names = sorted(labels)
    return "+".join(name.split("_", 1)[0].upper() for name in names) or "-"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect real-game correction data while the model drives.")
    parser.add_argument("--model", type=Path, default=Path("runs/party_kart_bc.pt"))
    parser.add_argument("--out", type=Path, default=Path("data/party_kart_corrections_01.h5"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--top", type=int, default=170)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--backend", choices=("auto", "mss", "pil"), default="auto")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-topmost", action="store_true")
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--w-threshold", type=float, default=0.50)
    parser.add_argument(
        "--w-release-threshold",
        type=float,
        default=0.40,
        help="Release W only after its probability drops below this value.",
    )
    parser.add_argument("--turn-threshold", type=float, default=0.50)
    parser.add_argument("--space-threshold", type=float, default=0.55)
    parser.add_argument("--s-threshold", type=float, default=0.80)
    parser.add_argument("--disable-s", action="store_true", default=True)
    parser.add_argument("--allow-s", action="store_false", dest="disable_s")
    parser.add_argument("--no-space", action="store_true")
    return parser.parse_args()


def next_numbered_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    prefix, separator, tail = stem.rpartition("_")
    if separator and tail.isdigit():
        width = len(tail)
        start = int(tail) + 1
        for index in range(start, 10_000):
            candidate = path.with_name(f"{prefix}_{index:0{width}d}{suffix}")
            if not candidate.exists():
                return candidate

    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
        if not candidate.exists():
            return candidate

    raise FileExistsError(f"Could not find available output path for {path}")


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


def capture_frame(sct: mss.MSS, crop: dict[str, int], backend: str) -> np.ndarray:
    if backend == "pil":
        return capture_with_pil(crop)
    try:
        return capture_with_mss(sct, crop)
    except ScreenShotError:
        if backend == "mss":
            raise
        return capture_with_pil(crop)


def draw_preview(
    frame: np.ndarray,
    control_active: bool,
    human_override: bool,
    human_labels: np.ndarray,
    model_labels: set[str],
    applied_labels: set[str],
    held_labels: set[str],
    probs: np.ndarray,
    label_names: list[str],
    count: int,
    topmost: bool,
) -> None:
    preview = cv2.resize(frame, (672, 288), interpolation=cv2.INTER_NEAREST)
    saving = control_active and human_override
    status = ("REC " if saving else "    ") + ("HUMAN" if human_override else ("MODEL" if control_active else "PAUSED"))
    color = (0, 0, 255) if saving else ((0, 180, 255) if control_active else (80, 80, 80))
    cv2.putText(preview, f"{status} samples={count}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(preview, f"model={compact_keys(model_labels)}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(preview, f"human={compact_keys(human_labels)}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(preview, f"apply={compact_keys(applied_labels)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(preview, f"held={compact_keys(held_labels)}", (10, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    y = 166
    for name, prob in zip(label_names, probs):
        cv2.putText(preview, f"{name}: {prob:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        y += 20

    cv2.imshow("PA_DRIVER correction collector", preview)
    if topmost:
        cv2.setWindowProperty("PA_DRIVER correction collector", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)


def main() -> None:
    args = parse_args()
    args.out = next_numbered_path(args.out)
    checkpoint = torch.load(args.model, map_location="cpu")
    label_names = list(checkpoint["label_names"])
    input_height, input_width = checkpoint.get("input_size", [48, 112])
    stack_size = int(checkpoint.get("stack_size", 4))

    model = GameDrivingCNN(in_channels=stack_size * 3, action_dim=len(label_names))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    key_state = CorrectionKeyState()
    listener = keyboard.Listener(on_press=key_state.on_press, on_release=key_state.on_release)
    listener.start()
    actuator = KeyActuator()

    crop = {"left": args.left, "top": args.top, "width": args.width, "height": args.height}
    writer = H5Writer(args.out, frame_shape=(96, 224, 3), crop=crop, fps=args.fps)
    frame_buffer: list[np.ndarray] = []
    label_buffer: list[np.ndarray] = []
    time_buffer: list[float] = []
    saved_label_totals = np.zeros(len(KEY_LABELS), dtype=np.float64)

    frames: deque[np.ndarray] = deque(maxlen=stack_size)
    interval = 1.0 / args.fps
    start_time = time.time()
    next_tick = time.perf_counter()
    preview_ready = False
    last_model_labels: set[str] = set()

    print("Correction collector ready.")
    print("F8 toggles model control. Hold Left Ctrl to take over, release Left Ctrl to return to model.")
    print("While holding Left Ctrl, use W/A/S/D/Space normally. F9 saves and quits.")
    print("Only Left-Ctrl human takeover frames are saved while model control is on.")
    print(f"Output: {args.out}")

    try:
        with mss.MSS() as sct:
            while not key_state.stop:
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(min(0.002, next_tick - now))
                    continue
                next_tick += interval

                raw_frame = capture_frame(sct, crop, args.backend)
                model_frame = cv2.resize(raw_frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
                frames.append(model_frame)
                while len(frames) < stack_size:
                    frames.append(model_frame)

                stacked = np.concatenate(list(frames), axis=2).astype(np.float32) / 255.0
                x = torch.from_numpy(np.transpose(stacked, (2, 0, 1))[None, ...])
                with torch.no_grad():
                    probs = torch.sigmoid(model(x)).squeeze(0).numpy()
                model_labels = choose_labels(label_names, probs, args, last_model_labels)
                last_model_labels = model_labels

                key_state.set_model_labels(model_labels)
                raw_human_labels = key_state.labels()
                human_override = key_state.human_control
                save_human_frame = human_override and bool(np.any(raw_human_labels[:5] > 0.5))
                human_labels = raw_human_labels
                applied_labels = set() if human_override else model_labels

                if key_state.recording:
                    if human_override:
                        key_state.set_model_labels(set())
                        actuator.release_all()
                    else:
                        key_state.set_model_labels(applied_labels)
                        actuator.apply(applied_labels)
                else:
                    key_state.set_model_labels(set())
                    actuator.release_all()
                    applied_labels = set()

                if key_state.recording and save_human_frame:
                    save_frame = cv2.resize(raw_frame, (224, 96), interpolation=cv2.INTER_AREA)
                    frame_buffer.append(save_frame)
                    label_buffer.append(human_labels)
                    time_buffer.append(time.time() - start_time)
                    saved_label_totals += human_labels
                    if len(frame_buffer) >= args.buffer_size:
                        writer.append(frame_buffer, label_buffer, time_buffer)
                        frame_buffer.clear()
                        label_buffer.clear()
                        time_buffer.clear()
                        print(f"samples={writer.count}", end="\r")

                if args.preview:
                    if not preview_ready:
                        cv2.namedWindow("PA_DRIVER correction collector", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("PA_DRIVER correction collector", 672, 288)
                        if args.preview_topmost:
                            cv2.setWindowProperty("PA_DRIVER correction collector", cv2.WND_PROP_TOPMOST, 1)
                        preview_ready = True
                    draw_preview(
                        model_frame,
                        key_state.recording,
                        human_override,
                        human_labels,
                        model_labels,
                        applied_labels,
                        set(actuator.held),
                        probs,
                        label_names,
                        writer.count + len(frame_buffer),
                        args.preview_topmost,
                    )
    finally:
        actuator.release_all()
        writer.append(frame_buffer, label_buffer, time_buffer)
        writer.close()
        listener.stop()
        if args.preview:
            cv2.destroyAllWindows()

    print()
    print(f"Saved {writer.count} correction samples to {args.out}")
    if writer.count:
        for name, count in zip(KEY_LABELS, saved_label_totals):
            print(f"{name}: count={int(count)} rate={count / writer.count:.3f}")
        if saved_label_totals[:4].sum() == 0 and saved_label_totals[4] > 0:
            print("WARNING: only Space was captured. Do not train with this file; run test_ctrl_key_capture.bat.")


if __name__ == "__main__":
    main()
