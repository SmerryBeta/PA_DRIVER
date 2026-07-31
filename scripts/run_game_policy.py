from __future__ import annotations

import argparse
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

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa_driver.game_model import GameDrivingCNN


KEY_MAP = {
    "w_forward": "w",
    "a_left": "a",
    "d_right": "d",
    "s_reverse": "s",
    "space_drift": keyboard.Key.space,
}


class RuntimeState:
    def __init__(self, start_active: bool = False) -> None:
        self.active = start_active
        self.stop = False
        self._last_toggle = 0.0

    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        now = time.perf_counter()
        if key == keyboard.Key.f8 and now - self._last_toggle > 0.35:
            self.active = not self.active
            self._last_toggle = now
            print("CONTROL ON" if self.active else "CONTROL PAUSED")
        elif key == keyboard.Key.f9:
            self.stop = True


class KeyActuator:
    def __init__(self) -> None:
        self.controller = keyboard.Controller()
        self.held: set[str] = set()

    def apply(self, labels: set[str], dry_run: bool = False) -> None:
        if dry_run:
            return

        wanted = {label for label in labels if label in KEY_MAP}
        for label in sorted(self.held - wanted):
            self.controller.release(KEY_MAP[label])
            self.held.remove(label)
        for label in sorted(wanted - self.held):
            self.controller.press(KEY_MAP[label])
            self.held.add(label)

    def release_all(self) -> None:
        for label in list(self.held):
            self.controller.release(KEY_MAP[label])
            self.held.remove(label)


class StuckRecovery:
    def __init__(
        self,
        fps: float,
        diff_threshold: float,
        stuck_seconds: float,
        recovery_seconds: float,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.stuck_frames_needed = max(1, int(stuck_seconds * fps))
        self.recovery_frames_left = 0
        self.recovery_frames = max(1, int(recovery_seconds * fps))
        self.still_frames = 0
        self.previous_gray: np.ndarray | None = None
        self.recovery_turn = "d_right"

    def update(self, frame: np.ndarray, model_labels: set[str], active: bool) -> set[str] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return None

        diff = float(np.mean(cv2.absdiff(gray, self.previous_gray)))
        self.previous_gray = gray

        if not active:
            self.still_frames = 0
            self.recovery_frames_left = 0
            return None

        if self.recovery_frames_left > 0:
            self.recovery_frames_left -= 1
            return {"s_reverse", self.recovery_turn}

        pushing = bool(model_labels & {"w_forward", "a_left", "d_right"})
        if pushing and diff < self.diff_threshold:
            self.still_frames += 1
        else:
            self.still_frames = 0

        if self.still_frames >= self.stuck_frames_needed:
            if "a_left" in model_labels:
                self.recovery_turn = "d_right"
            elif "d_right" in model_labels:
                self.recovery_turn = "a_left"
            else:
                self.recovery_turn = "a_left" if self.recovery_turn == "d_right" else "d_right"
            self.still_frames = 0
            self.recovery_frames_left = self.recovery_frames
            return {"s_reverse", self.recovery_turn}

        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained real-game keyboard policy.")
    parser.add_argument("--model", type=Path, default=Path("runs/party_kart_bc.pt"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--top", type=int, default=170)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--backend", choices=("auto", "mss", "pil"), default="auto")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-topmost", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-active", action="store_true")
    parser.add_argument("--w-threshold", type=float, default=0.50)
    parser.add_argument(
        "--w-release-threshold",
        type=float,
        default=0.40,
        help="Release W only after its probability drops below this value.",
    )
    parser.add_argument("--hold-w", action="store_true", help="Keep W pressed while model control is active.")
    parser.add_argument("--turn-threshold", type=float, default=0.50)
    parser.add_argument("--space-threshold", type=float, default=0.55)
    parser.add_argument("--s-threshold", type=float, default=0.80)
    parser.add_argument("--disable-s", action="store_true", default=True)
    parser.add_argument("--allow-s", action="store_false", dest="disable_s")
    parser.add_argument("--no-space", action="store_true")
    parser.add_argument("--stuck-recovery", action="store_true")
    parser.add_argument("--stuck-diff-threshold", type=float, default=1.2)
    parser.add_argument("--stuck-seconds", type=float, default=1.5)
    parser.add_argument("--recovery-seconds", type=float, default=1.1)
    return parser.parse_args()


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


def choose_labels(
    label_names: list[str],
    probs: np.ndarray,
    args: argparse.Namespace,
    previous_labels: set[str] | None = None,
) -> set[str]:
    prob_by_name = {name: float(prob) for name, prob in zip(label_names, probs)}
    labels: set[str] = set()
    previous_labels = previous_labels or set()

    w_prob = prob_by_name.get("w_forward", 0.0)
    w_release_threshold = getattr(args, "w_release_threshold", args.w_threshold)
    keep_previous_w = "w_forward" in previous_labels and w_prob >= w_release_threshold
    if getattr(args, "hold_w", False) or w_prob >= args.w_threshold or keep_previous_w:
        labels.add("w_forward")

    left = prob_by_name.get("a_left", 0.0)
    right = prob_by_name.get("d_right", 0.0)
    if max(left, right) >= args.turn_threshold:
        labels.add("a_left" if left > right else "d_right")

    if not args.disable_s and prob_by_name.get("s_reverse", 0.0) >= args.s_threshold:
        labels.add("s_reverse")

    if not args.no_space and prob_by_name.get("space_drift", 0.0) >= args.space_threshold:
        labels.add("space_drift")

    return labels


def draw_preview(
    frame: np.ndarray,
    probs: np.ndarray,
    label_names: list[str],
    labels: set[str],
    active: bool,
    dry_run: bool,
    topmost: bool,
) -> None:
    preview = cv2.resize(frame, (672, 288), interpolation=cv2.INTER_NEAREST)
    status = "ACTIVE" if active else "PAUSED"
    if dry_run:
        status += " DRY"
    color = (0, 220, 0) if active and not dry_run else (0, 180, 255)
    cv2.putText(preview, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    active_keys = "+".join(name.split("_", 1)[0].upper() for name in labels) or "-"
    cv2.putText(preview, f"keys={active_keys}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y = 88
    for name, prob in zip(label_names, probs):
        cv2.putText(preview, f"{name}: {prob:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22
    cv2.imshow("PA_DRIVER game policy", preview)
    if topmost:
        cv2.setWindowProperty("PA_DRIVER game policy", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    label_names = list(checkpoint["label_names"])
    input_height, input_width = checkpoint.get("input_size", [48, 112])
    stack_size = int(checkpoint.get("stack_size", 4))

    model = GameDrivingCNN(in_channels=stack_size * 3, action_dim=len(label_names))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    crop = {"left": args.left, "top": args.top, "width": args.width, "height": args.height}
    interval = 1.0 / args.fps
    frames: deque[np.ndarray] = deque(maxlen=stack_size)
    actuator = KeyActuator()
    state = RuntimeState(start_active=args.start_active)
    last_labels: set[str] = set()
    stuck_recovery = (
        StuckRecovery(args.fps, args.stuck_diff_threshold, args.stuck_seconds, args.recovery_seconds)
        if args.stuck_recovery
        else None
    )
    listener = keyboard.Listener(on_press=state.on_press)
    listener.start()

    print("Real-game policy ready.")
    print("F8 start/pause control, F9 quit and release keys.")
    print(f"Model: {args.model}")
    print(f"Crop: left={args.left}, top={args.top}, width={args.width}, height={args.height}")
    print(f"Mode: {'dry-run preview' if args.dry_run else 'keyboard control'}")
    preview_ready = False

    next_tick = time.perf_counter()
    try:
        with mss.MSS() as sct:
            while not state.stop:
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(min(0.002, next_tick - now))
                    continue
                next_tick += interval

                frame = capture_frame(sct, crop, args.backend)
                frame = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
                frames.append(frame)
                while len(frames) < stack_size:
                    frames.append(frame)

                stacked = np.concatenate(list(frames), axis=2).astype(np.float32) / 255.0
                x = torch.from_numpy(np.transpose(stacked, (2, 0, 1))[None, ...])
                with torch.no_grad():
                    probs = torch.sigmoid(model(x)).squeeze(0).numpy()

                labels = choose_labels(label_names, probs, args, last_labels)
                if stuck_recovery is not None:
                    recovery_labels = stuck_recovery.update(frame, labels, state.active)
                    if recovery_labels is not None:
                        labels = recovery_labels
                last_labels = labels
                if state.active:
                    actuator.apply(labels, dry_run=args.dry_run)
                else:
                    actuator.release_all()

                if args.preview:
                    if not preview_ready:
                        cv2.namedWindow("PA_DRIVER game policy", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("PA_DRIVER game policy", 672, 288)
                        if args.preview_topmost:
                            cv2.setWindowProperty("PA_DRIVER game policy", cv2.WND_PROP_TOPMOST, 1)
                        preview_ready = True
                    draw_preview(frame, probs, label_names, labels, state.active, args.dry_run, args.preview_topmost)
    finally:
        actuator.release_all()
        listener.stop()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



