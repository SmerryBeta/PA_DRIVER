from __future__ import annotations

import threading
import time
import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from pa_driver.game_model import GameDrivingCNN
from src.config import AppConfig
from src.input_control import InputController
from src.screen_capture import ScreenCapture

LOG = logging.getLogger("Model")


class StuckRecovery:
    def __init__(self, config: AppConfig) -> None:
        self.diff_threshold = config.stuck_diff_threshold
        self.stuck_frames_needed = max(1, int(config.stuck_seconds * config.model_fps))
        self.recovery_frames = max(1, int(config.recovery_seconds * config.model_fps))
        self.recovery_frames_left = 0
        self.still_frames = 0
        self.previous_gray: np.ndarray | None = None
        self.recovery_turn = "d_right"

    def update(self, frame: np.ndarray, labels: set[str]) -> set[str] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return None

        diff = float(np.mean(cv2.absdiff(gray, self.previous_gray)))
        self.previous_gray = gray

        if self.recovery_frames_left > 0:
            self.recovery_frames_left -= 1
            return {"s_reverse", self.recovery_turn}

        pushing = bool(labels & {"w_forward", "a_left", "d_right"})
        if pushing and diff < self.diff_threshold:
            self.still_frames += 1
        else:
            self.still_frames = 0

        if self.still_frames >= self.stuck_frames_needed:
            if "a_left" in labels:
                self.recovery_turn = "d_right"
            elif "d_right" in labels:
                self.recovery_turn = "a_left"
            else:
                self.recovery_turn = "a_left" if self.recovery_turn == "d_right" else "d_right"
            self.still_frames = 0
            self.recovery_frames_left = self.recovery_frames
            return {"s_reverse", self.recovery_turn}

        return None


class ModelDriver:
    def __init__(self, config: AppConfig, capture: ScreenCapture, input_controller: InputController) -> None:
        self.config = config
        self.capture = capture
        self.input = input_controller
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.loaded = False
        self._load_lock = threading.Lock()

    def load(self) -> None:
        with self._load_lock:
            checkpoint = torch.load(self.config.model_path, map_location="cpu")
            self.label_names = list(checkpoint["label_names"])
            self.input_height, self.input_width = checkpoint.get("input_size", [48, 112])
            self.stack_size = int(checkpoint.get("stack_size", 4))
            self.model = GameDrivingCNN(in_channels=self.stack_size * 3, action_dim=len(self.label_names))
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval()
            self.loaded = True
            LOG.info("loaded model=%s labels=%s input=%sx%s stack=%s", self.config.model_path, self.label_names, self.input_width, self.input_height, self.stack_size)

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        if not self.loaded:
            self.load()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="model-driver", daemon=True)
        self.thread.start()
        LOG.info("ON")

    def stop(self) -> None:
        self.stop_event.set()
        self.input.release_all()
        if self.thread:
            self.thread.join(timeout=2.0)
        LOG.info("OFF")

    def is_active(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def _choose_labels(self, probs: np.ndarray, previous_labels: set[str]) -> set[str]:
        prob_by_name = {name: float(prob) for name, prob in zip(self.label_names, probs)}
        labels: set[str] = set()

        w_prob = prob_by_name.get("w_forward", 0.0)
        keep_w = "w_forward" in previous_labels and w_prob >= self.config.model_w_release_threshold
        if w_prob >= self.config.model_w_threshold or keep_w:
            labels.add("w_forward")

        left = prob_by_name.get("a_left", 0.0)
        right = prob_by_name.get("d_right", 0.0)
        if max(left, right) >= self.config.model_turn_threshold:
            labels.add("a_left" if left > right else "d_right")

        if prob_by_name.get("s_reverse", 0.0) >= self.config.model_s_threshold:
            labels.add("s_reverse")

        if not self.config.model_no_space and prob_by_name.get("space_drift", 0.0) >= self.config.model_space_threshold:
            labels.add("space_drift")

        return labels

    def _run(self) -> None:
        frames: deque[np.ndarray] = deque(maxlen=self.stack_size)
        previous_labels: set[str] = set()
        recovery = StuckRecovery(self.config) if self.config.stuck_recovery else None
        next_tick = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                interval = 1.0 / max(1.0, self.config.model_fps)
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(min(0.002, next_tick - now))
                    continue
                next_tick += interval

                full_frame = self.capture.get_frame(copy=True)
                if full_frame is None:
                    continue
                crop = self.config.model_crop
                frame = self.capture.crop(full_frame, crop.left, crop.top, crop.width, crop.height)
                model_frame = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
                frames.append(model_frame)
                while len(frames) < self.stack_size:
                    frames.append(model_frame)

                stacked = np.concatenate(list(frames), axis=2).astype(np.float32) / 255.0
                x = torch.from_numpy(np.transpose(stacked, (2, 0, 1))[None, ...])
                with torch.no_grad():
                    probs = torch.sigmoid(self.model(x)).squeeze(0).numpy()

                labels = self._choose_labels(probs, previous_labels)
                if recovery is not None:
                    recovery_labels = recovery.update(model_frame, labels)
                    if recovery_labels is not None:
                        labels = recovery_labels
                        LOG.info("stuck recovery labels=%s", sorted(labels))
                previous_labels = labels
                self.input.apply_labels(labels)
        except Exception:
            LOG.exception("driver thread crashed")
        finally:
            self.input.release_all()
