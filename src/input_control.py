from __future__ import annotations

import threading
import time
import logging

from pynput import keyboard, mouse

from src.timing import cancellable_sleep

LOG = logging.getLogger("Input")


KEY_MAP = {
    "w_forward": "w",
    "a_left": "a",
    "d_right": "d",
    "s_reverse": "s",
    "space_drift": keyboard.Key.space,
}


class InputController:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.keyboard = keyboard.Controller()
        self.mouse = mouse.Controller()
        self.held: set[str] = set()
        self._lock = threading.Lock()

    def click(self, x: int, y: int) -> None:
        if self.dry_run:
            LOG.info("DRY click x=%s y=%s", x, y)
            return
        self.mouse.position = (x, y)
        time.sleep(0.03)
        self.mouse.click(mouse.Button.left, 1)

    def tap_key(self, key: str | keyboard.Key, duration: float = 0.04) -> None:
        if self.dry_run:
            LOG.info("DRY tap key=%s", key)
            return
        self.keyboard.press(key)
        time.sleep(duration)
        self.keyboard.release(key)

    def tap_space(self) -> None:
        self.tap_key(keyboard.Key.space)

    def tap_space_after(self, seconds: float, stop_event: threading.Event) -> bool:
        if not cancellable_sleep(seconds, stop_event):
            return False
        self.tap_space()
        return True

    def apply_labels(self, labels: set[str]) -> None:
        with self._lock:
            if self.dry_run:
                return
            wanted = {label for label in labels if label in KEY_MAP}
            for label in sorted(self.held - wanted):
                self.keyboard.release(KEY_MAP[label])
                self.held.remove(label)
            for label in sorted(wanted - self.held):
                self.keyboard.press(KEY_MAP[label])
                self.held.add(label)

    def release_all(self) -> None:
        with self._lock:
            for label in list(self.held):
                self.keyboard.release(KEY_MAP[label])
                self.held.remove(label)
