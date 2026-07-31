from __future__ import annotations

import threading
import time
import logging

import cv2
import mss
import numpy as np

LOG = logging.getLogger("ScreenCapture")


class ScreenCapture:
    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="screen-capture", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            LOG.warning("capture did not produce a frame within 5s")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_frame(self, copy: bool = True) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy() if copy else self._frame

    def crop(self, frame: np.ndarray, left: int, top: int, width: int, height: int) -> np.ndarray:
        return frame[top : top + height, left : left + width]

    def _run(self) -> None:
        next_tick = time.perf_counter()
        try:
            factory = getattr(mss, "MSS", None) or getattr(mss, "mss", None)
            if factory is None:
                raise AttributeError("mss module has neither MSS nor mss")
            with factory() as sct:
                monitor = sct.monitors[1]
                LOG.info("capture started monitor=%s fps=%.1f", monitor, self.fps)
                while not self._stop.is_set():
                    interval = 1.0 / max(1.0, self.fps)
                    now = time.perf_counter()
                    if now < next_tick:
                        time.sleep(min(0.002, next_tick - now))
                        continue
                    next_tick += interval
                    shot = np.asarray(sct.grab(monitor), dtype=np.uint8)
                    frame = cv2.cvtColor(shot[:, :, :3], cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._frame = frame
                    self._ready.set()
        except Exception:
            LOG.exception("capture thread crashed")
            self._ready.set()
