from __future__ import annotations

import threading
import time


def cancellable_sleep(seconds: float, stop_event: threading.Event, step: float = 0.05) -> bool:
    end_time = time.perf_counter() + seconds
    while not stop_event.is_set():
        remaining = end_time - time.perf_counter()
        if remaining <= 0:
            return True
        time.sleep(min(step, remaining))
    return False
