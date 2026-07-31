from __future__ import annotations

import threading
import time
import logging
from pathlib import Path
from typing import Callable

LOG = logging.getLogger("ConfigWatch")


class ConfigWatcher:
    def __init__(self, path: Path, on_change: Callable[[], None], poll_seconds: float = 0.8) -> None:
        self.path = path
        self.on_change = on_change
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer = None

    def start(self) -> None:
        if self._try_start_watchdog():
            return
        self._thread = threading.Thread(target=self._poll_loop, name="config-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _try_start_watchdog(self) -> bool:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except Exception:
            return False

        target = self.path.resolve()

        class Handler(FileSystemEventHandler):
            def __init__(self, callback: Callable[[], None]) -> None:
                self.callback = callback

            def on_modified(self, event) -> None:  # type: ignore[no-untyped-def]
                if Path(event.src_path).resolve() == target:
                    self.callback()

            def on_created(self, event) -> None:  # type: ignore[no-untyped-def]
                if Path(event.src_path).resolve() == target:
                    self.callback()

        observer = Observer()
        observer.schedule(Handler(self.on_change), str(target.parent), recursive=False)
        observer.start()
        self._observer = observer
        LOG.info("watcher=watchdog path=%s", self.path)
        return True

    def _poll_loop(self) -> None:
        LOG.info("watcher=polling path=%s", self.path)
        last_mtime = self._mtime()
        while not self._stop.is_set():
            time.sleep(self.poll_seconds)
            current_mtime = self._mtime()
            if current_mtime is not None and current_mtime != last_mtime:
                last_mtime = current_mtime
                self.on_change()

    def _mtime(self) -> float | None:
        try:
            return self.path.stat().st_mtime
        except FileNotFoundError:
            return None
