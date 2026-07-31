from __future__ import annotations

import logging
import sys
from pathlib import Path


class LogcatFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.DEBUG: "D",
        logging.INFO: "I",
        logging.WARNING: "W",
        logging.ERROR: "E",
        logging.CRITICAL: "F",
    }

    def format(self, record: logging.LogRecord) -> str:
        level = self.LEVEL_MAP.get(record.levelno, record.levelname[:1])
        timestamp = self.formatTime(record, "%H:%M:%S")
        tag = record.name[-22:]
        return f"{timestamp} {level}/{tag}: {record.getMessage()}"


def setup_logging(log_file: Path = Path("logs/pa_driver.log"), verbose: bool = False) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(LogcatFormatter())
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(LogcatFormatter())
    root.addHandler(file_handler)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
