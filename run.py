from __future__ import annotations

import argparse
import threading
from pathlib import Path
import logging

from src.config import load_config, save_default_config, update_config_from_file
from src.config_watch import ConfigWatcher
from src.input_control import InputController
from src.model_driver import ModelDriver
from src.screen_capture import ScreenCapture
from src.template_matcher import TemplateMatcher
from src.workflow import WorkflowController
from src.logging_setup import setup_logging

LOG = logging.getLogger("App")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PA_DRIVER application entrypoint.")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--write-default-config", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)
    config = load_config(args.config)
    if args.write_default_config:
        save_default_config(args.config, config)
        LOG.info("wrote %s", args.config)
        return

    reload_lock = threading.Lock()

    def reload_config() -> None:
        with reload_lock:
            try:
                update_config_from_file(config, args.config)
                capture.fps = config.capture_fps
                matcher.threshold = config.match_threshold
                LOG.info("reloaded %s", args.config)
            except Exception as exc:
                LOG.exception("failed to reload %s: %s", args.config, exc)

    capture = ScreenCapture(fps=config.capture_fps)
    input_controller = InputController(dry_run=args.dry_run)
    matcher = TemplateMatcher(config.image_dir, threshold=config.match_threshold)
    model_driver = ModelDriver(config, capture, input_controller)
    workflow = WorkflowController(config, capture, matcher, input_controller, model_driver)
    watcher = ConfigWatcher(args.config, reload_config)
    watcher.start()
    try:
        workflow.run_forever()
    finally:
        watcher.stop()


if __name__ == "__main__":
    main()
