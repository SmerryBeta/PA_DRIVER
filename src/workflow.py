from __future__ import annotations

import threading
import time
import logging

from pynput import keyboard

from src.config import AppConfig
from src.input_control import InputController
from src.model_driver import ModelDriver
from src.screen_capture import ScreenCapture
from src.template_matcher import MatchResult, TemplateMatcher
from src.timing import cancellable_sleep


CREATE_STEPS = ["diy", "create", "lab", "cart", "selectnon", "formula", "create_confirm"]
SETTING_STEPS = ["race_setting", "defaultcircle", "singlecircle", "apply"]
POST_RACE_PRIZE_STEPS = ["prize_open_all", "prize_continue"]
LOG = logging.getLogger("Workflow")


class WorkflowController:
    def __init__(
        self,
        config: AppConfig,
        capture: ScreenCapture,
        matcher: TemplateMatcher,
        input_controller: InputController,
        model_driver: ModelDriver,
    ) -> None:
        self.config = config
        self.capture = capture
        self.matcher = matcher
        self.input = input_controller
        self.model = model_driver
        self.current_stop = threading.Event()
        self.app_stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.listener: keyboard.Listener | None = None

    def run_forever(self) -> None:
        self.capture.start()
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        LOG.info("ready")
        LOG.info("F8 start/stop script or model. F9 quit.")
        try:
            while not self.app_stop.is_set():
                time.sleep(0.1)
        finally:
            self.stop_current()
            self.capture.stop()
            if self.listener:
                self.listener.stop()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key == keyboard.Key.f8:
            if self.is_busy():
                self.stop_current()
            else:
                self.start_current()
        elif key == keyboard.Key.f9:
            self.app_stop.set()

    def is_busy(self) -> bool:
        return self.model.is_active() or bool(self.worker and self.worker.is_alive())

    def start_current(self) -> None:
        self.current_stop = threading.Event()
        self.worker = threading.Thread(target=self._entrypoint, name="workflow", daemon=True)
        self.worker.start()

    def stop_current(self) -> None:
        self.current_stop.set()
        self.model.stop()
        self.input.release_all()

    def _entrypoint(self) -> None:
        frame = self._latest_frame(timeout=2.0)
        if frame is None:
            LOG.error("no screen frame available")
            return

        found = self.matcher.any_found(frame, ["diy", "start"])
        if "diy" in found:
            LOG.info("found diy; creating room")
            self._create_until_ready()
        elif "start" in found:
            LOG.info("found start; running race loop")
            self._race_loop()
        else:
            LOG.info("no diy/start found; starting model directly")
            self.model.start()

    def _latest_frame(self, timeout: float = 10.0):
        end_time = time.perf_counter() + timeout
        while not self.current_stop.is_set() and time.perf_counter() < end_time:
            frame = self.capture.get_frame(copy=True)
            if frame is not None:
                return frame
            time.sleep(0.05)
        return None

    def _wait_match(self, name: str, timeout: float | None = 30.0) -> MatchResult | None:
        end_time = None if timeout is None else time.perf_counter() + timeout
        while not self.current_stop.is_set():
            frame = self.capture.get_frame(copy=True)
            if frame is not None:
                match = self.matcher.find(frame, name)
                if match is not None:
                    return match
            if end_time is not None and time.perf_counter() >= end_time:
                return None
            time.sleep(0.08)
        return None

    def _is_same_position(self, left: MatchResult, right: MatchResult, tolerance_px: int) -> bool:
        return (
            abs(left.center[0] - right.center[0]) <= tolerance_px
            and abs(left.center[1] - right.center[1]) <= tolerance_px
        )

    def _stable_click_settings(self, name: str) -> tuple[int, int, float, float]:
        template_name = self.matcher.normalize_name(name)
        animated = {
            self.matcher.normalize_name(item)
            for item in self.config.animated_templates
        }
        if template_name in animated:
            return (
                max(1, int(self.config.animated_stable_frames)),
                max(0, int(self.config.animated_stable_tolerance_px)),
                max(0.01, float(self.config.click_stable_interval)),
                max(0.0, float(self.config.animated_settle_delay)),
            )
        return (
            max(1, int(self.config.click_stable_frames)),
            max(0, int(self.config.click_stable_tolerance_px)),
            max(0.01, float(self.config.click_stable_interval)),
            0.0,
        )

    def _wait_stable_match(self, name: str, timeout: float | None = 30.0) -> MatchResult | None:
        stable_frames, tolerance_px, interval, settle_delay = self._stable_click_settings(name)
        if stable_frames <= 1 and settle_delay <= 0:
            return self._wait_match(name, timeout)

        end_time = None if timeout is None else time.perf_counter() + timeout
        last_match: MatchResult | None = None
        stable_count = 0

        while not self.current_stop.is_set():
            if end_time is not None and time.perf_counter() >= end_time:
                return None

            frame = self.capture.get_frame(copy=True)
            if frame is not None:
                match = self.matcher.find(frame, name)
                if match is None:
                    last_match = None
                    stable_count = 0
                elif last_match is not None and self._is_same_position(match, last_match, tolerance_px):
                    last_match = match
                    stable_count += 1
                else:
                    last_match = match
                    stable_count = 1

                if last_match is not None and stable_count >= stable_frames:
                    if settle_delay > 0:
                        if not cancellable_sleep(settle_delay, self.current_stop):
                            return None
                        frame = self.capture.get_frame(copy=True)
                        if frame is None:
                            continue
                        settled_match = self.matcher.find(frame, name)
                        if settled_match is not None and self._is_same_position(
                            settled_match,
                            last_match,
                            tolerance_px,
                        ):
                            LOG.info(
                                "stable %s score=%.3f at=%s frames=%s",
                                name,
                                settled_match.score,
                                settled_match.center,
                                stable_count,
                            )
                            return settled_match
                        last_match = settled_match
                        stable_count = 1 if settled_match is not None else 0
                    else:
                        LOG.info(
                            "stable %s score=%.3f at=%s frames=%s",
                            name,
                            last_match.score,
                            last_match.center,
                            stable_count,
                        )
                        return last_match

            time.sleep(interval)
        return None

    def _click_template(self, name: str, timeout: float | None = 30.0) -> bool:
        match = self._wait_stable_match(name, timeout)
        if match is None:
            LOG.warning("timeout waiting for %s", name)
            return False
        LOG.info("click %s score=%.3f at=%s", name, match.score, match.center)
        self.input.click(*match.center)
        return cancellable_sleep(self.config.step_delay, self.current_stop)

    def _create_until_ready(self) -> None:
        while not self.current_stop.is_set():
            frame = self.capture.get_frame(copy=True)
            if frame is not None and self.matcher.find(frame, "diy_room_home") is not None:
                self._click_template("return", timeout=10.0)
                continue

            if not self._run_create_steps():
                continue
            if self._run_room_settings():
                self._race_loop()
                return

            frame = self.capture.get_frame(copy=True)
            if frame is not None and self.matcher.find(frame, "diy_room_home") is not None:
                self._click_template("return", timeout=10.0)

    def _run_create_steps(self) -> bool:
        for name in CREATE_STEPS:
            if self.current_stop.is_set():
                return False
            if not self._click_template(name, timeout=20.0):
                return False
        return True

    def _run_room_settings(self) -> bool:
        if not self._click_template("lock", timeout=60.0):
            LOG.warning("no lock found within 60s; treat as create failure")
            return False
        if not cancellable_sleep(1.5, self.current_stop):
            return False
        self.input.tap_space()
        if not cancellable_sleep(self.config.step_delay, self.current_stop):
            return False

        for name in SETTING_STEPS:
            if not self._click_template(name, timeout=20.0):
                return False
        return True

    def _race_loop(self) -> None:
        while not self.current_stop.is_set():
            if not self._click_template("start", timeout=None):
                return
            if not cancellable_sleep(10.0, self.current_stop):
                return

            if self._wait_match("continue", timeout=120.0) is not None:
                self.input.tap_space()
            if not cancellable_sleep(self.config.step_delay, self.current_stop):
                return

            if self._wait_match("space", timeout=120.0) is not None:
                self.input.tap_space()
            if not cancellable_sleep(5.0, self.current_stop):
                return

            self.model.start()
            self._wait_finish_and_return_to_start()

    def _wait_finish_and_return_to_start(self) -> None:
        while not self.current_stop.is_set():
            frame = self.capture.get_frame(copy=True)
            if frame is not None and self.matcher.find(frame, "finish_all_race") is not None:
                LOG.info("race finished")
                self.model.stop()
                self.input.release_all()
                self._handle_post_race_until_start(timeout=120.0)
                return
            time.sleep(0.2)

    def _handle_post_race_until_start(self, timeout: float = 120.0) -> bool:
        end_time = time.perf_counter() + timeout
        while not self.current_stop.is_set() and time.perf_counter() < end_time:
            frame = self.capture.get_frame(copy=True)
            if frame is None:
                time.sleep(0.1)
                continue

            if self.matcher.find(frame, "start") is not None:
                LOG.info("returned to start")
                return True

            for name in POST_RACE_PRIZE_STEPS:
                if self.matcher.find(frame, name) is not None:
                    LOG.info("post-race popup: %s", name)
                    if not self._click_template(name, timeout=5.0):
                        return False
                    break
            else:
                time.sleep(0.2)

        LOG.warning("timeout waiting for start after race")
        return False
