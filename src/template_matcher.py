from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ALIASES = {
    "defaultcircle": "default_circle",
    "singlecircle": "single_circle",
    "selectnon": "select_non",
}


@dataclass(frozen=True)
class MatchResult:
    name: str
    score: float
    center: tuple[int, int]
    rect: tuple[int, int, int, int]


class TemplateMatcher:
    def __init__(self, image_dir: Path, threshold: float = 0.82) -> None:
        self.image_dir = image_dir
        self.threshold = threshold
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}
        for path in self.image_dir.glob("*.png"):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            templates[path.stem] = image
        if not templates:
            raise FileNotFoundError(f"No png templates found in {self.image_dir}")
        return templates

    def normalize_name(self, name: str) -> str:
        return ALIASES.get(name, name)

    def find(self, frame_rgb: np.ndarray, name: str, threshold: float | None = None) -> MatchResult | None:
        template_name = self.normalize_name(name)
        template = self.templates.get(template_name)
        if template is None:
            raise KeyError(f"Missing template: {template_name}.png")

        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if frame_gray.shape[0] < template.shape[0] or frame_gray.shape[1] < template.shape[1]:
            return None

        result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        min_score = self.threshold if threshold is None else threshold
        if max_val < min_score:
            return None

        x, y = max_loc
        h, w = template.shape[:2]
        return MatchResult(
            name=template_name,
            score=float(max_val),
            center=(x + w // 2, y + h // 2),
            rect=(x, y, w, h),
        )

    def any_found(self, frame_rgb: np.ndarray, names: list[str]) -> dict[str, MatchResult]:
        found: dict[str, MatchResult] = {}
        for name in names:
            match = self.find(frame_rgb, name)
            if match is not None:
                found[name] = match
        return found
