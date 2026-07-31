from __future__ import annotations

import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray, size: int = 96) -> np.ndarray:
    """Convert an RGB frame to normalized grayscale data."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape HxWx3, got {frame.shape}")

    if frame.shape[0] != size or frame.shape[1] != size:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def make_frame_stack(frames: np.ndarray, end_index: int, stack_size: int = 4) -> np.ndarray:
    """Build a stack ending at end_index, repeating the first frame near boundaries."""
    if end_index < 0 or end_index >= len(frames):
        raise IndexError(f"end_index {end_index} outside frame array of length {len(frames)}")

    stack = []
    for offset in range(stack_size - 1, -1, -1):
        frame_index = max(0, end_index - offset)
        stack.append(preprocess_frame(frames[frame_index]))
    return np.stack(stack, axis=0)
