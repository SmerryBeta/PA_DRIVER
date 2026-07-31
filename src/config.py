from __future__ import annotations

from dataclasses import dataclass, field, fields
import logging
from pathlib import Path

import yaml

LOG = logging.getLogger("Config")


@dataclass
class Crop:
    left: int = 0
    top: int = 170
    width: int = 1920
    height: int = 800


@dataclass
class AppConfig:
    image_dir: Path = Path("img")
    model_path: Path = Path("runs/party_kart_bc_corrected.pt")
    capture_fps: float = 30.0
    model_fps: float = 30.0
    match_threshold: float = 0.82
    step_delay: float = 0.6
    model_crop: Crop = field(default_factory=Crop)
    click_stable_frames: int = 2
    click_stable_tolerance_px: int = 6
    click_stable_interval: float = 0.08
    animated_templates: list[str] = field(default_factory=lambda: ["create", "lock"])
    animated_stable_frames: int = 6
    animated_stable_tolerance_px: int = 3
    animated_settle_delay: float = 0.3
    model_no_space: bool = False
    model_turn_threshold: float = 0.65
    model_w_threshold: float = 0.50
    model_w_release_threshold: float = 0.40
    model_s_threshold: float = 0.45
    model_space_threshold: float = 0.55
    stuck_recovery: bool = True
    stuck_diff_threshold: float = 1.2
    stuck_seconds: float = 1.5
    recovery_seconds: float = 1.1


def choose_default_model() -> Path:
    candidates = [
        Path("runs/party_kart_bc_cleaned_corrected.pt"),
        Path("runs/party_kart_bc_corrected.pt"),
        Path("runs/party_kart_bc_cleaned.pt"),
        Path("runs/party_kart_bc.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _coerce_path(value: object) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def load_config(path: Path) -> AppConfig:
    config = AppConfig(model_path=choose_default_model())
    if not path.exists():
        save_default_config(path, config)
        return config

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    apply_config_dict(config, data)
    return config


def save_default_config(path: Path, config: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_dir": str(config.image_dir),
        "model_path": str(config.model_path),
        "capture_fps": config.capture_fps,
        "model_fps": config.model_fps,
        "match_threshold": config.match_threshold,
        "step_delay": config.step_delay,
        "model_crop": {
            "left": config.model_crop.left,
            "top": config.model_crop.top,
            "width": config.model_crop.width,
            "height": config.model_crop.height,
        },
        "click_stable_frames": config.click_stable_frames,
        "click_stable_tolerance_px": config.click_stable_tolerance_px,
        "click_stable_interval": config.click_stable_interval,
        "animated_templates": config.animated_templates,
        "animated_stable_frames": config.animated_stable_frames,
        "animated_stable_tolerance_px": config.animated_stable_tolerance_px,
        "animated_settle_delay": config.animated_settle_delay,
        "model_no_space": config.model_no_space,
        "model_space_threshold": config.model_space_threshold,
        "model_turn_threshold": config.model_turn_threshold,
        "model_w_threshold": config.model_w_threshold,
        "model_w_release_threshold": config.model_w_release_threshold,
        "model_s_threshold": config.model_s_threshold,
        "stuck_recovery": config.stuck_recovery,
        "stuck_diff_threshold": config.stuck_diff_threshold,
        "stuck_seconds": config.stuck_seconds,
        "recovery_seconds": config.recovery_seconds,
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def apply_config_dict(config: AppConfig, data: dict[str, object]) -> None:
    valid_fields = {field.name for field in fields(AppConfig)}
    for key, value in data.items():
        if key not in valid_fields:
            LOG.warning("unknown config key ignored: %s", key)
            continue
        if key == "model_crop":
            crop_data = value or {}
            if not isinstance(crop_data, dict):
                raise ValueError("model_crop must be a mapping")
            config.model_crop = Crop(
                left=int(crop_data.get("left", config.model_crop.left)),
                top=int(crop_data.get("top", config.model_crop.top)),
                width=int(crop_data.get("width", config.model_crop.width)),
                height=int(crop_data.get("height", config.model_crop.height)),
            )
        elif key in {"image_dir", "model_path"}:
            setattr(config, key, _coerce_path(value))
        elif key == "animated_templates":
            if value is None:
                setattr(config, key, [])
            elif isinstance(value, str):
                setattr(config, key, [item.strip() for item in value.split(",") if item.strip()])
            elif isinstance(value, list):
                setattr(config, key, [str(item) for item in value])
            else:
                raise ValueError("animated_templates must be a list or comma-separated string")
        else:
            current = getattr(config, key)
            if isinstance(current, bool):
                setattr(config, key, bool(value))
            elif isinstance(current, int):
                setattr(config, key, int(value))
            elif isinstance(current, float):
                setattr(config, key, float(value))
            else:
                setattr(config, key, value)


def update_config_from_file(config: AppConfig, path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yml must contain a mapping")
    old_model = config.model_path
    apply_config_dict(config, data)
    if config.model_path != old_model:
        LOG.warning("model_path changed; restart is recommended before using the new model")
