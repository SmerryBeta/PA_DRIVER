from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import pygame


KEY_LABELS = ("w_gas", "a_left", "d_right", "s_reverse", "space_drift")
KEY_BINDINGS = {
    "w_gas": {pygame.K_w, pygame.K_UP},
    "a_left": {pygame.K_a, pygame.K_LEFT},
    "d_right": {pygame.K_d, pygame.K_RIGHT},
    "s_reverse": {pygame.K_s, pygame.K_DOWN},
    "space_drift": {pygame.K_SPACE},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect human driving data from CarRacing-v3.")
    parser.add_argument("--out", type=Path, default=Path("data/carracing_manual.h5"))
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-idle", action="store_true", help="Keep frames where no driving key is pressed.")
    parser.add_argument(
        "--auto-gas-with-steer",
        action="store_true",
        help="In CarRacing-v3 only, treat A/D steering as W+A or W+D unless S is pressed.",
    )
    parser.add_argument("--steer-step", type=float, default=0.12, help="How quickly keyboard steering moves toward the target.")
    parser.add_argument("--gas-rise", type=float, default=0.04, help="How quickly W increases throttle.")
    parser.add_argument("--gas-decay", type=float, default=0.03, help="How quickly throttle returns to zero after releasing W.")
    parser.add_argument("--brake-power", type=float, default=0.8, help="Brake value used while S is held.")
    return parser.parse_args()


def key_labels_from_pressed(pressed_keys: set[int]) -> np.ndarray:
    return np.array(
        [
            bool(KEY_BINDINGS["w_gas"] & pressed_keys),
            bool(KEY_BINDINGS["a_left"] & pressed_keys),
            bool(KEY_BINDINGS["d_right"] & pressed_keys),
            bool(KEY_BINDINGS["s_reverse"] & pressed_keys),
            bool(KEY_BINDINGS["space_drift"] & pressed_keys),
        ],
        dtype=np.float32,
    )


def action_from_labels(key_labels: np.ndarray, auto_gas_with_steer: bool = False) -> np.ndarray:
    steer = 0.0
    if key_labels[1]:
        steer -= 1.0
    if key_labels[2]:
        steer += 1.0

    gas = 1.0 if key_labels[0] or (auto_gas_with_steer and steer != 0.0 and not key_labels[3]) else 0.0
    brake = 1.0 if key_labels[3] else 0.0
    return np.array([steer, gas, brake], dtype=np.float32)


def update_smooth_action(
    previous_action: np.ndarray,
    key_labels: np.ndarray,
    steer_step: float,
    gas_rise: float,
    gas_decay: float,
    brake_power: float,
    auto_gas_with_steer: bool = False,
) -> np.ndarray:
    target_steer = 0.0
    if key_labels[1]:
        target_steer -= 1.0
    if key_labels[2]:
        target_steer += 1.0

    steer_delta = np.clip(target_steer - previous_action[0], -steer_step, steer_step)
    steer = float(previous_action[0] + steer_delta)

    wants_gas = bool(key_labels[0] or (auto_gas_with_steer and target_steer != 0.0 and not key_labels[3]))
    gas_delta = gas_rise if wants_gas else -gas_decay
    gas = float(np.clip(previous_action[1] + gas_delta, 0.0, 1.0))

    brake = float(brake_power if key_labels[3] else 0.0)
    if brake > 0.0:
        gas = 0.0

    return np.array([steer, gas, brake], dtype=np.float32)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    pygame.init()
    env = gym.make("CarRacing-v3", render_mode="human")
    obs, _ = env.reset(seed=args.seed)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    key_labels: list[np.ndarray] = []
    rewards: list[float] = []

    running = True
    pressed_keys: set[int] = set()
    action = np.zeros(3, dtype=np.float32)
    steps = 0
    print("Controls: W gas, A left, D right, S reverse/brake, Space drift label, R reset, Esc quit.")
    print("Note: CarRacing-v3 has no drift action; Space is recorded in key_labels only.")
    if not args.keep_idle:
        print("Idle frames are skipped. Use --keep-idle if you need to record no-input frames.")

    while running and steps < args.max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                pressed_keys.add(event.key)
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
            elif event.type == pygame.KEYUP:
                pressed_keys.discard(event.key)

        labels = key_labels_from_pressed(pressed_keys)
        action = update_smooth_action(
            action,
            labels,
            steer_step=args.steer_step,
            gas_rise=args.gas_rise,
            gas_decay=args.gas_decay,
            brake_power=args.brake_power,
            auto_gas_with_steer=args.auto_gas_with_steer,
        )
        next_obs, reward, terminated, truncated, _ = env.step(action)

        should_keep = args.keep_idle or bool(np.any(labels > 0.5)) or bool(np.any(np.abs(action) > 0.01))
        if should_keep:
            observations.append(obs)
            actions.append(action)
            key_labels.append(labels)
            rewards.append(float(reward))

        obs = next_obs
        steps += 1
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    pygame.quit()

    with h5py.File(args.out, "w") as h5:
        h5.create_dataset("observations", data=np.asarray(observations, dtype=np.uint8), compression="gzip")
        h5.create_dataset("actions", data=np.asarray(actions, dtype=np.float32), compression="gzip")
        h5.create_dataset("key_labels", data=np.asarray(key_labels, dtype=np.float32), compression="gzip")
        h5.attrs["key_labels"] = ",".join(KEY_LABELS)
        h5.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32), compression="gzip")

    actions_array = np.asarray(actions, dtype=np.float32)
    labels_array = np.asarray(key_labels, dtype=np.float32)
    print(f"Saved {len(actions)} samples to {args.out}")
    if len(actions_array) > 0:
        print(
            "Action mean: "
            f"steer={actions_array[:, 0].mean():.3f}, gas={actions_array[:, 1].mean():.3f}, "
            f"brake={actions_array[:, 2].mean():.3f}"
        )
        counts = labels_array.sum(axis=0).astype(int)
        print("Key counts: " + ", ".join(f"{name}={count}" for name, count in zip(KEY_LABELS, counts)))


if __name__ == "__main__":
    main()
