from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import pygame
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa_driver.model import DrivingCNN
from pa_driver.preprocessing import preprocess_frame
from scripts.collect_carracing import KEY_LABELS, key_labels_from_pressed, update_smooth_action
from scripts.run_carracing_policy import get_carracing_speed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect correction data while a model drives CarRacing-v3.")
    parser.add_argument("--model", type=Path, default=Path("runs/bc_drive04_balanced.pt"))
    parser.add_argument("--out", type=Path, default=Path("data/drive05_corrections.h5"))
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-steer", type=float, default=0.28)
    parser.add_argument("--max-gas", type=float, default=0.30)
    parser.add_argument("--target-speed", type=float, default=14.0)
    parser.add_argument("--smooth", type=float, default=0.82)
    parser.add_argument("--steer-step", type=float, default=0.08)
    parser.add_argument("--gas-rise", type=float, default=0.025)
    parser.add_argument("--gas-decay", type=float, default=0.05)
    parser.add_argument("--brake-power", type=float, default=0.8)
    return parser.parse_args()


def policy_action(
    model: DrivingCNN,
    frames: deque[np.ndarray],
    previous_action: np.ndarray,
    env: gym.Env,
    max_steer: float,
    max_gas: float,
    target_speed: float,
    smooth: float,
) -> np.ndarray:
    stacked = np.stack(list(frames), axis=0)[None, ...]
    with torch.no_grad():
        action = model(torch.from_numpy(stacked)).squeeze(0).numpy().astype(np.float32)

    action[0] = np.clip(action[0], -max_steer, max_steer)
    action[1] = np.clip(action[1], 0.0, max_gas)
    if get_carracing_speed(env) >= target_speed:
        action[1] = 0.0
    action[2] = 0.0
    return (smooth * previous_action + (1.0 - smooth) * action).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model, map_location="cpu")
    stack_size = int(checkpoint.get("stack_size", 4))
    model = DrivingCNN(in_channels=stack_size)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pygame.init()
    env = gym.make("CarRacing-v3", render_mode="human")
    obs, _ = env.reset(seed=args.seed)
    frames: deque[np.ndarray] = deque([preprocess_frame(obs)] * stack_size, maxlen=stack_size)

    observations: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    model_actions: list[np.ndarray] = []
    key_labels: list[np.ndarray] = []
    rewards: list[float] = []

    pressed_keys: set[int] = set()
    model_prev = np.zeros(3, dtype=np.float32)
    human_action = np.zeros(3, dtype=np.float32)
    running = True
    steps = 0
    correction_steps = 0

    print("Model drives by default. Hold W/A/S/D or arrows to override and record corrections.")
    print("Use R reset, Esc save and quit.")

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
                    frames = deque([preprocess_frame(obs)] * stack_size, maxlen=stack_size)
                    model_prev = np.zeros(3, dtype=np.float32)
                    human_action = np.zeros(3, dtype=np.float32)
            elif event.type == pygame.KEYUP:
                pressed_keys.discard(event.key)

        labels = key_labels_from_pressed(pressed_keys)
        human_action = update_smooth_action(
            human_action,
            labels,
            steer_step=args.steer_step,
            gas_rise=args.gas_rise,
            gas_decay=args.gas_decay,
            brake_power=args.brake_power,
        )
        model_action = policy_action(
            model,
            frames,
            model_prev,
            env,
            max_steer=args.max_steer,
            max_gas=args.max_gas,
            target_speed=args.target_speed,
            smooth=args.smooth,
        )
        model_prev = model_action

        human_override = bool(np.any(labels > 0.5) or np.any(np.abs(human_action) > 0.02))
        action = human_action if human_override else model_action
        next_obs, reward, terminated, truncated, _ = env.step(action)

        if human_override:
            observations.append(obs)
            expert_actions.append(human_action.copy())
            model_actions.append(model_action.copy())
            key_labels.append(labels.copy())
            rewards.append(float(reward))
            correction_steps += 1

        obs = next_obs
        frames.append(preprocess_frame(obs))
        steps += 1

        if terminated or truncated:
            obs, _ = env.reset()
            frames = deque([preprocess_frame(obs)] * stack_size, maxlen=stack_size)
            model_prev = np.zeros(3, dtype=np.float32)
            human_action = np.zeros(3, dtype=np.float32)

    env.close()
    pygame.quit()

    with h5py.File(args.out, "w") as h5:
        h5.create_dataset("observations", data=np.asarray(observations, dtype=np.uint8), compression="gzip")
        h5.create_dataset("actions", data=np.asarray(expert_actions, dtype=np.float32), compression="gzip")
        h5.create_dataset("model_actions", data=np.asarray(model_actions, dtype=np.float32), compression="gzip")
        h5.create_dataset("key_labels", data=np.asarray(key_labels, dtype=np.float32), compression="gzip")
        h5.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32), compression="gzip")
        h5.attrs["key_labels"] = ",".join(KEY_LABELS)
        h5.attrs["source_model"] = str(args.model)

    print(f"Saved {correction_steps} correction samples to {args.out}")
    if correction_steps:
        actions = np.asarray(expert_actions, dtype=np.float32)
        print(
            "Correction action mean: "
            f"steer={actions[:, 0].mean():.3f}, gas={actions[:, 1].mean():.3f}, brake={actions[:, 2].mean():.3f}"
        )


if __name__ == "__main__":
    main()
