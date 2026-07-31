from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa_driver.model import DrivingCNN
from pa_driver.preprocessing import preprocess_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained behavior cloning policy in CarRacing-v3.")
    parser.add_argument("--model", type=Path, default=Path("runs/bc_carracing.pt"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steer", type=float, default=0.45)
    parser.add_argument("--steer-bias", type=float, default=0.0, help="Subtract this value from predicted steering before clipping.")
    parser.add_argument("--min-gas", type=float, default=0.25)
    parser.add_argument("--max-gas", type=float, default=1.0)
    parser.add_argument("--fixed-gas", type=float, default=None, help="Override model throttle with a fixed cruise value.")
    parser.add_argument("--pulse-gas", type=float, default=None, help="Throttle value used for pulse-and-coast mode.")
    parser.add_argument("--pulse-period", type=int, default=45, help="Number of frames in one throttle pulse cycle.")
    parser.add_argument("--pulse-on", type=int, default=8, help="Number of throttle-on frames per pulse cycle.")
    parser.add_argument("--target-speed", type=float, default=None, help="Set throttle to zero when CarRacing speed is above this value.")
    parser.add_argument("--cruise-control", action="store_true", help="Use a simple speed controller for throttle instead of model throttle.")
    parser.add_argument("--cruise-min-speed", type=float, default=5.0)
    parser.add_argument("--cruise-turn-speed", type=float, default=10.0)
    parser.add_argument("--cruise-straight-speed", type=float, default=16.0)
    parser.add_argument("--cruise-gas", type=float, default=0.25)
    parser.add_argument("--ignore-brake", action="store_true", help="Force brake to zero during inference.")
    parser.add_argument("--smooth", type=float, default=0.65, help="Previous-action smoothing factor from 0 to 0.95.")
    return parser.parse_args()


def get_carracing_speed(env: gym.Env) -> float:
    car = env.unwrapped.car
    velocity = car.hull.linearVelocity
    return float(np.hypot(velocity.x, velocity.y))


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    stack_size = int(checkpoint.get("stack_size", 4))

    model = DrivingCNN(in_channels=stack_size)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    env = gym.make("CarRacing-v3", render_mode="human")
    total_rewards = []

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        frames: deque[np.ndarray] = deque(maxlen=stack_size)
        for _ in range(stack_size):
            frames.append(preprocess_frame(obs))

        previous_action = np.array([0.0, args.min_gas, 0.0], dtype=np.float32)
        done = False
        episode_reward = 0.0
        episode_actions = []
        episode_speeds = []
        step_index = 0
        while not done:
            stacked = np.stack(list(frames), axis=0)[None, ...]
            with torch.no_grad():
                action = model(torch.from_numpy(stacked)).squeeze(0).numpy().astype(np.float32)

            action[0] = np.clip(action[0] - args.steer_bias, -args.max_steer, args.max_steer)
            speed = get_carracing_speed(env)
            episode_speeds.append(speed)
            if args.cruise_control:
                steer_abs = abs(float(action[0]))
                desired_speed = args.cruise_turn_speed if steer_abs > 0.12 else args.cruise_straight_speed
                if speed < args.cruise_min_speed:
                    action[1] = args.cruise_gas
                elif speed < desired_speed:
                    action[1] = args.cruise_gas * 0.6
                else:
                    action[1] = 0.0
            elif args.pulse_gas is not None:
                period = max(1, args.pulse_period)
                on_frames = int(np.clip(args.pulse_on, 0, period))
                action[1] = float(args.pulse_gas if (step_index % period) < on_frames else 0.0)
            elif args.fixed_gas is not None:
                action[1] = float(np.clip(args.fixed_gas, 0.0, 1.0))
            else:
                action[1] = max(float(action[1]), args.min_gas)

            if args.target_speed is not None and speed >= args.target_speed:
                action[1] = 0.0

            action[1] = float(np.clip(action[1], 0.0, args.max_gas))
            action[2] = 0.0 if args.ignore_brake else float(action[2])
            smooth = float(np.clip(args.smooth, 0.0, 0.95))
            action = (smooth * previous_action + (1.0 - smooth) * action).astype(np.float32)
            previous_action = action

            obs, reward, terminated, truncated, _ = env.step(action)
            frames.append(preprocess_frame(obs))
            episode_reward += float(reward)
            episode_actions.append(action)
            done = terminated or truncated
            step_index += 1

        total_rewards.append(episode_reward)
        actions = np.asarray(episode_actions, dtype=np.float32)
        speeds = np.asarray(episode_speeds, dtype=np.float32)
        action_mean = actions.mean(axis=0) if len(actions) else np.zeros(3, dtype=np.float32)
        speed_mean = float(speeds.mean()) if len(speeds) else 0.0
        speed_max = float(speeds.max()) if len(speeds) else 0.0
        print(
            f"episode={episode + 1} reward={episode_reward:.2f} "
            f"mean_action=[steer={action_mean[0]:.3f}, gas={action_mean[1]:.3f}, brake={action_mean[2]:.3f}] "
            f"speed=[mean={speed_mean:.2f}, max={speed_max:.2f}]"
        )

    env.close()
    print(f"mean_reward={np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    main()
