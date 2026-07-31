from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym


def main() -> None:
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")

    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.close()

    print(f"obs_shape={obs.shape}")
    print(f"action_space={env.action_space}")
    print(f"reward={float(reward):.4f}, terminated={terminated}, truncated={truncated}")
    print("CarRacing-v3 smoke test passed.")


if __name__ == "__main__":
    main()
