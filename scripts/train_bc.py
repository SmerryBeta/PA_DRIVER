from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa_driver.model import DrivingCNN
from pa_driver.preprocessing import make_frame_stack


class DrivingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_paths: list[Path],
        stack_size: int = 4,
        drop_idle: bool = False,
        gas_when_steering: bool = False,
        augment_flip: bool = False,
    ) -> None:
        observations = []
        actions = []
        for data_path in data_paths:
            with h5py.File(data_path, "r") as h5:
                observations.append(h5["observations"][:])
                actions.append(h5["actions"][:])
        self.observations = np.concatenate(observations, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.stack_size = stack_size
        self.gas_when_steering = gas_when_steering
        self.augment_flip = augment_flip

        if drop_idle:
            active = np.any(np.abs(self.actions) > 0.01, axis=1)
            self.indices = np.flatnonzero(active)
        else:
            self.indices = np.arange(len(self.actions))

    def __len__(self) -> int:
        multiplier = 2 if self.augment_flip else 1
        return len(self.indices) * multiplier

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.action_at(index)
        base_length = len(self.indices)
        flip = self.augment_flip and index >= base_length
        source_index = int(self.indices[index % base_length])
        frames = make_frame_stack(self.observations, source_index, self.stack_size)
        if flip:
            frames = np.flip(frames, axis=2).copy()
        return torch.from_numpy(frames), torch.from_numpy(action)

    def action_at(self, index: int) -> np.ndarray:
        base_length = len(self.indices)
        flip = self.augment_flip and index >= base_length
        source_index = int(self.indices[index % base_length])
        action = self.actions[source_index].astype(np.float32).copy()
        if flip:
            action[0] = -action[0]
        if self.gas_when_steering and abs(float(action[0])) > 0.01 and float(action[2]) < 0.5:
            action[1] = 1.0
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a behavior cloning policy for CarRacing-v3.")
    parser.add_argument("--data", type=Path, nargs="+", default=[Path("data/carracing_manual.h5")])
    parser.add_argument("--out", type=Path, default=Path("runs/bc_carracing.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stack-size", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--drop-idle", action="store_true", help="Ignore frames where no driving action was pressed.")
    parser.add_argument("--gas-when-steering", action="store_true", help="Treat steering without braking as steering while holding gas.")
    parser.add_argument("--augment-flip", action="store_true", help="Horizontally flip frames and invert steering to balance left/right turns.")
    parser.add_argument("--balanced-sampler", action="store_true", help="Balance idle, left, right, and straight samples during training.")
    parser.add_argument("--steer-loss-weight", type=float, default=2.0)
    parser.add_argument("--gas-loss-weight", type=float, default=1.0)
    parser.add_argument("--brake-loss-weight", type=float, default=1.0)
    return parser.parse_args()


def action_bucket(action: np.ndarray) -> str:
    if float(np.abs(action).sum()) < 0.02:
        return "idle"
    if float(action[0]) < -0.05:
        return "left"
    if float(action[0]) > 0.05:
        return "right"
    return "straight"


def make_balanced_sampler(dataset: DrivingDataset, subset_indices: list[int]) -> WeightedRandomSampler:
    buckets = [action_bucket(dataset.action_at(int(index))) for index in subset_indices]
    counts = {bucket: buckets.count(bucket) for bucket in sorted(set(buckets))}
    weights = [1.0 / counts[bucket] for bucket in buckets]
    print("train_bucket_counts=" + ", ".join(f"{bucket}={count}" for bucket, count in counts.items()))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for frames, actions in loader:
            frames = frames.to(device)
            actions = actions.to(device)
            pred = model(frames)
            loss = loss_fn(pred, actions)
            total_loss += float(loss.item()) * len(actions)
            total_items += len(actions)
    return total_loss / max(1, total_items)


def main() -> None:
    args = parse_args()
    for data_path in args.data:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DrivingDataset(
        args.data,
        stack_size=args.stack_size,
        drop_idle=args.drop_idle,
        gas_when_steering=args.gas_when_steering,
        augment_flip=args.augment_flip,
    )
    if len(dataset) < 10:
        raise ValueError("Need at least 10 samples. Collect more driving data first.")

    sample_actions = np.stack([dataset[i][1].numpy() for i in range(min(len(dataset), 2000))])
    print(f"samples_used={len(dataset)} of data={', '.join(str(p) for p in args.data)}")
    print(
        "sample_action_mean="
        f"[steer={sample_actions[:, 0].mean():.3f}, gas={sample_actions[:, 1].mean():.3f}, "
        f"brake={sample_actions[:, 2].mean():.3f}]"
    )

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_sampler = None
    train_shuffle = True
    if args.balanced_sampler:
        train_sampler = make_balanced_sampler(dataset, list(train_set.indices))
        train_shuffle = False

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=train_shuffle, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = DrivingCNN(in_channels=args.stack_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.HuberLoss(reduction="none")
    component_weights = torch.tensor(
        [args.steer_loss_weight, args.gas_loss_weight, args.brake_loss_weight],
        dtype=torch.float32,
        device=device,
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for frames, actions in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            frames = frames.to(device)
            actions = actions.to(device)

            pred = model(frames)
            loss = (loss_fn(pred, actions) * component_weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * len(actions)
            seen += len(actions)

        train_loss = running / max(1, seen)
        val_loss = evaluate(model, val_loader, nn.HuberLoss(), device)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "stack_size": args.stack_size,
                    "val_loss": best_val,
                },
                args.out,
            )
            print(f"saved {args.out}")


if __name__ == "__main__":
    main()
