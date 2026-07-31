from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path

import h5py
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa_driver.game_model import GameDrivingCNN


DEFAULT_LABELS = ("w_forward", "a_left", "d_right", "s_reverse", "space_drift")


class GameplayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        paths: list[Path],
        stack_size: int = 4,
        label_indices: tuple[int, ...] = (0, 1, 2, 3, 4),
        width: int = 112,
        height: int = 48,
    ) -> None:
        self.paths = paths
        self.stack_size = stack_size
        self.label_indices = label_indices
        self.width = width
        self.height = height
        self.frames, self.labels, self.lengths, self.all_label_names = self._load_files()
        self.cumulative = np.cumsum(self.lengths).tolist()
        self.starts = [0] + self.cumulative[:-1]
        self.label_names = [self.all_label_names[index] for index in self.label_indices]

    def _load_files(self) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
        frame_parts = []
        label_parts = []
        lengths = []
        all_label_names = list(DEFAULT_LABELS)
        for path in self.paths:
            with h5py.File(path, "r") as h5:
                raw_names = str(h5.attrs.get("key_labels", ",".join(DEFAULT_LABELS))).split(",")
                all_label_names = raw_names
                frames = h5["frames"][:]
                if frames.shape[1] != self.height or frames.shape[2] != self.width:
                    frames = np.asarray(
                        [cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA) for frame in frames],
                        dtype=np.uint8,
                    )
                labels = h5["key_labels"][:, self.label_indices].astype(np.float32)
                frame_parts.append(frames)
                label_parts.append(labels)
                lengths.append(len(frames))
                print(f"loaded {path} frames={len(frames)}")
        return np.concatenate(frame_parts, axis=0), np.concatenate(label_parts, axis=0), lengths, all_label_names

    def close(self) -> None:
        return None

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def locate(self, index: int) -> tuple[int, int]:
        file_index = bisect.bisect_right(self.cumulative, index)
        previous = 0 if file_index == 0 else self.cumulative[file_index - 1]
        return file_index, index - previous

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_index, frame_index = self.locate(index)
        file_start = self.starts[file_index]
        global_index = file_start + frame_index

        frames = []
        for offset in range(self.stack_size - 1, -1, -1):
            source_index = max(file_start, global_index - offset)
            frames.append(self.frames[source_index])

        stacked = np.concatenate(frames, axis=2)
        stacked = stacked.astype(np.float32) / 255.0
        stacked = np.transpose(stacked, (2, 0, 1))

        labels = self.labels[global_index].astype(np.float32)
        return torch.from_numpy(stacked), torch.from_numpy(labels)

    def label_counts(self) -> tuple[np.ndarray, int]:
        return self.labels.sum(axis=0).astype(np.float64), len(self.labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-label keyboard policy from real gameplay captures.")
    parser.add_argument("--data", type=Path, nargs="+", default=sorted(Path("data").glob("party_kart*.h5")))
    parser.add_argument("--exclude-data", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=Path("runs/party_kart_bc.pt"))
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stack-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--labels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--no-pos-weight", action="store_true")
    return parser.parse_args()


def resolve_data_paths(patterns: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in patterns:
        if any(char in str(path) for char in "*?[]"):
            paths.extend(sorted(path.parent.glob(path.name)))
        elif path.exists():
            paths.append(path)
    return paths


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = None
    positives = None
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            logits = model(frames)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            total_items += len(labels)

            pred = (torch.sigmoid(logits) >= 0.5).float()
            batch_correct = (pred == labels).sum(dim=0).cpu().numpy()
            batch_pos = pred.sum(dim=0).cpu().numpy()
            correct = batch_correct if correct is None else correct + batch_correct
            positives = batch_pos if positives is None else positives + batch_pos

    return total_loss / max(1, total_items), correct / max(1, total_items), positives / max(1, total_items)


def main() -> None:
    args = parse_args()
    data_paths = resolve_data_paths(args.data)
    excluded_paths = set(resolve_data_paths(args.exclude_data))
    if excluded_paths:
        before = len(data_paths)
        data_paths = [path for path in data_paths if path not in excluded_paths]
        print(f"excluded_files={before - len(data_paths)}")
    if not data_paths:
        raise FileNotFoundError("No gameplay HDF5 files found.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset = GameplayDataset(
        data_paths,
        stack_size=args.stack_size,
        label_indices=tuple(args.labels),
        width=args.width,
        height=args.height,
    )
    counts, total = dataset.label_counts()
    rates = counts / max(1, total)
    label_names = dataset.label_names

    print(f"files={len(data_paths)} samples={len(dataset)}")
    for name, count, rate in zip(label_names, counts.astype(int), rates):
        print(f"{name}: count={count} rate={rate:.3f}")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GameDrivingCNN(in_channels=args.stack_size * 3, action_dim=len(args.labels)).to(device)
    if args.init_model is not None:
        checkpoint = torch.load(args.init_model, map_location=device)
        checkpoint_labels = list(checkpoint.get("label_names", []))
        if checkpoint_labels and checkpoint_labels != label_names:
            raise ValueError(f"init model labels {checkpoint_labels} do not match training labels {label_names}")
        model.load_state_dict(checkpoint["model"])
        print(f"initialized_from={args.init_model}")

    if args.no_pos_weight:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        negatives = np.maximum(total - counts, 1)
        positives = np.maximum(counts, 1)
        pos_weight = np.clip(negatives / positives, 0.25, args.pos_weight_cap).astype(np.float32)
        print("pos_weight=" + ", ".join(f"{name}:{weight:.2f}" for name, weight in zip(label_names, pos_weight)))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            for frames, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
                frames = frames.to(device)
                labels = labels.to(device)
                logits = model(frames)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += float(loss.item()) * len(labels)
                seen += len(labels)

            train_loss = running / max(1, seen)
            val_loss, val_acc, pred_rates = evaluate(model, val_loader, loss_fn, device)
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            print("val_acc=" + ", ".join(f"{name}:{acc:.3f}" for name, acc in zip(label_names, val_acc)))
            print("pred_rates=" + ", ".join(f"{name}:{rate:.3f}" for name, rate in zip(label_names, pred_rates)))

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "stack_size": args.stack_size,
                        "label_indices": list(args.labels),
                        "label_names": label_names,
                        "input_size": [args.height, args.width],
                        "val_loss": best_val,
                    },
                    args.out,
                )
                print(f"saved {args.out}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
