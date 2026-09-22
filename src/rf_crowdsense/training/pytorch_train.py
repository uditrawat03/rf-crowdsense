from __future__ import annotations

from pathlib import Path
import random

import numpy as np

from ..data import load_example, load_manifest
from ..models.pytorch_models import build_model


def train(
    dataset: Path,
    epochs: int,
    batch_size: int,
    output: Path,
    model_name: str = "cnn",
    seed: int = 42,
    amp: bool = True,
) -> Path:
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset

    rows = load_manifest(dataset)
    if len(rows) < 2:
        raise ValueError("At least two dataset samples are required for training")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    class RFDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, idx):
            x, score, _count, cls = load_example(dataset, rows[idx])
            x = np.transpose(x, (2, 0, 1))
            return (
                torch.from_numpy(x),
                torch.tensor([score], dtype=torch.float32),
                torch.tensor(cls, dtype=torch.long),
            )

    indices = np.arange(len(rows))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    split = min(split, len(indices) - 1)
    train_indices = indices[:split].tolist()
    val_indices = indices[split:].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(amp and device.type == "cuda")
    model = build_model(model_name).to(device)
    train_loader = DataLoader(
        Subset(RFDataset(), train_indices),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(RFDataset(), val_indices),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    score_loss_fn = torch.nn.MSELoss()
    class_loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for x, score, cls in train_loader:
            x = x.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)
            cls = cls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                pred_score, pred_class = model(x)
                score_loss = score_loss_fn(pred_score, score)
                class_loss = class_loss_fn(pred_class, cls)
                loss = score_loss + 0.25 * class_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * x.size(0)
            total_items += x.size(0)

        model.eval()
        val_loss = 0.0
        val_items = 0
        correct = 0
        with torch.inference_mode():
            for x, score, cls in val_loader:
                x = x.to(device, non_blocking=True)
                score = score.to(device, non_blocking=True)
                cls = cls.to(device, non_blocking=True)
                pred_score, pred_class = model(x)
                loss = score_loss_fn(pred_score, score) + 0.25 * class_loss_fn(pred_class, cls)
                val_loss += float(loss.detach().cpu()) * x.size(0)
                val_items += x.size(0)
                correct += int((pred_class.argmax(dim=1) == cls).sum().item())

        train_mean = total_loss / max(total_items, 1)
        val_mean = val_loss / max(val_items, 1)
        val_accuracy = correct / max(val_items, 1)
        print(
            f"epoch={epoch} train_loss={train_mean:.6f} val_loss={val_mean:.6f} "
            f"val_acc={val_accuracy:.3f} device={device} amp={use_amp} model={model_name}"
        )

        if val_mean < best_val:
            best_val = val_mean
            torch.save(
                {
                    "version": 2,
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val,
                    "activity_classes": ["0-5", "6-20", "21-50", "51-100", "101+"],
                },
                output,
            )

    return output
