from __future__ import annotations

from pathlib import Path
import random

import numpy as np

from ..data import (
    load_cache_metadata,
    load_cached_splits,
    load_example,
    load_manifest,
    make_splits,
)
from ..models.pytorch_models import build_model


def _resolve_cache(dataset: Path, cache: Path | None) -> Path | None:
    if cache is not None:
        cache = cache.resolve()
        load_cache_metadata(cache)
        return cache
    default_cache = dataset / "cache"
    if (default_cache / "cache.json").exists():
        return default_cache.resolve()
    return None


def train(
    dataset: Path,
    epochs: int,
    batch_size: int,
    output: Path,
    model_name: str = "cnn",
    seed: int = 42,
    amp: bool = True,
    cache: Path | None = None,
) -> Path:
    import torch
    from torch.utils.data import DataLoader, Dataset

    dataset = dataset.resolve()
    rows = load_manifest(dataset)
    if len(rows) < 3:
        raise ValueError("At least three dataset samples are required for training")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cache_path = _resolve_cache(dataset, cache)

    if cache_path is not None:
        metadata = load_cache_metadata(cache_path)
        if int(metadata["samples"]) != len(rows):
            raise ValueError(
                f"Cache sample count ({metadata['samples']}) does not match dataset ({len(rows)}). "
                "Rebuild the cache."
            )
        specs = np.load(cache_path / "spectrograms.npy", mmap_mode="r")
        scores = np.load(cache_path / "scores.npy", mmap_mode="r")
        classes = np.load(cache_path / "classes.npy", mmap_mode="r")
        splits = load_cached_splits(cache_path)

        class CachedRFDataset(Dataset):
            def __init__(self, indices: np.ndarray) -> None:
                self.indices = np.asarray(indices, dtype=np.int64)

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                sample_index = int(self.indices[idx])
                # torch.tensor copies the read-only memmap slice into writable tensor storage.
                x = torch.tensor(specs[sample_index][None, ...], dtype=torch.float32)
                return (
                    x,
                    torch.tensor([float(scores[sample_index])], dtype=torch.float32),
                    torch.tensor(int(classes[sample_index]), dtype=torch.long),
                )

        train_dataset = CachedRFDataset(splits.train)
        val_dataset = CachedRFDataset(splits.validation)
        test_dataset = CachedRFDataset(splits.test)
        data_mode = f"cache:{cache_path}"
    else:
        splits = make_splits(len(rows), seed=seed)

        class RawRFDataset(Dataset):
            def __init__(self, indices: np.ndarray) -> None:
                self.indices = np.asarray(indices, dtype=np.int64)

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                sample_index = int(self.indices[idx])
                x, score, _count, cls = load_example(dataset, rows[sample_index])
                x = np.transpose(x, (2, 0, 1))
                return (
                    torch.from_numpy(x),
                    torch.tensor([score], dtype=torch.float32),
                    torch.tensor(cls, dtype=torch.long),
                )

        train_dataset = RawRFDataset(splits.train)
        val_dataset = RawRFDataset(splits.validation)
        test_dataset = RawRFDataset(splits.test)
        data_mode = "raw-iq"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(amp and device.type == "cuda")
    model = build_model(model_name).to(device)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": device.type == "cuda",
        "num_workers": 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    score_loss_fn = torch.nn.MSELoss()
    class_loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def evaluate(loader) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_items = 0
        correct = 0
        with torch.inference_mode():
            for x, score, cls in loader:
                x = x.to(device, non_blocking=True)
                score = score.to(device, non_blocking=True)
                cls = cls.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    pred_score, pred_class = model(x)
                    loss = score_loss_fn(pred_score, score) + 0.25 * class_loss_fn(pred_class, cls)
                total_loss += float(loss.detach().cpu()) * x.size(0)
                total_items += x.size(0)
                correct += int((pred_class.argmax(dim=1) == cls).sum().item())
        return total_loss / max(total_items, 1), correct / max(total_items, 1)

    best_val = float("inf")
    best_val_accuracy = 0.0
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"data={data_mode} train={len(train_dataset)} validation={len(val_dataset)} "
        f"test={len(test_dataset)} device={device} amp={use_amp} model={model_name}"
    )

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

        train_mean = total_loss / max(total_items, 1)
        val_mean, val_accuracy = evaluate(val_loader)
        print(
            f"epoch={epoch} train_loss={train_mean:.6f} val_loss={val_mean:.6f} "
            f"val_acc={val_accuracy:.3f}"
        )

        if val_mean < best_val:
            best_val = val_mean
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "version": 3,
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val,
                    "best_val_accuracy": best_val_accuracy,
                    "activity_classes": ["0-5", "6-20", "21-50", "51-100", "101+"],
                    "seed": seed,
                    "data_mode": data_mode,
                    "split_counts": {
                        "train": len(train_dataset),
                        "validation": len(val_dataset),
                        "test": len(test_dataset),
                    },
                },
                output,
            )

    checkpoint = torch.load(output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_accuracy = evaluate(test_loader)
    checkpoint["test_loss"] = test_loss
    checkpoint["test_accuracy"] = test_accuracy
    torch.save(checkpoint, output)
    print(f"test_loss={test_loss:.6f} test_acc={test_accuracy:.3f}")

    return output
