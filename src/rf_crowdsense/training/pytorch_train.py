from __future__ import annotations

from pathlib import Path
import random

import numpy as np

from ..calibration import (
    count_interval,
    empirical_interval_coverage,
    fit_count_interval,
)
from ..data import (
    load_cache_metadata,
    load_cached_splits,
    load_example,
    load_manifest,
    make_splits,
    resolve_count_scale,
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
    count_coverage: float = 0.90,
) -> Path:
    import torch
    from torch.utils.data import DataLoader, Dataset

    if not 0.0 < count_coverage < 1.0:
        raise ValueError("count_coverage must be between 0 and 1")

    dataset = dataset.resolve()
    rows = load_manifest(dataset)
    if len(rows) < 3:
        raise ValueError("At least three dataset samples are required for training")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    count_scale = resolve_count_scale(dataset, rows)
    cache_path = _resolve_cache(dataset, cache)
    preprocessing = {"nperseg": 256, "overlap": 0.5}

    if cache_path is not None:
        metadata = load_cache_metadata(cache_path)
        if int(metadata["samples"]) != len(rows):
            raise ValueError(
                f"Cache sample count ({metadata['samples']}) does not match dataset ({len(rows)}). "
                "Rebuild the cache."
            )
        preprocessing = dict(metadata.get("preprocessing", preprocessing))
        specs = np.load(cache_path / "spectrograms.npy", mmap_mode="r")
        scores = np.load(cache_path / "scores.npy", mmap_mode="r")
        counts = np.load(cache_path / "counts.npy", mmap_mode="r")
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
                    torch.tensor(float(counts[sample_index]), dtype=torch.float32),
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
                x, score, count, cls = load_example(dataset, rows[sample_index])
                x = np.transpose(x, (2, 0, 1))
                return (
                    torch.from_numpy(x),
                    torch.tensor([score], dtype=torch.float32),
                    torch.tensor(float(count), dtype=torch.float32),
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

    def evaluate(loader, *, collect_counts: bool = False):
        model.eval()
        total_loss = 0.0
        total_items = 0
        correct = 0
        count_absolute_error = 0.0
        true_counts: list[np.ndarray] = []
        predicted_counts: list[np.ndarray] = []

        with torch.inference_mode():
            for x, score, count, cls in loader:
                x = x.to(device, non_blocking=True)
                score = score.to(device, non_blocking=True)
                count = count.to(device, non_blocking=True)
                cls = cls.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    pred_score, pred_class = model(x)
                    loss = score_loss_fn(pred_score, score) + 0.25 * class_loss_fn(pred_class, cls)

                pred_count = pred_score.squeeze(1).float() * count_scale
                batch_size_actual = x.size(0)
                total_loss += float(loss.detach().cpu()) * batch_size_actual
                total_items += batch_size_actual
                correct += int((pred_class.argmax(dim=1) == cls).sum().item())
                count_absolute_error += float(torch.abs(pred_count - count).sum().detach().cpu())

                if collect_counts:
                    true_counts.append(count.detach().cpu().numpy().astype(np.float64))
                    predicted_counts.append(pred_count.detach().cpu().numpy().astype(np.float64))

        metrics = {
            "loss": total_loss / max(total_items, 1),
            "accuracy": correct / max(total_items, 1),
            "count_mae": count_absolute_error / max(total_items, 1),
        }
        if not collect_counts:
            return metrics, None, None
        return (
            metrics,
            np.concatenate(true_counts) if true_counts else np.empty(0, dtype=np.float64),
            np.concatenate(predicted_counts) if predicted_counts else np.empty(0, dtype=np.float64),
        )

    best_val = float("inf")
    best_val_accuracy = 0.0
    best_val_count_mae = float("inf")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"data={data_mode} train={len(train_dataset)} validation={len(val_dataset)} "
        f"test={len(test_dataset)} device={device} amp={use_amp} model={model_name} "
        f"count_scale={count_scale:.1f}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for x, score, _count, cls in train_loader:
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
        val_metrics, _, _ = evaluate(val_loader)
        print(
            f"epoch={epoch} train_loss={train_mean:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_count_mae={val_metrics['count_mae']:.2f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_val_accuracy = val_metrics["accuracy"]
            best_val_count_mae = val_metrics["count_mae"]
            torch.save(
                {
                    "version": 4,
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val,
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_count_mae": best_val_count_mae,
                    "activity_classes": ["0-5", "6-20", "21-50", "51-100", "101+"],
                    "seed": seed,
                    "data_mode": data_mode,
                    "preprocessing": preprocessing,
                    "count_scale": count_scale,
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

    val_metrics, val_true_counts, val_predicted_counts = evaluate(val_loader, collect_counts=True)
    calibration = fit_count_interval(
        val_true_counts,
        val_predicted_counts,
        coverage=count_coverage,
    )

    test_metrics, test_true_counts, test_predicted_counts = evaluate(test_loader, collect_counts=True)
    test_interval_coverage = empirical_interval_coverage(
        test_true_counts,
        test_predicted_counts,
        calibration,
    )
    widths = []
    for prediction in test_predicted_counts:
        lower, upper = count_interval(
            float(prediction),
            calibration,
            lower_bound=0.0,
            upper_bound=count_scale,
        )
        widths.append(upper - lower)

    checkpoint["count_calibration"] = calibration.to_dict()
    checkpoint["validation_count_mae"] = val_metrics["count_mae"]
    checkpoint["test_loss"] = test_metrics["loss"]
    checkpoint["test_accuracy"] = test_metrics["accuracy"]
    checkpoint["test_count_mae"] = test_metrics["count_mae"]
    checkpoint["test_count_interval_coverage"] = test_interval_coverage
    checkpoint["test_count_interval_mean_width"] = float(np.mean(widths)) if widths else 0.0
    torch.save(checkpoint, output)

    print(
        f"test_loss={test_metrics['loss']:.6f} test_acc={test_metrics['accuracy']:.3f} "
        f"test_count_mae={test_metrics['count_mae']:.2f} "
        f"count_interval_target={count_coverage:.3f} "
        f"count_interval_coverage={test_interval_coverage:.3f} "
        f"count_interval_radius={calibration.radius:.2f}"
    )

    return output
