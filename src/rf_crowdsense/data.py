from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from .preprocessing.spectrogram import iq_to_spectrogram

CACHE_VERSION = "spectrogram-cache-v1"


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def load_manifest(dataset: Path) -> list[dict]:
    manifest = dataset / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    with manifest.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_count_scale(dataset: Path, rows: list[dict] | None = None) -> float:
    """Return the aggregate-count scale used to convert normalized activity to counts."""
    summary_path = dataset / "dataset.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            max_devices = float(summary.get("config", {}).get("max_devices", 0))
            if max_devices > 0:
                return max_devices
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    rows = rows if rows is not None else load_manifest(dataset)
    observed = [float(row.get("active_devices", 0)) for row in rows]
    return max(max(observed, default=1.0), 1.0)


def manifest_sha256(dataset: Path) -> str:
    manifest = dataset / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    digest = sha256()
    with manifest.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_splits(
    sample_count: int,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> DatasetSplits:
    if sample_count < 3:
        raise ValueError("At least three samples are required for train/validation/test splits")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count, dtype=np.int64)
    rng.shuffle(indices)

    train_count = max(1, int(sample_count * train_ratio))
    validation_count = max(1, int(sample_count * validation_ratio))
    test_count = sample_count - train_count - validation_count

    if test_count < 1:
        deficit = 1 - test_count
        if train_count - deficit >= 1:
            train_count -= deficit
        elif validation_count - deficit >= 1:
            validation_count -= deficit
        else:
            raise ValueError("Unable to create non-empty train/validation/test splits")
        test_count = sample_count - train_count - validation_count

    train_end = train_count
    validation_end = train_end + validation_count
    return DatasetSplits(
        train=indices[:train_end],
        validation=indices[train_end:validation_end],
        test=indices[validation_end : validation_end + test_count],
    )


def load_example(dataset: Path, row: dict) -> tuple[np.ndarray, float, int, int]:
    payload = np.load(dataset / row["file"])
    iq = payload["iq"]
    sample_rate = float(payload["sample_rate"])
    spec = iq_to_spectrogram(iq, sample_rate)
    x = spec[..., None].astype(np.float32)
    score = float(payload["activity_score"])
    count = int(payload["active_devices"])
    cls = int(payload["activity_class"])
    return x, score, count, cls


def _cache_files_present(cache: Path) -> bool:
    required = (
        "cache.json",
        "spectrograms.npy",
        "scores.npy",
        "counts.npy",
        "classes.npy",
        "splits.npz",
    )
    return all((cache / name).exists() for name in required)


def load_cache_metadata(cache: Path) -> dict:
    metadata_path = cache / "cache.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cache metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_cached_splits(cache: Path) -> DatasetSplits:
    payload = np.load(cache / "splits.npz")
    return DatasetSplits(
        train=np.asarray(payload["train"], dtype=np.int64),
        validation=np.asarray(payload["validation"], dtype=np.int64),
        test=np.asarray(payload["test"], dtype=np.int64),
    )


def build_spectrogram_cache(
    dataset: Path,
    output: Path | None = None,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    nperseg: int = 256,
    overlap: float = 0.5,
    overwrite: bool = False,
) -> dict:
    dataset = dataset.resolve()
    output = (output or (dataset / "cache")).resolve()
    rows = load_manifest(dataset)
    if len(rows) < 3:
        raise ValueError("At least three dataset samples are required to build a cache")

    source_hash = manifest_sha256(dataset)
    desired = {
        "version": CACHE_VERSION,
        "source_manifest_sha256": source_hash,
        "samples": len(rows),
        "preprocessing": {"nperseg": nperseg, "overlap": overlap},
        "split": {
            "seed": seed,
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": 1.0 - train_ratio - validation_ratio,
        },
    }

    if _cache_files_present(output) and not overwrite:
        existing = load_cache_metadata(output)
        if all(existing.get(key) == value for key, value in desired.items()):
            return {**existing, "reused": True}
        raise FileExistsError(
            f"Cache already exists with different settings: {output}. Use --overwrite to rebuild it."
        )

    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in (
            "spectrograms.npy",
            "scores.npy",
            "counts.npy",
            "classes.npy",
            "splits.npz",
            "cache.json",
        ):
            path = output / name
            if path.exists():
                path.unlink()

    first_payload = np.load(dataset / rows[0]["file"])
    first_spec = iq_to_spectrogram(
        first_payload["iq"],
        float(first_payload["sample_rate"]),
        nperseg=nperseg,
        overlap=overlap,
    )
    spec_shape = first_spec.shape

    spectrograms = np.lib.format.open_memmap(
        output / "spectrograms.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), *spec_shape),
    )
    scores = np.lib.format.open_memmap(
        output / "scores.npy", mode="w+", dtype=np.float32, shape=(len(rows),)
    )
    counts = np.lib.format.open_memmap(
        output / "counts.npy", mode="w+", dtype=np.int32, shape=(len(rows),)
    )
    classes = np.lib.format.open_memmap(
        output / "classes.npy", mode="w+", dtype=np.int64, shape=(len(rows),)
    )

    for index, row in enumerate(rows):
        payload = np.load(dataset / row["file"])
        spec = iq_to_spectrogram(
            payload["iq"],
            float(payload["sample_rate"]),
            nperseg=nperseg,
            overlap=overlap,
        )
        if spec.shape != spec_shape:
            raise ValueError(
                f"Spectrogram shape changed at sample {index}: {spec.shape} != {spec_shape}. "
                "Use a dataset with a consistent IQ sample length."
            )
        spectrograms[index] = spec
        scores[index] = float(payload["activity_score"])
        counts[index] = int(payload["active_devices"])
        classes[index] = int(payload["activity_class"])

    spectrograms.flush()
    scores.flush()
    counts.flush()
    classes.flush()

    splits = make_splits(
        len(rows), seed=seed, train_ratio=train_ratio, validation_ratio=validation_ratio
    )
    np.savez(
        output / "splits.npz",
        train=splits.train,
        validation=splits.validation,
        test=splits.test,
    )

    metadata = {
        **desired,
        "source_dataset": str(dataset),
        "spectrogram_shape": list(spec_shape),
        "dtype": "float32",
        "split_counts": {
            "train": int(len(splits.train)),
            "validation": int(len(splits.validation)),
            "test": int(len(splits.test)),
        },
        "files": {
            "spectrograms": "spectrograms.npy",
            "scores": "scores.npy",
            "counts": "counts.npy",
            "classes": "classes.npy",
            "splits": "splits.npz",
        },
    }
    (output / "cache.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "reused": False}
