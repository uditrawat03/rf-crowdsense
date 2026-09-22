from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .preprocessing.spectrogram import iq_to_spectrogram


def load_manifest(dataset: Path) -> list[dict]:
    manifest = dataset / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    with manifest.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
