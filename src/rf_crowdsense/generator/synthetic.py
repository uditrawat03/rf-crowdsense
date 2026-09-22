from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np


ACTIVITY_BINS = (5, 20, 50, 100)
ACTIVITY_LABELS = ("0-5", "6-20", "21-50", "51-100", "101+")


@dataclass(slots=True)
class GeneratorConfig:
    sample_rate: float = 1_000_000.0
    num_samples: int = 4096
    min_devices: int = 0
    max_devices: int = 120
    min_snr_db: float = -5.0
    max_snr_db: float = 25.0


def activity_class(active_devices: int) -> int:
    return int(np.digitize(active_devices, ACTIVITY_BINS, right=True))


def _qpsk(rng: np.random.Generator, n: int) -> np.ndarray:
    symbols = rng.integers(0, 4, size=n)
    constellation = np.asarray(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64
    )
    return constellation[symbols] / np.sqrt(2.0)


def _burst_envelope(rng: np.random.Generator, n: int) -> np.ndarray:
    envelope = np.zeros(n, dtype=np.float32)
    cursor = 0
    while cursor < n:
        burst = int(rng.integers(max(16, n // 128), max(32, n // 16)))
        gap = int(rng.integers(0, max(1, n // 64)))
        if rng.random() < 0.78:
            envelope[cursor : min(n, cursor + burst)] = 1.0
        cursor += burst + gap
    return envelope


def generate_iq(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    device_count: int,
) -> tuple[np.ndarray, dict]:
    n = cfg.num_samples
    t = np.arange(n, dtype=np.float32) / cfg.sample_rate
    signal = np.zeros(n, dtype=np.complex64)

    active_devices = 0
    occupied_samples = np.zeros(n, dtype=bool)
    for _ in range(device_count):
        if rng.random() > rng.uniform(0.55, 0.98):
            continue
        envelope = _burst_envelope(rng, n)
        if not envelope.any():
            continue
        active_devices += 1
        occupied_samples |= envelope.astype(bool)
        symbols = _qpsk(rng, n)
        freq = rng.uniform(-0.42, 0.42) * cfg.sample_rate
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = rng.uniform(0.08, 0.75)
        fading = rng.rayleigh(scale=0.8)
        carrier = np.exp(1j * (2.0 * np.pi * freq * t + phase)).astype(np.complex64)
        signal += (amplitude * fading * envelope * symbols * carrier).astype(np.complex64)

    signal_power = float(np.mean(np.abs(signal) ** 2))
    snr_db = float(rng.uniform(cfg.min_snr_db, cfg.max_snr_db))
    noise_power = 1.0 if signal_power == 0.0 else signal_power / (10.0 ** (snr_db / 10.0))
    noise_sigma = float(np.sqrt(noise_power / 2.0))
    noise = (
        rng.normal(0.0, noise_sigma, n) + 1j * rng.normal(0.0, noise_sigma, n)
    ).astype(np.complex64)
    iq = signal + noise

    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    if rms > 0:
        iq = iq / rms

    cls = activity_class(active_devices)
    metadata = {
        "requested_devices": int(device_count),
        "active_devices": int(active_devices),
        "activity_class": cls,
        "activity_label": ACTIVITY_LABELS[cls],
        "snr_db": snr_db,
        "sample_rate": cfg.sample_rate,
        "num_samples": n,
        "time_occupancy": float(np.mean(occupied_samples)),
        "activity_score": float(active_devices / max(cfg.max_devices, 1)),
    }
    return iq.astype(np.complex64), metadata


def generate_dataset(
    output: Path,
    samples: int,
    seed: int = 42,
    cfg: GeneratorConfig | None = None,
) -> dict:
    cfg = cfg or GeneratorConfig()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest_path = output / "manifest.jsonl"

    class_counts = [0] * len(ACTIVITY_LABELS)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in range(samples):
            device_count = int(rng.integers(cfg.min_devices, cfg.max_devices + 1))
            iq, metadata = generate_iq(rng, cfg, device_count)
            class_counts[int(metadata["activity_class"])] += 1
            filename = f"sample_{index:06d}.npz"
            np.savez_compressed(output / filename, iq=iq, **metadata)
            manifest.write(json.dumps({"file": filename, **metadata}) + "\n")

    summary = {
        "version": "synthetic-v0.2",
        "output": str(output),
        "samples": samples,
        "seed": seed,
        "config": asdict(cfg),
        "classes": dict(zip(ACTIVITY_LABELS, class_counts, strict=True)),
        "manifest": str(manifest_path),
    }
    (output / "dataset.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
