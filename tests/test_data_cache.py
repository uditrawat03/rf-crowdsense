from pathlib import Path

import numpy as np
import pytest

from rf_crowdsense.data import (
    build_spectrogram_cache,
    load_cached_splits,
    make_splits,
)
from rf_crowdsense.generator.synthetic import GeneratorConfig, generate_dataset


def test_make_splits_is_deterministic_and_complete():
    a = make_splits(100, seed=7)
    b = make_splits(100, seed=7)

    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.validation, b.validation)
    assert np.array_equal(a.test, b.test)

    combined = np.concatenate([a.train, a.validation, a.test])
    assert len(combined) == 100
    assert len(np.unique(combined)) == 100
    assert set(combined.tolist()) == set(range(100))


def test_make_splits_rejects_too_few_samples():
    with pytest.raises(ValueError):
        make_splits(2)


def test_build_spectrogram_cache_and_reuse(tmp_path: Path):
    dataset = tmp_path / "dataset"
    cfg = GeneratorConfig(num_samples=512, max_devices=20)
    generate_dataset(dataset, samples=12, seed=11, cfg=cfg)

    cache = dataset / "cache"
    summary = build_spectrogram_cache(dataset, cache, seed=99, nperseg=64)

    assert summary["reused"] is False
    assert summary["samples"] == 12
    assert summary["split_counts"] == {"train": 9, "validation": 1, "test": 2}

    specs = np.load(cache / "spectrograms.npy", mmap_mode="r")
    scores = np.load(cache / "scores.npy", mmap_mode="r")
    classes = np.load(cache / "classes.npy", mmap_mode="r")
    splits = load_cached_splits(cache)

    assert specs.shape[0] == 12
    assert specs.dtype == np.float32
    assert scores.shape == (12,)
    assert classes.shape == (12,)
    assert len(splits.train) + len(splits.validation) + len(splits.test) == 12

    reused = build_spectrogram_cache(dataset, cache, seed=99, nperseg=64)
    assert reused["reused"] is True


def test_cache_requires_overwrite_when_settings_change(tmp_path: Path):
    dataset = tmp_path / "dataset"
    cfg = GeneratorConfig(num_samples=512, max_devices=20)
    generate_dataset(dataset, samples=6, seed=5, cfg=cfg)
    cache = dataset / "cache"

    build_spectrogram_cache(dataset, cache, nperseg=64)
    with pytest.raises(FileExistsError):
        build_spectrogram_cache(dataset, cache, nperseg=128)
