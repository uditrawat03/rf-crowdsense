import numpy as np
import pytest

from rf_crowdsense.preprocessing.spectrogram import iq_to_spectrogram


def test_spectrogram_shape_and_finite_values():
    rng = np.random.default_rng(42)
    iq = (rng.normal(size=1024) + 1j * rng.normal(size=1024)).astype(np.complex64)
    spec = iq_to_spectrogram(iq, 1_000_000.0, nperseg=128)
    assert spec.shape[0] == 128
    assert spec.shape[1] > 1
    assert spec.dtype == np.float32
    assert np.isfinite(spec).all()
    assert abs(float(spec.mean())) < 1e-4


def test_spectrogram_rejects_short_input():
    with pytest.raises(ValueError):
        iq_to_spectrogram(np.ones(16, dtype=np.complex64), 1_000_000.0, nperseg=256)
