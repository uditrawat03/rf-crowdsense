from __future__ import annotations

import numpy as np


def iq_to_spectrogram(
    iq: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    overlap: float = 0.5,
) -> np.ndarray:
    """Convert complex IQ samples to a normalized log-power spectrogram using NumPy only."""
    del sample_rate  # Reserved for future frequency-axis metadata.
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.ndim != 1:
        raise ValueError("iq must be a one-dimensional complex array")
    if nperseg < 8 or len(iq) < nperseg:
        raise ValueError("nperseg must be >= 8 and <= the IQ sample count")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    hop = max(1, int(nperseg * (1.0 - overlap)))
    frame_count = 1 + (len(iq) - nperseg) // hop
    starts = np.arange(frame_count) * hop
    frames = np.stack([iq[start : start + nperseg] for start in starts], axis=0)
    window = np.hanning(nperseg).astype(np.float32)
    spectrum = np.fft.fft(frames * window[None, :], axis=1)
    spectrum = np.fft.fftshift(spectrum, axes=1)
    power = np.abs(spectrum) ** 2
    log_power = 10.0 * np.log10(power + 1e-8)
    log_power = log_power.T

    mean = float(log_power.mean())
    std = float(log_power.std()) + 1e-6
    return ((log_power - mean) / std).astype(np.float32)
