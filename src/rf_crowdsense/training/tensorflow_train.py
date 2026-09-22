from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data import load_example, load_manifest
from ..models.tensorflow_model import build_model


def train(dataset: Path, epochs: int, batch_size: int, output: Path) -> Path:
    import tensorflow as tf

    rows = load_manifest(dataset)
    examples = [load_example(dataset, row) for row in rows]
    x = np.stack([item[0] for item in examples], axis=0)
    scores = np.asarray([item[1] for item in examples], dtype=np.float32)
    classes = np.asarray([item[3] for item in examples], dtype=np.int32)

    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model = build_model(tuple(x.shape[1:]))
    model.fit(
        x,
        {"activity_score": scores, "activity_class": classes},
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    return output
