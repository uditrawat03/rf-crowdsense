from __future__ import annotations


def build_model(input_shape: tuple[int, int, int], num_classes: int = 5):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape, name="spectrogram")
    x = tf.keras.layers.Conv2D(24, 5, padding="same", activation="gelu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(48, 3, padding="same", activation="gelu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(96, 3, padding="same", activation="gelu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    score = tf.keras.layers.Dense(1, activation="sigmoid", name="activity_score")(x)
    classes = tf.keras.layers.Dense(num_classes, name="activity_class")(x)
    model = tf.keras.Model(inputs, {"activity_score": score, "activity_class": classes})
    model.compile(
        optimizer="adam",
        loss={
            "activity_score": "mse",
            "activity_class": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        },
        loss_weights={"activity_score": 1.0, "activity_class": 0.25},
        metrics={
            "activity_score": ["mae"],
            "activity_class": ["accuracy"],
        },
    )
    return model
