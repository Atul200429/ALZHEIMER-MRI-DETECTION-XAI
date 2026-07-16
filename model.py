"""
CNN architecture for Alzheimer's MRI classification.

Kept intentionally close to the original notebook's architecture so results
stay comparable -- refactored out into a function so it's testable and
reusable from train.py, evaluate.py, and explain.py instead of copy-pasted.
"""

import tensorflow as tf

from src import config


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", activation="relu",
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
        ),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",  # matches one-hot Y from data.py
        metrics=["accuracy"],
    )
    return model


def get_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            return layer.name
    raise ValueError("No convolutional layer found in model.")
