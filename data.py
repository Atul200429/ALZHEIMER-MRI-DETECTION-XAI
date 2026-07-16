"""
Data loading and preprocessing for the Alzheimer's MRI classification task.

This replaces the Google-Colab-only cells in the original notebook
(``drive.mount(...)``) with a plain local-filesystem loader, so the project
can be cloned and run anywhere -- not just inside Colab.
"""

import os
import random

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from src import config


def load_category_images(data_dir: str, categories: list[str]) -> list[tuple[np.ndarray, int]]:
    """Load every image under ``data_dir/<category>/`` and label it.

    Returns a list of (image_array, class_index) tuples. Skips files that
    fail to load instead of crashing the whole run, and logs how many were
    skipped -- the original notebook silently assumed every file was a
    readable image.
    """
    samples = []
    skipped = 0
    for category in categories:
        path = os.path.join(data_dir, category)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Expected a folder at '{path}'. See README for the expected "
                f"data layout."
            )
        class_index = categories.index(category)
        for filename in tqdm(os.listdir(path), desc=f"Loading {category}"):
            img_path = os.path.join(path, filename)
            img_array = cv2.imread(img_path)
            if img_array is None:
                skipped += 1
                continue
            samples.append((img_array, class_index))

    if skipped:
        print(f"Warning: skipped {skipped} unreadable file(s).")
    return samples


def build_dataset(data_dir: str = config.DATA_DIR, seed: int = config.RANDOM_SEED):
    """Load, shuffle, normalize, and one-hot encode the full dataset.

    Returns (X, Y) where X is a float32 array of normalized images and Y is
    one-hot encoded labels.
    """
    samples = load_category_images(data_dir, config.CATEGORIES)

    random.seed(seed)
    random.shuffle(samples)

    X = np.array([s[0] for s in samples], dtype="float32") / 255.0
    y = np.array([s[1] for s in samples])
    Y = to_categorical(y, num_classes=config.NUM_CLASSES)

    return X, Y
