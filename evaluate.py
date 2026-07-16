"""
Evaluate a trained model: classification report + confusion matrix.

Usage:
    python -m src.evaluate --data-dir data --model outputs/model/ad_xai_model
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src import config
from src.data import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--output", default=config.METRICS_PATH)
    parser.add_argument("--val-split", type=float, default=config.VALIDATION_SPLIT)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    X, Y = build_dataset(data_dir=args.data_dir, seed=args.seed)
    _, X_test, _, y_test = train_test_split(
        X, Y, test_size=args.val_split, random_state=args.seed
    )

    model = tf.keras.models.load_model(args.model)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred, target_names=config.CATEGORIES)
    print(report)
    with open(os.path.join(args.output, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CATEGORIES, yticklabels=config.CATEGORIES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(args.output, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
