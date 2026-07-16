"""
Train the Alzheimer's MRI classifier and save the model + metrics.

Usage:
    python -m src.train --data-dir data --epochs 10 --output outputs
"""

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.data import build_dataset
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Alzheimer's XAI model.")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--output", default=config.OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--val-split", type=float, default=config.VALIDATION_SPLIT)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()

    X, Y = build_dataset(data_dir=args.data_dir, seed=args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=args.val_split, random_state=args.seed
    )

    model = build_model()
    model.summary()

    # The OASIS dataset is heavily imbalanced (NonDemented has ~140x more
    # images than ModerateDemented), so weight classes inversely to their
    # frequency -- otherwise the model can get a deceptively high accuracy
    # by mostly predicting the majority class.
    y_train_indices = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(config.NUM_CLASSES),
        y=y_train_indices,
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Using class weights: {class_weight_dict}")

    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
    )

    model_path = os.path.join(args.output, "model", "ad_xai_model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")

    score = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {score[0]:.4f}  |  Test accuracy: {score[1]:.4f}")


if __name__ == "__main__":
    main()
