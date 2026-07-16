"""
Generate LIME and Grad-CAM explanations for a single MRI image.

Replaces the notebook's interactive `input()` prompt with a CLI argument so
it can be run non-interactively (e.g. from a script, CI, or a small demo
app) instead of only inside a live Jupyter session.

Usage:
    python -m src.explain --image path/to/scan.jpg --model outputs/model/ad_xai_model
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries

from src import config
from src.model import get_last_conv_layer


def load_and_preprocess(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'.")
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def grad_cam(model, image, class_index, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    guided_grads = (
        tf.cast(output > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    )
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
    return np.maximum(cam, 0)


def generate_lime_explanation(model, image):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image[0].astype("double"), model.predict,
        top_labels=1, hide_color=0, num_samples=1000,
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    return mark_boundaries(temp / 2 + 0.5, mask)


def parse_args():
    parser = argparse.ArgumentParser(description="Explain one prediction with LIME + Grad-CAM.")
    parser.add_argument("--image", required=True, help="Path to a single MRI scan image.")
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--output", default=config.EXPLANATIONS_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    model = tf.keras.models.load_model(args.model)
    image = load_and_preprocess(args.image)

    prediction = model.predict(image)
    class_index = int(np.argmax(prediction))
    predicted_class = config.CATEGORIES[class_index]
    print(f"Predicted class: {predicted_class} (confidence {prediction[0][class_index]:.2%})")

    lime_explanation = generate_lime_explanation(model, image)
    last_conv_layer = get_last_conv_layer(model)
    grad_cam_result = grad_cam(model, image, class_index, last_conv_layer)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image[0])
    ax[0].set_title("Original Image")

    ax[1].imshow(lime_explanation)
    ax[1].set_title(f"LIME Explanation: {predicted_class}")

    ax[2].imshow(image[0])
    ax[2].imshow(
        cv2.resize(grad_cam_result, (image.shape[2], image.shape[1])),
        alpha=0.5, cmap="jet",
    )
    ax[2].set_title(f"Grad-CAM: {predicted_class}")

    for a in ax:
        a.axis("off")
    plt.tight_layout()

    out_path = os.path.join(args.output, f"explanation_{os.path.basename(args.image)}.png")
    plt.savefig(out_path)
    print(f"Saved explanation figure to {out_path}")


if __name__ == "__main__":
    main()
