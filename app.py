"""
Streamlit demo app for the Alzheimer's MRI classifier.

Lets anyone upload a brain MRI scan and see:
  - the predicted stage (with confidence)
  - a LIME explanation (which regions drove the decision)
  - a Grad-CAM heatmap (where the CNN "looked")

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud (share.streamlit.io):
    1. Push this repo to GitHub.
    2. Go to share.streamlit.io -> New app -> point it at this repo, main
       file path "app.py".
    3. Make sure a trained model is reachable -- see MODEL SETUP below.
"""

import os
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries

from src import config
from src.model import get_last_conv_layer

# --- MODEL SETUP -----------------------------------------------------------
# The trained model isn't committed to git (see .gitignore). For a live demo
# you have two options:
#   1. Commit a trained model under outputs/model/ad_xai_model anyway (fine
#      if it's a small CNN like this one, usually well under GitHub's 100MB
#      cap) and remove that path from .gitignore.
#   2. Host the model elsewhere (Hugging Face Hub, a GitHub Release asset,
#      Google Drive) and set the MODEL_URL secret/environment variable to a
#      direct-download link -- the app will fetch it on first run.
MODEL_PATH = config.MODEL_PATH
MODEL_URL = os.environ.get("MODEL_URL") or st.secrets.get("MODEL_URL", None) if hasattr(st, "secrets") else os.environ.get("MODEL_URL")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        if MODEL_URL:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        else:
            st.error(
                f"No model found at `{MODEL_PATH}` and no `MODEL_URL` is set. "
                "Train a model with `python -m src.train`, or set MODEL_URL "
                "in Streamlit secrets to a direct-download link for a "
                "trained model."
            )
            st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (config.IMG_WIDTH, config.IMG_HEIGHT))
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


@st.cache_data(show_spinner="Generating LIME explanation (this takes a moment)...")
def generate_lime_explanation(_model, image):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image[0].astype("double"), _model.predict,
        top_labels=1, hide_color=0, num_samples=1000,
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    return mark_boundaries(temp / 2 + 0.5, mask)


# --- UI ----------------------------------------------------------------
st.set_page_config(page_title="Alzheimer's MRI Detection + XAI", page_icon="🧠", layout="wide")

st.title("🧠 Alzheimer's Detection from MRI, with Explainable AI")
st.caption(
    "CNN classifier trained on the OASIS Alzheimer's Detection dataset. "
    "Upload a brain MRI scan to see the predicted stage, plus LIME and "
    "Grad-CAM explanations of *why* the model made that call."
)

with st.expander("⚠️ Disclaimer"):
    st.write(
        "This is a research/portfolio project, not a medical device. "
        "Predictions are not a diagnosis and should never be used for real "
        "clinical decisions."
    )

model = load_model()

uploaded_file = st.file_uploader("Upload an MRI scan (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Couldn't read that file as an image. Try a different file.")
        st.stop()

    image = preprocess(img_bgr)

    prediction = model.predict(image)
    class_index = int(np.argmax(prediction))
    predicted_class = config.CATEGORIES[class_index]
    confidence = float(prediction[0][class_index])

    st.subheader(f"Prediction: **{predicted_class}**  ({confidence:.1%} confidence)")

    st.bar_chart({
        "confidence": {
            cat: float(prediction[0][i]) for i, cat in enumerate(config.CATEGORIES)
        }
    }["confidence"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original**")
        st.image(cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.spinner("Generating explanations..."):
        lime_explanation = generate_lime_explanation(model, image)
        last_conv_layer = get_last_conv_layer(model)
        grad_cam_result = grad_cam(model, image, class_index, last_conv_layer)

    with col2:
        st.markdown("**LIME** (which regions drove the decision)")
        st.image(lime_explanation, use_container_width=True)

    with col3:
        st.markdown("**Grad-CAM** (where the CNN focused)")
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB))
        ax.imshow(
            cv2.resize(grad_cam_result, (image.shape[2], image.shape[1])),
            alpha=0.5, cmap="jet",
        )
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
else:
    st.info("Upload an MRI scan above to get a prediction and explanation.")

st.divider()
st.caption(
    "[View source on GitHub](https://github.com/Atul200429) · "
    "Built with TensorFlow, LIME, and Grad-CAM."
)
