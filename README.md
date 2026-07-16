# 🧠 Early Detection System for Alzheimer's Disease

A CNN-based classifier for Alzheimer's stage detection from MRI scans, with
**LIME** and **Grad-CAM** explainability so predictions are interpretable
rather than a black box. Originally built as a college research project;
this version restructures the notebook prototype into a runnable, reusable
pipeline — plus a Streamlit app so anyone can try it without touching code.

## 🚀 Live demo

```bash
streamlit run app.py
```

Upload an MRI scan and get the predicted stage plus LIME + Grad-CAM
explanations, right in the browser. See [Deploying the demo](#deploying-the-demo-for-free)
to put this live on a public URL for recruiters to try.

## Overview

The model classifies a brain MRI scan into one of four stages:

- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

Every prediction can be explained two ways:

- **LIME** — highlights the image regions that most influenced the
  predicted class by perturbing superpixels and fitting a local surrogate
  model.
- **Grad-CAM** — visualizes which regions the CNN's last convolutional
  layer "looked at" via gradient-weighted class activation maps.

## Project structure

```
.
├── app.py                # Streamlit demo: upload an MRI, get prediction + explanation
├── src/
│   ├── config.py      # paths, image size, categories, hyperparameters
│   ├── data.py         # dataset loading + preprocessing
│   ├── model.py        # CNN architecture
│   ├── train.py         # CLI: train and save a model
│   ├── evaluate.py      # CLI: classification report + confusion matrix
│   └── explain.py       # CLI: LIME + Grad-CAM for a single image
├── notebooks/           # original exploratory notebooks (fixed, kept for reference)
├── requirements.txt
└── .github/workflows/ci.yml   # installs deps + import-checks src/ on every push
```

## Setup

```bash
git clone https://github.com/Atul200429/ALZHEIMER-MRI-DETECTION-XAI.git
cd ALZHEIMER-MRI-DETECTION-XAI
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### Dataset

Trained on the [OASIS Alzheimer's Detection dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
on Kaggle — a large-scale brain MRI collection (~80,000 images across the
4 classes above). Note that this dataset is **heavily class-imbalanced**
(roughly NonDemented ≫ VeryMildDemented > MildDemented ≫ ModerateDemented),
so accuracy alone is a misleading metric here — check the per-class
precision/recall/F1 in `outputs/metrics/classification_report.txt` rather
than relying on overall accuracy, and consider class weighting or
oversampling if minority-class recall is poor.

Download it from Kaggle and lay it out locally as:

```
data/
├── MildDemented/*.jpg
├── ModerateDemented/*.jpg
├── NonDemented/*.jpg
└── VeryMildDemented/*.jpg
```

`data/` is git-ignored — the dataset is not committed to this repo. Point
`--data-dir` at a different location if you keep it elsewhere.

Each image is resized to 176×208, cast to float32, and scaled to [0, 1]
before being fed to the model (see `src/data.py` / `src/explain.py`).

## Usage

Train:

```bash
python -m src.train --data-dir data --epochs 10 --output outputs
```

Evaluate (classification report + confusion matrix):

```bash
python -m src.evaluate --data-dir data --model outputs/model/ad_xai_model
```

Explain a single prediction:

```bash
python -m src.explain --image path/to/scan.jpg --model outputs/model/ad_xai_model
```

## Model architecture

A compact CNN trained from scratch (see `src/model.py`):

| Layer | Details | Output shape |
|---|---|---|
| Conv2D | 32 filters, 3×3, ReLU, same padding | 208×176×32 |
| MaxPooling2D | 2×2, stride 2 | 104×88×32 |
| Conv2D | 32 filters, 3×3, ReLU, same padding | 104×88×32 |
| MaxPooling2D | 2×2, stride 2 | 52×44×32 |
| Dropout | rate 0.5 (regularization) | 52×44×32 |
| Flatten | — | 73,216 |
| Dense | 128 units, ReLU | 128 |
| Dense | 4 units, softmax | 4 |

Optimizer: Adam. Loss: categorical cross-entropy. Metrics: accuracy
(plus precision/recall/F1/confusion matrix computed separately in
`src/evaluate.py`).

## Results

*Fill in after training on your own run:*

| Metric | Value |
|---|---|
| Test accuracy | — |
| Test loss | — |
| Precision / Recall / F1 (per class) | see `outputs/metrics/classification_report.txt` |

Run `python -m src.evaluate` to regenerate the classification report and
confusion matrix, then drop the numbers in here — recruiters generally trust
a filled-in results table over a claimed-but-unverifiable accuracy figure.

## Changes from the original notebook prototype

- Removed the Google Colab-only cells (`drive.mount`, `files.upload`) so the
  project runs on any machine, not just inside Colab.
- Added inverse-frequency class weighting in `train.py` to handle the OASIS
  dataset's severe class imbalance (NonDemented vastly outnumbers
  ModerateDemented) — without it, accuracy alone is misleading.
- Fixed a loss-function/label mismatch: the model was compiled with
  `sparse_categorical_crossentropy` while being fed one-hot labels; now uses
  `categorical_crossentropy` to match.
- Fixed the evaluation confusion matrix, which was comparing one-hot labels
  against class-index predictions; now both are class indices.
- Replaced the interactive `input()` prompt for picking an image to explain
  with a `--image` CLI argument, so explanation generation can run
  non-interactively.
- Split one monolithic notebook into `data.py` / `model.py` / `train.py` /
  `evaluate.py` / `explain.py` so each stage is independently runnable,
  testable, and reusable.
- Fixed the notebooks' `metadata.widgets` block, which was missing the
  required `state` key and made GitHub refuse to render both `.ipynb` files
  ("Invalid Notebook" error).
- Added `requirements.txt`, `.gitignore`, `LICENSE`, and a CI workflow that
  installs dependencies and import-checks the pipeline on every push.

## Deploying the demo for free

[Streamlit Community Cloud](https://share.streamlit.io) hosts this kind of
app for free and gives you a public URL to put in your resume/LinkedIn.

1. Push this repo to GitHub (model included — see the note below).
2. Go to share.streamlit.io → **New app** → pick this repo, branch `main`,
   main file path `app.py`.
3. Click **Deploy**. First build takes a few minutes (installing
   TensorFlow); after that it's live at `https://<something>.streamlit.app`.

**Getting the trained model onto the deployed app** — `outputs/` is
git-ignored by default, so the app needs the model reachable one of two
ways:

- **Commit it anyway.** This CNN is small; remove `outputs/` from
  `.gitignore` (or `git add -f outputs/model/ad_xai_model`) and commit it.
  Simplest option if the model directory is well under GitHub's 100MB file
  limit.
- **Host it externally.** Upload the trained model to a GitHub Release,
  Hugging Face Hub, or Google Drive (direct-download link), then set a
  `MODEL_URL` secret in the Streamlit Cloud app settings — `app.py`
  downloads it automatically on first run.

## References

- Lokesh, K., Challa, N. P., Satwik, A. S., Kiran, J. C., Rao, N. K., & Naseeba, B. (2023). *Early Alzheimer's Disease Detection Using Deep Learning*. doi: 10.4108/eetpht.9.3966
- Foroughipoor, S., Moradi, K., & Bolhasani, H. (2023). *Alzheimer's Disease Diagnosis by Deep Learning Using MRI-Based Approaches*. https://doi.org/10.48550/arXiv.2310.17755
- Aghaei, A., & Moghaddam, M. E. (2023). *Smart ROI Detection for Alzheimer's Disease prediction using explainable AI*. https://doi.org/10.48550/arXiv.2303.10401

## Collaborators

- [Atul Krishna](https://github.com/Atul200429)

## License

MIT — see [LICENSE](LICENSE).
