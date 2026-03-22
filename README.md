# 🔬 Colorectal Cancer Histology Classification

A deep learning research project for classifying colorectal histology images into 8 tissue types using a custom CNN, with Grad-CAM explainability (XAI) for clinical interpretability.

---

## Overview

This project builds and evaluates a custom Convolutional Neural Network (CNN) on the TensorFlow Datasets `colorectal_histology` dataset. The model classifies microscopic tissue images into 8 categories, with a focus on reproducibility, rigorous evaluation, and explainable AI — making it suitable for a medical/research context.

---

## Classes (8 Tissue Types)

| Class | Description |
|---|---|
| Tumor | Cancerous tissue |
| Stroma | Connective tissue |
| Complex | Complex glandular structures |
| Lympho | Lymphocyte-rich regions |
| Debris | Necrotic / debris tissue |
| Mucosa | Normal mucosal lining |
| Adipose | Fatty tissue |
| Empty | Background / empty regions |

---

## Features

- Reproducibility setup — fixed random seeds across Python, NumPy, TensorFlow, and CUDA
- 70/15/15 train/validation/test stratified split via TensorFlow Datasets
- Data augmentation — random flip, rotation, zoom applied only to training data
- Custom CNN architecture with BatchNormalization and MaxPooling blocks
- Training with ReduceLROnPlateau and EarlyStopping callbacks
- Comprehensive evaluation — accuracy/loss curves, confusion matrix, per-class classification report
- ROC curves with AUC scores (One-vs-Rest) for all 8 classes
- **Explainable AI (XAI)** — Grad-CAM heatmap visualizations highlighting which image regions the model focused on
- Per-class clinical explanations describing tissue meaning and pathological significance

---

## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building, training, augmentation |
| TensorFlow Datasets | Dataset loading and splitting |
| scikit-learn | Confusion matrix, ROC-AUC, classification report |
| OpenCV | Image processing for Grad-CAM |
| matplotlib / seaborn | Visualizations |
| NumPy | Numerical operations |

---

## Model Architecture

```
Input (224×224×3)
├── Conv2D(32) → BatchNorm → MaxPool
├── Conv2D(64) → BatchNorm → MaxPool
├── Conv2D(128) → BatchNorm → MaxPool
├── Flatten
├── Dense → Dropout
└── Dense(8, softmax)
```

- Optimizer: Adam (lr=1e-4, with ReduceLROnPlateau)
- Loss: Sparse Categorical Crossentropy
- Epochs: up to 50 (EarlyStopping, patience=5)
- Batch size: 8

---

## Explainability (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping) generates heatmaps overlaid on input images, showing which tissue regions the model weighted most for its prediction — an important step for building trust in medical AI systems.

```
Blue   → Low importance
Yellow → Moderate importance  
Red    → High importance (model focused here)
```

---

## How to Run

1. Open `colon_cancer_detection_final_.ipynb` in Google Colab
2. Mount Google Drive (model checkpoints saved there)
3. Run all cells in order — the dataset downloads automatically via `tensorflow_datasets`

---

## Project Structure

```
colon-cancer-detection/
├── colon_cancer_detection_final_.ipynb   # Full pipeline notebook
└── README.md
```

> **Note:** Model weights are saved to Google Drive during training and are not included in this repo.
