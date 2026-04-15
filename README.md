# Container-for-tom500
# TOM500 ML Pipeline — Developer Guide

## 🧠 Overview

This repository implements a **modular end-to-end machine learning pipeline** for processing MRI data from the TOM500 dataset.

The system covers the full lifecycle:

* feature extraction from medical images
* data fusion (clinical + imaging)
* model training and evaluation
* explainability via SHAP

The codebase is structured to be **extensible, reproducible, and research-ready**.

---

## 🏗️ Architecture

The pipeline is divided into independent stages:

```
MRI (.nii.gz) + Labels
        │
        ▼
[Stage 1] Feature Extraction
        │
        ▼
mri_features.csv
        │
        ▼
[Stage 2] Data Fusion (clinical + MRI)
        │
        ▼
merged_features.csv
        │
        ▼
[Stage 3] ML Training (3 arms)
        │
        ▼
results_summary.csv + dashboard
        │
        ▼
[Stage 4] SHAP Analysis
        │
        ▼
feature importance + plots
```

---

## 📦 Modules

### 1. `tom500_pipeline.py` (MAIN ENTRYPOINT)

Core pipeline orchestrator.

Responsibilities:

* MRI feature extraction
* clinical data integration
* ML training (3 arms)
* optional SHAP analysis

Key components:

* feature extractors (shape, intensity, texture, asymmetry)
* nested cross-validation
* model comparison framework

---

### 2. `one_dimensional_feature_extraction.py`

Low-level feature extraction from:

* DICOM images
* segmentation masks

Outputs:

* signal intensity (SI0–SI9)
* volumes (V0–V9)

Parallelized using `ProcessPoolExecutor`.

---

### 3. `ml_pipeline_extended.py`

Extended ML experimentation module:

* compares Clinical vs Imaging vs Combined
* supports AutoML (TPOT fallback)
* generates performance dashboards
* computes feature importance (Random Forest)

---

### 4. `shap_pipeline.py`

Lightweight SHAP analysis:

* feature ranking
* simple plots
* basic feature selection

---

### 5. `shap_analysis_pipeline.py`

Advanced interpretability module:

* grouping by anatomical structures
* threshold + cumulative feature selection
* publication-quality visualizations
* statistical summaries

---

## ⚙️ Setup

### Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:

* numpy, pandas
* scikit-learn
* matplotlib
* nibabel
* SimpleITK
* scikit-image

Optional:

* shap (for interpretability)
* tpot (for AutoML)

---

## ▶️ Running the Pipeline

### Full pipeline

```bash
python tom500_pipeline.py \
    --image-dir data/image \
    --label-dir data/label \
    --clinical clinical_characteristics.csv \
    --workers 4
```

### Skip feature extraction

```bash
python tom500_pipeline.py \
    --mri-csv mri_features.csv \
    --clinical clinical_characteristics.csv \
    --skip-extraction
```

---

## 🧪 Machine Learning Design

### Three-arm evaluation

| Arm | Features |
| --- | -------- |
| A   | Clinical |
| B   | MRI      |
| C   | Combined |

### Models

* Logistic Regression
* Random Forest
* Gradient Boosting
* SVM (RBF)
* AutoML (TPOT / GridSearch fallback)

### Validation

* Nested Cross-Validation (5 outer / 3 inner)
* Metric: ROC AUC
* Stability: std(AUC)

---

## 📊 Feature Engineering

MRI features include:

* Shape (volumes, bounding boxes)
* Intensity statistics
* Texture (GLCM)
* Asymmetry
* Anatomical ratios

Clinical features:

* Age, Sex, SmokingIndex
* DiseaseDuration
* Derived features (LogSmoke, IsSmoker)

---

## 🔍 SHAP Integration

If `shap` is installed:

* automatic feature importance computation
* export to `shap_importance.csv`
* advanced analysis via `shap_analysis_pipeline.py`

---

## 📁 Data Requirements

Expected structure:

```
data/
├── image/   # .nii.gz MRI scans
├── label/   # segmentation masks
└── clinical_characteristics.csv
```

Clinical CSV must include:

* Id or case_id
* CAS (for target generation)

---

## 🧩 Extending the Project

### Add new features

* extend:

  * `extract_*_features()` in `tom500_pipeline.py`
* ensure:

  * consistent naming
  * numeric outputs

---

### Add new models

* modify `_build_pipelines()`
* define:

  * sklearn Pipeline
  * param_grid

---

### Add new evaluation metrics

* extend:

  * `_nested_cv()`
* include:

  * additional metrics (F1, precision, etc.)

---

### Modify feature selection (SHAP)

* edit:

  * `select_by_threshold()`
  * `select_by_cumulative()`

---

## ⚡ Performance Notes

* Feature extraction supports multiprocessing
* ML pipelines use nested CV (computationally expensive)
* AutoML significantly increases runtime

---

## 🐳 Docker (optional)

If using containerized environment:

```bash
docker build -t tom500 .
docker run -it tom500
```

---

## 🛠️ Development Guidelines

* Follow modular design (each stage independent)
* Avoid hardcoded paths
* Use logging instead of print
* Keep feature naming consistent

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Submit PR with description

---

## 📜 License

Specify license (MIT / Apache 2.0)

---

