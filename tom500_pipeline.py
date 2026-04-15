"""
TOM500 Full ML Pipeline
=======================
Single entry-point that runs the complete workflow in three stages:

  STAGE 1 — MRI Feature Extraction
  ─────────────────────────────────
  Loads every paired (image, label) NIfTI case and extracts:
    • Shape      : volume (voxels + mm³), bounding-box dims, aspect ratios
    • Intensity  : mean, std, median, p10, p90, skewness, kurtosis
    • Texture    : GLCM contrast / homogeneity / energy / correlation
    • Asymmetry  : left/right split, asymmetry index
    • Ratios     : fat/muscle, nerve/fat, SR/IR, …
  One row per case → saved as  mri_features.csv

  STAGE 2 — Data Fusion
  ──────────────────────
  Merges mri_features.csv with clinical_characteristics.csv on case_id.
  Clinical feature engineering:
    Age, Sex_enc, SmokingIndex (raw + log), DiseaseDuration, IsSmoker
  Target: CAS ≥ 2  (active thyroid-eye disease)

  STAGE 3 — Three-Arm ML Comparison
  ───────────────────────────────────
  ARM A : Clinical features only          (6 features)
  ARM B : MRI imaging features only       (234 features)
  ARM C : Clinical + MRI combined         (240 features)

  Each arm runs:
    • LogReg, RandomForest, GradBoost, SVM-RBF  (nested 5-outer/3-inner CV)
    • AutoML (TPOT if installed, else exhaustive GridSearch ensemble)
  Metrics: AUC, training time, σ AUC (stability)

Dataset layout expected
-----------------------
  <image_dir>/   1.nii.gz  2.nii.gz  ...
  <label_dir>/   1.nii.gz  2.nii.gz  ...

Label classes
-------------
  1 eyeball  2 fat  3 optic_nerve  4 muscle  5 lacrimal_gland
  6 superior_rectus  7 inferior_rectus  8 medial_rectus  9 lateral_rectus

Outputs
-------
  mri_features.csv            per-case MRI feature table (Stage 1)
  merged_features.csv         clinical + MRI fused table (Stage 2)
  results_summary.csv         AUC / time / σ per arm × model (Stage 3)
  ml_combined_dashboard.png   6-panel comparison dashboard  (Stage 3)
  shap_summary.png            SHAP beeswarm summary plot    (Stage 4)
  shap_importance.csv         mean |SHAP| ranked features   (Stage 4)

Usage
-----
  # Full pipeline (all three stages)
  python tom500_pipeline.py \\
      --image-dir  train/image \\
      --label-dir  train/label \\
      --clinical   clinical_characteristics.csv \\
      --workers    4

  # Skip extraction if mri_features.csv already exists
  python tom500_pipeline.py \\
      --image-dir  train/image  --label-dir  train/label \\
      --clinical   clinical_characteristics.csv \\
      --mri-csv    mri_features.csv  --skip-extraction
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────────────────────────────────────
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from skimage.feature import graycomatrix, graycoprops

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from tpot import TPOTClassifier
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logging.getLogger("tom500").warning(
        "shap not installed — Stage 4 will be skipped. "
        "Install with: pip install shap"
    )


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# ── MRI labels ────────────────────────────────────────────────────────────────
LABEL_MAP: Dict[int, str] = {
    1: "eyeball",
    2: "fat",
    3: "optic_nerve",
    4: "muscle",
    5: "lacrimal_gland",
    6: "superior_rectus",
    7: "inferior_rectus",
    8: "medial_rectus",
    9: "lateral_rectus",
}

# ── GLCM settings ─────────────────────────────────────────────────────────────
GLCM_DISTANCES = [1]
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS    = 64
GLCM_PROPS     = ["contrast", "homogeneity", "energy", "correlation"]

# ── Clinical feature columns ──────────────────────────────────────────────────
CLINICAL_FEATS = [
    "Age", "Sex_enc", "SmokingIndex", "LogSmoke",
    "DiseaseDuration", "IsSmoker",
]

# ── ML / visualisation ────────────────────────────────────────────────────────
ARM_LABELS  = ["Clinical", "MRI Imaging", "Combined"]
MODEL_NAMES = ["LogReg", "RandForest", "GradBoost", "SVM-RBF", "AutoML"]

PALETTE = dict(
    bg     = "#ffffff",
    panel  = "#ebeffe",
    border = "#191c2b",
    text   = "#0B0D15",
    muted  = "#222939",
    gridl  = "#141720",
    armA   = "#4d8df5",   # blue   – clinical
    armB   = "#f5c94d",   # amber  – MRI
    armC   = "#4df5a8",   # teal   – combined
    danger = "#f54d7a",
)
ARM_COLORS = [PALETTE["armA"], PALETTE["armB"], PALETTE["armC"]]


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tom500")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1  —  MRI FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (float32 array [X,Y,Z], voxel_spacing_mm [3])."""
    img   = nib.load(str(path))
    arr   = np.asarray(img.dataobj, dtype=np.float32)
    zooms = np.abs(np.array(img.header.get_zooms()[:3], dtype=np.float64))
    zooms = np.where(zooms == 0, 1.0, zooms)
    return arr, zooms


def _discover_cases(image_dir: Path, label_dir: Path) -> List[str]:
    """Sorted list of case IDs present in BOTH directories."""
    img_ids = {p.name.replace(".nii.gz", "") for p in image_dir.glob("*.nii.gz")}
    lbl_ids = {p.name.replace(".nii.gz", "") for p in label_dir.glob("*.nii.gz")}
    only_i  = img_ids - lbl_ids
    only_l  = lbl_ids - img_ids
    if only_i:
        log.warning("Image-only (no label, skipped): %s", only_i)
    if only_l:
        log.warning("Label-only (no image, skipped): %s", only_l)
    common = sorted(img_ids & lbl_ids, key=lambda x: int(x) if x.isdigit() else x)
    log.info("Discovered %d paired cases.", len(common))
    return common


# ── 1a. Shape features ────────────────────────────────────────────────────────

def extract_shape_features(
    mask:          np.ndarray,
    label_id:      int,
    voxel_spacing: np.ndarray,
) -> Dict[str, float]:
    """
    Volume (voxels + mm³), bounding-box dimensions (voxels + mm),
    and three aspect ratios for *label_id*.

    All values are NaN when the label is absent in *mask*.
    """
    prefix        = f"{LABEL_MAP[label_id]}_shape"
    roi           = mask == label_id
    voxel_vol_mm3 = float(np.prod(voxel_spacing))
    _nan_keys     = [
        "volume_voxels", "volume_mm3",
        "bb_dx", "bb_dy", "bb_dz",
        "bb_dx_mm", "bb_dy_mm", "bb_dz_mm",
        "ar_dx_dy", "ar_dx_dz", "ar_dy_dz",
    ]

    if not roi.any():
        return {f"{prefix}_{k}": np.nan for k in _nan_keys}

    idx         = np.argwhere(roi)
    mn, mx      = idx.min(axis=0), idx.max(axis=0)
    dx, dy, dz  = (mx - mn + 1).astype(float)
    vol_vox     = float(roi.sum())

    def _r(a: float, b: float) -> float:
        return a / b if b != 0 else np.nan

    return {
        f"{prefix}_volume_voxels": vol_vox,
        f"{prefix}_volume_mm3":    vol_vox * voxel_vol_mm3,
        f"{prefix}_bb_dx":         dx,
        f"{prefix}_bb_dy":         dy,
        f"{prefix}_bb_dz":         dz,
        f"{prefix}_bb_dx_mm":      dx * voxel_spacing[0],
        f"{prefix}_bb_dy_mm":      dy * voxel_spacing[1],
        f"{prefix}_bb_dz_mm":      dz * voxel_spacing[2],
        f"{prefix}_ar_dx_dy":      _r(dx, dy),
        f"{prefix}_ar_dx_dz":      _r(dx, dz),
        f"{prefix}_ar_dy_dz":      _r(dy, dz),
    }


# ── 1b. Intensity features ────────────────────────────────────────────────────

def extract_intensity_features(
    image:    np.ndarray,
    mask:     np.ndarray,
    label_id: int,
) -> Dict[str, float]:
    """
    First-order intensity statistics (mean, std, median, p10, p90,
    skewness, kurtosis) for voxels belonging to *label_id*.

    Returns NaN for all keys when fewer than 2 voxels are present.
    """
    prefix  = f"{LABEL_MAP[label_id]}_int"
    voxels  = image[mask == label_id]
    _nan    = {f"{prefix}_{k}": np.nan
               for k in ["mean", "std", "median", "p10", "p90",
                          "skewness", "kurtosis"]}

    if voxels.size < 2:
        return _nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            f"{prefix}_mean":     float(voxels.mean()),
            f"{prefix}_std":      float(voxels.std()),
            f"{prefix}_median":   float(np.median(voxels)),
            f"{prefix}_p10":      float(np.percentile(voxels, 10)),
            f"{prefix}_p90":      float(np.percentile(voxels, 90)),
            f"{prefix}_skewness": float(stats.skew(voxels)),
            f"{prefix}_kurtosis": float(stats.kurtosis(voxels)),
        }


# ── 1c. Texture features ──────────────────────────────────────────────────────

def _rescale_uint8(patch: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    lo, hi = patch.min(), patch.max()
    if hi == lo:
        return np.zeros_like(patch, dtype=np.uint8)
    return ((patch - lo) / (hi - lo) * (levels - 1)).astype(np.uint8)


def extract_texture_features(
    image:    np.ndarray,
    mask:     np.ndarray,
    label_id: int,
) -> Dict[str, float]:
    """
    GLCM texture features (contrast, homogeneity, energy, correlation)
    computed on the central axial slice of the label's bounding box,
    averaged across four angles.

    Returns NaN for all keys when the label is absent or the patch
    is degenerate (< 4 pixels).
    """
    prefix  = f"{LABEL_MAP[label_id]}_tex"
    _nan    = {f"{prefix}_{p}": np.nan for p in GLCM_PROPS}
    roi_3d  = mask == label_id

    if not roi_3d.any():
        return _nan

    # central occupied Z slice
    z_occ  = np.where(roi_3d.any(axis=(0, 1)))[0]
    z_cent = int(np.median(z_occ))
    roi_2d = roi_3d[:, :, z_cent]
    if not roi_2d.any():
        z_cent = z_occ[0]
        roi_2d = roi_3d[:, :, z_cent]

    img_2d = image[:, :, z_cent].copy()
    img_2d[~roi_2d] = img_2d[roi_2d].min()

    # crop to bounding box
    rows, cols = np.where(roi_2d)
    patch = img_2d[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    if patch.size < 4:
        return _nan

    try:
        glcm = graycomatrix(
            _rescale_uint8(patch, GLCM_LEVELS),
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=GLCM_LEVELS,
            symmetric=True,
            normed=True,
        )
        return {f"{prefix}_{p}": float(graycoprops(glcm, p).mean())
                for p in GLCM_PROPS}
    except Exception as exc:
        log.debug("GLCM failed label %d: %s", label_id, exc)
        return _nan


# ── 1d. Asymmetry features ────────────────────────────────────────────────────

def extract_asymmetry_features(
    mask:     np.ndarray,
    label_id: int,
) -> Dict[str, float]:
    """
    Left / right volumetric asymmetry by splitting at mid-X.

        asymmetry_index = |left − right| / total
    """
    prefix = f"{LABEL_MAP[label_id]}_asym"
    roi    = mask == label_id
    total  = int(roi.sum())
    _nan   = {f"{prefix}_{k}": np.nan
              for k in ["left_voxels", "right_voxels", "asymmetry"]}

    if total == 0:
        return _nan

    mid   = mask.shape[0] // 2
    left  = int(roi[:mid].sum())
    right = int(roi[mid:].sum())

    return {
        f"{prefix}_left_voxels":  float(left),
        f"{prefix}_right_voxels": float(right),
        f"{prefix}_asymmetry":    abs(left - right) / total,
    }


# ── 1e. Ratio features ────────────────────────────────────────────────────────

def extract_ratio_features(shape_feats: Dict[str, float]) -> Dict[str, float]:
    """
    Nine clinically motivated volume ratios derived from the merged
    shape feature dict (all labels combined).
    """
    def _vol(name: str) -> float:
        return shape_feats.get(f"{name}_shape_volume_voxels", np.nan)

    def _div(n: float, d: float) -> float:
        return n / d if (not np.isnan(n) and not np.isnan(d) and d != 0) else np.nan

    fat      = _vol("fat")
    eyeball  = _vol("eyeball")
    optic_n  = _vol("optic_nerve")
    lacrimal = _vol("lacrimal_gland")
    sr       = _vol("superior_rectus")
    ir       = _vol("inferior_rectus")
    mr       = _vol("medial_rectus")
    lr       = _vol("lateral_rectus")
    muscle   = _vol("muscle")

    # aggregate all rectus + generic muscle label
    all_muscle_parts = [v for v in [muscle, sr, ir, mr, lr] if not np.isnan(v)]
    all_muscle = float(sum(all_muscle_parts)) if all_muscle_parts else np.nan

    return {
        "ratio_fat_to_muscle":       _div(fat,       all_muscle),
        "ratio_muscle_to_eyeball":   _div(all_muscle, eyeball),
        "ratio_nerve_to_fat":        _div(optic_n,   fat),
        "ratio_lacrimal_to_fat":     _div(lacrimal,  fat),
        "ratio_sr_to_ir":            _div(sr,        ir),
        "ratio_mr_to_lr":            _div(mr,        lr),
        "ratio_eyeball_to_fat":      _div(eyeball,   fat),
        "ratio_nerve_to_eyeball":    _div(optic_n,   eyeball),
        "ratio_total_muscle_to_fat": _div(all_muscle, fat),
    }


# ── 1f. Per-case worker ───────────────────────────────────────────────────────

def _extract_one_case(
    case_id:   str,
    image_dir: Path,
    label_dir: Path,
) -> Optional[Dict]:
    """
    Full feature extraction for one case.  Called inside worker processes.
    Returns a flat dict (one DataFrame row) or None on unrecoverable error.
    """
    try:
        image, voxel_spacing = _load_nifti(image_dir / f"{case_id}.nii.gz")
        mask,  _             = _load_nifti(label_dir / f"{case_id}.nii.gz")
        mask                 = mask.astype(np.int32)
    except Exception as exc:
        log.error("Case %s — NIfTI load failed: %s", case_id, exc)
        return None

    if image.shape != mask.shape:
        log.warning("Case %s — shape mismatch %s vs %s, skipped.",
                    case_id, image.shape, mask.shape)
        return None

    row: Dict        = {"case_id": case_id}
    all_shape: Dict  = {}

    for lid in LABEL_MAP:
        for fn, kwargs in [
            (extract_shape_features,     dict(mask=mask, label_id=lid,
                                              voxel_spacing=voxel_spacing)),
            (extract_intensity_features, dict(image=image, mask=mask, label_id=lid)),
            (extract_texture_features,   dict(image=image, mask=mask, label_id=lid)),
            (extract_asymmetry_features, dict(mask=mask, label_id=lid)),
        ]:
            try:
                feats = fn(**kwargs)
            except Exception as exc:
                log.debug("Case %s  %s  label %d: %s",
                          case_id, fn.__name__, lid, exc)
                feats = {}
            row.update(feats)
            if fn is extract_shape_features:
                all_shape.update(feats)

    try:
        row.update(extract_ratio_features(all_shape))
    except Exception as exc:
        log.debug("Case %s  ratios: %s", case_id, exc)

    return row


# ── 1g. Dataset builder ───────────────────────────────────────────────────────

def build_features_dataset(
    image_dir: str | Path,
    label_dir: str | Path,
    n_workers: int = 1,
    out_csv:   Optional[str | Path] = "mri_features.csv",
) -> pd.DataFrame:
    """
    Iterate all paired cases in *image_dir* / *label_dir*, extract
    shape + intensity + texture + asymmetry + ratio features, and
    return a tidy DataFrame (one row per case).

    Parameters
    ----------
    image_dir : directory containing <case_id>.nii.gz MRI volumes
    label_dir : directory containing <case_id>.nii.gz segmentation masks
    n_workers : parallel worker processes  (1 = serial)
    out_csv   : where to write mri_features.csv  (None = skip)

    Returns
    -------
    pd.DataFrame
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    case_ids = _discover_cases(image_dir, label_dir)
    if not case_ids:
        log.warning("No cases found — returning empty DataFrame.")
        return pd.DataFrame()

    rows:   List[Dict] = []
    failed: List[str]  = []

    progress = (
        _tqdm(total=len(case_ids), desc="MRI extraction", unit="case",
              dynamic_ncols=True)
        if HAS_TQDM else None
    )

    def _collect(result: Optional[Dict], cid: str) -> None:
        (rows if result is not None else failed).append(
            result if result is not None else cid
        )
        if progress:
            progress.update(1)

    n_eff = min(n_workers, len(case_ids), cpu_count())

    if n_eff > 1:
        log.info("Parallel extraction: %d workers.", n_eff)
        with ProcessPoolExecutor(max_workers=n_eff) as ex:
            futs = {ex.submit(_extract_one_case, c, image_dir, label_dir): c
                    for c in case_ids}
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    _collect(fut.result(), cid)
                except Exception as exc:
                    log.error("Worker exception case %s: %s", cid, exc)
                    _collect(None, cid)
    else:
        log.info("Serial extraction.")
        for cid in case_ids:
            _collect(_extract_one_case(cid, image_dir, label_dir), cid)

    if progress:
        progress.close()
    if failed:
        log.warning("Failed cases (%d): %s", len(failed), failed)
    if not rows:
        log.error("No features extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # numeric sort by case_id
    try:
        df["_k"] = df["case_id"].apply(lambda x: int(x) if str(x).isdigit() else x)
        df = df.sort_values("_k").drop(columns=["_k"]).reset_index(drop=True)
    except Exception:
        pass

    log.info("Extraction done: %d cases × %d features.", len(df), df.shape[1] - 1)

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info("MRI features saved → %s", out_path)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2  —  DATA FUSION
# ═════════════════════════════════════════════════════════════════════════════

def _engineer_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered columns to the clinical DataFrame in-place.
    Works whether *df* is the raw clinical CSV or the post-merge DataFrame.
    """
    for col in ["Age", "SmokingIndex", "DiseaseDuration"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["CAS_num"]   = pd.to_numeric(df["CAS"], errors="coerce")
    df["Sex_enc"]   = (df["Sex"] == "M").astype(int)
    df["IsSmoker"]  = (df["SmokingIndex"].fillna(0) > 0).astype(int)
    df["LogSmoke"]  = np.log1p(df["SmokingIndex"].fillna(0))
    return df


def merge_with_clinical(
    clinical_csv: str | Path,
    mri_csv:      str | Path,
    how:          str = "inner",
    threshold:    int = 2,
    out_csv:      Optional[str | Path] = "merged_features.csv",
) -> pd.DataFrame:
    """
    Merge MRI features with clinical data and engineer features.

    The clinical CSV must have an 'Id' column  (<case_id>_<date>)  or
    a plain 'case_id' column.  Binary target: CAS >= *threshold* → 1.

    Parameters
    ----------
    clinical_csv : path to clinical_characteristics.csv
    mri_csv      : path to mri_features.csv
    how          : pandas merge strategy (inner / left / right / outer)
    threshold    : CAS binarisation cutoff
    out_csv      : path for merged CSV output  (None = skip)

    Returns
    -------
    pd.DataFrame with columns from both sources + engineered features + target
    """
    clin = pd.read_csv(clinical_csv)
    mri  = pd.read_csv(mri_csv)

    # normalise case_id key in clinical table
    if "Id" in clin.columns and "case_id" not in clin.columns:
        clin["case_id"] = clin["Id"].apply(lambda x: str(x).split("_")[0])
    elif "case_id" not in clin.columns:
        raise KeyError("Clinical CSV must have an 'Id' or 'case_id' column.")

    clin["case_id"] = clin["case_id"].astype(str)
    mri["case_id"]  = mri["case_id"].astype(str)

    merged = pd.merge(clin, mri, on="case_id", how=how,
                      suffixes=("_clin", "_mri"))
    merged = _engineer_clinical(merged)
    merged["target"] = (merged["CAS_num"] >= threshold).astype(int)

    log.info(
        "Merged clinical (%d) + MRI (%d) → %d rows, %d columns.",
        len(clin), len(mri), len(merged), merged.shape[1],
    )

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        log.info("Merged dataset saved → %s", out_path)

    return merged


def prepare_arm_arrays(
    merged: pd.DataFrame,
    mri_feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, List[str], List[str], List[str]]:
    """
    Extract numpy arrays for each arm from the merged DataFrame.

    Parameters
    ----------
    merged           : output of merge_with_clinical()
    mri_feature_cols : list of MRI feature column names to use

    Returns
    -------
    X_clin, X_mri, X_comb, y, feat_clin, feat_mri, feat_comb
    """
    clin_cols = [c for c in CLINICAL_FEATS if c in merged.columns]
    mri_cols  = [c for c in mri_feature_cols if c in merged.columns]
    comb_cols = clin_cols + mri_cols

    imp = SimpleImputer(strategy="median")

    X_clin = imp.fit_transform(merged[clin_cols].values)
    X_mri  = imp.fit_transform(merged[mri_cols].values)
    X_comb = imp.fit_transform(merged[comb_cols].values)
    y      = merged["target"].values

    log.info(
        "Arm arrays — clinical: %d  MRI: %d  combined: %d  "
        "n_samples: %d  active: %.1f%%",
        X_clin.shape[1], X_mri.shape[1], X_comb.shape[1],
        len(y), y.mean() * 100,
    )
    return X_clin, X_mri, X_comb, y, clin_cols, mri_cols, comb_cols


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3  —  ML MODELS & NESTED CV
# ═════════════════════════════════════════════════════════════════════════════

def _build_pipelines() -> Dict[str, Tuple[Pipeline, Dict]]:
    """Return {name: (pipeline, param_grid)} for all four manual classifiers."""
    base = [("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler())]
    return {
        "LogReg": (
            Pipeline(base + [("clf", LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=42))]),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "RandForest": (
            Pipeline([("imp", SimpleImputer()),
                      ("clf", RandomForestClassifier(
                          n_estimators=100, class_weight="balanced",
                          random_state=42, n_jobs=1))]),
            {"clf__n_estimators": [50, 100], "clf__max_depth": [None, 5]},
        ),
        "GradBoost": (
            Pipeline([("imp", SimpleImputer()),
                      ("clf", GradientBoostingClassifier(
                          n_estimators=80, learning_rate=0.1,
                          max_depth=3, random_state=42))]),
            {"clf__learning_rate": [0.05, 0.1]},
        ),
        "SVM-RBF": (
            Pipeline(base + [("clf", SVC(
                kernel="rbf", probability=True,
                class_weight="balanced", random_state=42))]),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale"]},
        ),
    }


def _nested_cv(
    pipe:       Pipeline,
    X:          np.ndarray,
    y:          np.ndarray,
    name:       str,
    param_grid: Optional[Dict] = None,
    outer:      int = 5,
    inner:      int = 3,
) -> Dict:
    """
    Outer loop estimates generalisation performance.
    Inner loop selects hyperparameters via GridSearchCV.

    Returns dict with model name, per-fold AUCs, mean, std and wall-time.
    """
    outer_cv = StratifiedKFold(outer, shuffle=True, random_state=42)
    aucs, t0 = [], time.time()

    for tr, te in outer_cv.split(X, y):
        if param_grid:
            gs = GridSearchCV(
                pipe, param_grid,
                cv=StratifiedKFold(inner, shuffle=True, random_state=42),
                scoring="roc_auc", n_jobs=1,
            )
            gs.fit(X[tr], y[tr])
            best = gs.best_estimator_
        else:
            pipe.fit(X[tr], y[tr])
            best = pipe

        aucs.append(roc_auc_score(y[te], best.predict_proba(X[te])[:, 1]))

    return dict(
        model           = name,
        auc_scores      = aucs,
        mean_auc        = float(np.mean(aucs)),
        std_auc         = float(np.std(aucs)),
        training_time_s = float(time.time() - t0),
    )


def _run_automl(
    X:      np.ndarray,
    y:      np.ndarray,
    models: Dict,
    cv:     int = 5,
) -> Dict:
    """
    TPOT AutoML if installed; otherwise exhaustive per-fold best-of-grids.
    """
    outer_cv   = StratifiedKFold(cv, shuffle=True, random_state=42)
    aucs, t0   = [], time.time()
    label      = "AutoML-TPOT" if HAS_TPOT else "AutoML"

    for tr, te in outer_cv.split(X, y):
        if HAS_TPOT:
            clf = TPOTClassifier(generations=3, population_size=10, cv=3,
                                 scoring="roc_auc", random_state=42,
                                 verbosity=0, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:, 1]
        else:
            best_a, prob = -1.0, None
            icv = StratifiedKFold(3, shuffle=True, random_state=42)
            for _, (pipe, pg) in models.items():
                gs = GridSearchCV(pipe, pg, cv=icv,
                                  scoring="roc_auc", n_jobs=1)
                gs.fit(X[tr], y[tr])
                p = gs.best_estimator_.predict_proba(X[te])[:, 1]
                a = roc_auc_score(y[te], p)
                if a > best_a:
                    best_a, prob = a, p
        aucs.append(roc_auc_score(y[te], prob))

    return dict(
        model           = label,
        auc_scores      = aucs,
        mean_auc        = float(np.mean(aucs)),
        std_auc         = float(np.std(aucs)),
        training_time_s = float(time.time() - t0),
    )


def run_arm(X: np.ndarray, y: np.ndarray, label: str) -> List[Dict]:
    """Run all five models for one arm, print fold-level progress."""
    models = _build_pipelines()
    log.info("ARM  %s  [%d features, %d samples]", label, X.shape[1], len(y))
    results = []

    for name, (pipe, pg) in models.items():
        sys.stdout.write(f"  {label} / {name} … ")
        sys.stdout.flush()
        r = _nested_cv(pipe, X, y, name, param_grid=pg)
        sys.stdout.write(f"AUC={r['mean_auc']:.3f}  σ={r['std_auc']:.3f}\n")
        sys.stdout.flush()
        results.append(r)

    sys.stdout.write(f"  {label} / AutoML … ")
    sys.stdout.flush()
    r = _run_automl(X, y, models)
    sys.stdout.write(f"AUC={r['mean_auc']:.3f}  σ={r['std_auc']:.3f}\n")
    results.append(r)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3b  —  FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════

def compute_feature_importance(
    X:          np.ndarray,
    y:          np.ndarray,
    feat_names: List[str],
    top_n:      int = 20,
) -> Tuple[List[str], np.ndarray]:
    """
    Fit a Random Forest on the full dataset and return the top-N MDI
    feature names and importance scores (ascending, for barh plotting).
    """
    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ])
    rf.fit(X, y)
    mdi = rf.named_steps["clf"].feature_importances_
    idx = np.argsort(mdi)[-top_n:]
    return [feat_names[i] for i in idx], mdi[idx]


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4  —  SHAP EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════

def run_shap_analysis(
    X:              np.ndarray,
    y:              np.ndarray,
    feature_names:  List[str],
    output_path:    str | Path = "shap_summary.png",
    importance_csv: str | Path = "shap_importance.csv",
    top_n_console:  int = 10,
) -> Optional[pd.DataFrame]:
    """
    Train a RandomForest, compute SHAP values via TreeExplainer, and produce:
      • shap_summary.png    — beeswarm summary plot styled to match the dashboard
      • shap_importance.csv — mean |SHAP| ranked feature importance table

    Parameters
    ----------
    X              : feature matrix (will be imputed internally)
    y              : binary target vector
    feature_names  : ordered list of feature names matching X columns
    output_path    : destination for the SHAP summary PNG
    importance_csv : destination for the ranked importance CSV
    top_n_console  : how many top features to print to stdout

    Returns
    -------
    pd.DataFrame of ranked features, or None if shap is unavailable.
    """
    if not HAS_SHAP:
        log.warning("Skipping SHAP analysis — shap package not available.")
        return None

    # ── 1. Train RandomForest (same settings as compute_feature_importance) ──
    log.info("Fitting RandomForest for SHAP analysis …")
    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ])
    rf.fit(X, y)

    # Separate the imputed matrix and bare classifier so TreeExplainer
    # receives a plain ndarray and a plain sklearn estimator (not a Pipeline).
    imp_X  = rf.named_steps["imp"].transform(X)
    rf_clf = rf.named_steps["clf"]

    # ── 2. SHAP values ────────────────────────────────────────────────────────
    log.info("Computing SHAP values (TreeExplainer) …")
    explainer   = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(imp_X)

    # sklearn RF returns a list [class0_array, class1_array]; guard for newer
    # shap versions that may return a single 3-D array instead.
    if isinstance(shap_values, list):
        sv1 = shap_values[1]
    else:
        sv1 = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    # ── 3. Summary plot (styled to match the dashboard palette) ──────────────
    log.info("Generating SHAP summary plot → %s", output_path)

    # Reset any lingering rcParams from previous plots, then apply our theme.
    plt.rcdefaults()
    plt.rcParams.update({
        "text.color":       PALETTE["text"],
        "axes.labelcolor":  PALETTE["muted"],
        "xtick.color":      PALETTE["muted"],
        "ytick.color":      PALETTE["text"],
        "axes.facecolor":   PALETTE["panel"],
        "figure.facecolor": PALETTE["bg"],
    })

    n_display = min(30, len(feature_names))
    fig_h     = max(8, n_display * 0.30)
    plt.figure(figsize=(14, fig_h), facecolor=PALETTE["bg"])

    shap.summary_plot(
        sv1,
        imp_X,
        feature_names = feature_names,
        show          = False,
        plot_size     = None,           # we own the figure sizing
        color_bar     = True,
        max_display   = n_display,
    )

    ax = plt.gca()
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_edgecolor(PALETTE["border"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("SHAP value  (impact on model output — predicting CAS ≥ 2)",
                  color=PALETTE["muted"], fontsize=9)
    ax.grid(axis="x", color=PALETTE["gridl"], lw=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    plt.title(
        "SHAP Summary  ·  CAS ≥ 2 Prediction  (Combined Arm · RF)",
        color=PALETTE["text"], fontsize=15, fontweight="bold",
        fontfamily="monospace", pad=14,
    )

    # Annotate with a concise legend explaining colour bar direction
    plt.gcf().text(
        0.98, 0.01,
        "Colour bar: feature value  (red = high, blue = low)",
        ha="right", va="bottom",
        fontsize=14, color=PALETTE["muted"], fontfamily="monospace",
    )

    plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    plt.rcdefaults()   # restore rcParams so later plots are unaffected
    log.info("SHAP summary plot saved → %s", output_path)

    # ── 4. Mean |SHAP| importance table ──────────────────────────────────────
    importance = np.abs(sv1).mean(axis=0)

    df_imp = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    df_imp.to_csv(str(importance_csv), index=False)
    log.info("SHAP importance CSV saved → %s", importance_csv)

    # ── 5. Console summary ────────────────────────────────────────────────────
    n_show = min(top_n_console, len(df_imp))
    print(f"\n  Top {n_show} features by mean |SHAP| value  "
          f"(Combined Arm · RF · CAS ≥ 2):")
    print(f"  {'Rank':<5} {'Feature':<50} {'Mean |SHAP|':>12}")
    print(f"  {'─'*5} {'─'*50} {'─'*12}")
    for rank, row in df_imp.head(n_show).iterrows():
        print(f"  {rank + 1:<5} {row['feature']:<50} {row['mean_abs_shap']:>12.5f}")
    print()

    return df_imp


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3c  —  SAVE RESULTS CSV
# ═════════════════════════════════════════════════════════════════════════════

def save_results_csv(
    res_A: List[Dict],
    res_B: List[Dict],
    res_C: List[Dict],
    path:  str | Path = "results_summary.csv",
) -> pd.DataFrame:
    """Flatten all arm × model results to a tidy CSV."""
    rows = []
    for arm_lbl, arm_res in zip(ARM_LABELS, [res_A, res_B, res_C]):
        for r in arm_res:
            rows.append({
                "arm":      arm_lbl,
                "model":    r["model"],
                "mean_auc": round(r["mean_auc"], 4),
                "std_auc":  round(r["std_auc"],  4),
                "time_s":   round(r["training_time_s"], 2),
                **{f"fold_{i+1}": round(v, 4)
                   for i, v in enumerate(r["auc_scores"])},
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info("Results CSV saved → %s", path)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3d  —  DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def _style_ax(ax: plt.Axes, title: str = "", grid_axis: str = "y") -> None:
    ax.set_facecolor(PALETTE["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["border"])
        sp.set_linewidth(1.1)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=12,
                     fontweight="bold", pad=7, fontfamily="monospace")
    if grid_axis:
        ax.grid(axis=grid_axis, color=PALETTE["gridl"], lw=0.9, alpha=0.9)
        ax.set_axisbelow(True)


def _feat_color(name: str, clin_set: set) -> str:
    """Blue for clinical features, amber for any MRI-derived feature."""
    return PALETTE["armA"] if name in clin_set else PALETTE["armB"]


def make_dashboard(
    res_A:      List[Dict],
    res_B:      List[Dict],
    res_C:      List[Dict],
    X_comb:     np.ndarray,
    y:          np.ndarray,
    feat_comb:  List[str],
    feat_clin:  List[str],
    out:        str | Path = "ml_combined_dashboard.png",
) -> str:
    """
    Build and save a 6-panel comparison dashboard.

    Panels
    ------
    A  Grouped bar  — mean AUC by model × arm
    B  Heat-map     — training time (log scale)
    C  Box + swarm  — fold AUC distributions
    D  Bar          — σ AUC stability per model × arm
    E  Radar        — normalised AUC / speed / stability for best model per arm
    F  Horizontal bar — top-20 feature importances (combined arm, RF MDI)
    """
    arms   = [res_A, res_B, res_C]
    nmod   = len(MODEL_NAMES)
    auc_m  = np.array([[r["mean_auc"] for r in a] for a in arms])
    std_m  = np.array([[r["std_auc"]  for r in a] for a in arms])
    tim_m  = np.array([[r["training_time_s"] for r in a] for a in arms])

    fi_names, fi_vals = compute_feature_importance(X_comb, y, feat_comb)
    clin_set = set(feat_clin)

    # ── figure skeleton ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 20), facecolor=PALETTE["bg"])
    ogs = GridSpec(4, 1, figure=fig, hspace=0.58, top=0.93, bottom=0.04,
                   left=0.05, right=0.97,
                   height_ratios=[0.07, 1, 1, 1])

    # ── title ─────────────────────────────────────────────────────────────────
    ax_t = fig.add_subplot(ogs[0])
    ax_t.set_facecolor(PALETTE["bg"])
    for sp in ax_t.spines.values():
        sp.set_visible(False)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    ax_t.text(
        0.5, 0.78, "TOM500  ·  THREE-ARM ML COMPARISON",
        ha="center", transform=ax_t.transAxes,
        fontsize=26, fontweight="900",
        color=PALETTE["text"], fontfamily="monospace",
    )
    ax_t.text(
        0.5, 0.20,
        "ARM A: Clinical features only   ·   ARM B: MRI imaging features   ·   "
        "ARM C: Clinical + MRI combined   |   Nested 5-outer / 3-inner CV",
        ha="center", transform=ax_t.transAxes,
        fontsize=14, color=PALETTE["muted"], fontfamily="monospace",
    )
    for i, (lbl, col) in enumerate(zip(ARM_LABELS, ARM_COLORS)):
        ax_t.text(0.27 + i * 0.155, 0.02, f"■  {lbl}",
                  ha="center", transform=ax_t.transAxes,
                  fontsize=12, color=col,
                  fontfamily="monospace", fontweight="bold")

    x   = np.arange(nmod)
    w   = 0.24
    off = [-w, 0, w]

    # ── (A) grouped AUC bar ───────────────────────────────────────────────────
    gs1   = GridSpecFromSubplotSpec(1, 2, subplot_spec=ogs[1],
                                    wspace=0.30, width_ratios=[3, 2])
    ax_a  = fig.add_subplot(gs1[0])
    _style_ax(ax_a, "(A)  Mean AUC by Model & Arm")

    for ai, (arm, col, lbl) in enumerate(zip(arms, ARM_COLORS, ARM_LABELS)):
        vals = [r["mean_auc"] for r in arm]
        errs = [r["std_auc"]  for r in arm]
        bars = ax_a.bar(
            x + off[ai], vals, w * 0.9, color=col, alpha=0.88,
            yerr=errs, capsize=3,
            error_kw=dict(ecolor=PALETTE["muted"], elinewidth=1.2, capthick=1.2),
            label=lbl,
        )
        for b, v in zip(bars, vals):
            ax_a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.013,
                      f"{v:.3f}", ha="center", va="bottom",
                      fontsize=9.5, color=PALETTE["text"], fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(MODEL_NAMES, fontsize=14, color=PALETTE["text"])
    ax_a.set_ylim(0.38, 0.90)
    ax_a.set_ylabel("Mean AUC", color=PALETTE["muted"])
    ax_a.axhline(0.5, ls="--", lw=1.2, color=PALETTE["danger"],
                 alpha=0.5, label="Chance (0.5)")
    ax_a.legend(framealpha=0, labelcolor=PALETTE["text"],
                fontsize=12.5, loc="upper left")

    bi, bj = np.unravel_index(auc_m.argmax(), auc_m.shape)
    ax_a.annotate(
        f"BEST\n{auc_m[bi, bj]:.3f}",
        xy=(bj + off[bi], auc_m[bi, bj]),
        xytext=(bj + off[bi] + 0.38, auc_m[bi, bj] + 0.03),
        fontsize=10.5, color=ARM_COLORS[bi], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ARM_COLORS[bi], lw=1.4),
        fontfamily="monospace",
    )

    # ── (B) training-time heat-map ────────────────────────────────────────────
    ax_h = fig.add_subplot(gs1[1])
    _style_ax(ax_h, "(B)  Training Time Heat-map (s)", grid_axis=None)
    im = ax_h.imshow(np.log1p(tim_m), aspect="auto",
                     cmap="YlOrRd", origin="upper")
    ax_h.set_xticks(range(nmod))
    ax_h.set_xticklabels(MODEL_NAMES, fontsize=14, color=PALETTE["text"],
                          rotation=30, ha="right")
    ax_h.set_yticks(range(3))
    ax_h.set_yticklabels(ARM_LABELS, fontsize=14, color=PALETTE["text"])
    for r in range(3):
        for c in range(nmod):
            ax_h.text(c, r, f"{tim_m[r, c]:.1f}s",
                      ha="center", va="center",
                      fontsize=12, color="black", fontweight="bold")
    cb = fig.colorbar(im, ax=ax_h, fraction=0.04, pad=0.04)
    cb.ax.tick_params(colors=PALETTE["muted"], labelsize=10)
    cb.set_label("log(1+s)", color=PALETTE["muted"], fontsize=10.5)

    # ── (C) fold distributions ────────────────────────────────────────────────
    gs2   = GridSpecFromSubplotSpec(1, 2, subplot_spec=ogs[2],
                                    wspace=0.28, width_ratios=[3, 2])
    ax_b  = fig.add_subplot(gs2[0])
    _style_ax(ax_b, "(C)  Fold AUC Distributions  (all arms × models)")

    rng2 = np.random.default_rng(1)
    pos, dat, bcols   = [], [], []
    t_pos, t_lbl      = [], []
    cur = 1.0; gap = 0.35

    for mi, mn in enumerate(MODEL_NAMES):
        t_pos.append(cur + w)
        t_lbl.append(mn)
        for arm, col in zip(arms, ARM_COLORS):
            pos.append(cur)
            dat.append(arm[mi]["auc_scores"])
            bcols.append(col)
            cur += w + 0.03
        cur += gap

    bp = ax_b.boxplot(
        dat, positions=pos, widths=w * 0.82, patch_artist=True,
        medianprops=dict(color=PALETTE["text"], lw=2.0),
        whiskerprops=dict(color=PALETTE["muted"]),
        capprops=dict(color=PALETTE["muted"]),
        flierprops=dict(marker=".", color=PALETTE["muted"], ms=3),
    )
    for patch, col in zip(bp["boxes"], bcols):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
    for p2, sc, col in zip(pos, dat, bcols):
        j = rng2.uniform(-w * 0.28, w * 0.28, len(sc))
        ax_b.scatter(p2 + j, sc, color=col, s=28, zorder=5, alpha=0.88)

    ax_b.set_xticks(t_pos)
    ax_b.set_xticklabels(t_lbl, fontsize=12, color=PALETTE["text"])
    ax_b.set_ylabel("AUC", color=PALETTE["muted"])
    ax_b.set_ylim(0.25, 1.05)
    ax_b.axhline(0.5, ls="--", lw=0.9, color=PALETTE["danger"], alpha=0.4)
    patches = [mpatches.Patch(color=c, label=l, alpha=0.8)
               for c, l in zip(ARM_COLORS, ARM_LABELS)]
    ax_b.legend(handles=patches, framealpha=0,
                labelcolor=PALETTE["text"], fontsize=12, loc="upper left")

    # ── (D) stability bar ─────────────────────────────────────────────────────
    ax_s = fig.add_subplot(gs2[1])
    _style_ax(ax_s, "(D)  Stability  (σ AUC — lower is better)")
    for ai, (arm, col, lbl) in enumerate(zip(arms, ARM_COLORS, ARM_LABELS)):
        stds  = [r["std_auc"] for r in arm]
        bars2 = ax_s.bar(x + off[ai], stds, w * 0.9,
                         color=col, alpha=0.88, label=lbl)
        for b, v in zip(bars2, stds):
            ax_s.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                      f"{v:.3f}", ha="center", va="bottom",
                      fontsize=9.5, color=PALETTE["text"])
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(MODEL_NAMES, fontsize=14, color=PALETTE["text"])
    ax_s.set_ylabel("σ AUC", color=PALETTE["muted"])

    # ── (E) radar ─────────────────────────────────────────────────────────────
    gs3   = GridSpecFromSubplotSpec(1, 2, subplot_spec=ogs[3],
                                    wspace=0.28, width_ratios=[1.6, 2.4])
    ax_r  = fig.add_subplot(gs3[0], polar=True)
    ax_r.set_facecolor(PALETTE["panel"])
    ax_r.spines["polar"].set_color(PALETTE["border"])
    ax_r.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax_r.set_title("(E)  Best-Model Radar per Arm",
                   color=PALETTE["text"], fontsize=12,
                   fontweight="bold", pad=16, fontfamily="monospace")

    cats   = ["Mean AUC", "Speed\n(inv-log t)", "Stability\n(inv-σ)"]
    NC     = len(cats)
    angles = [i / NC * 2 * np.pi for i in range(NC)] + [0]
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(cats, color=PALETTE["text"], fontsize=12)
    ax_r.set_ylim(0, 1)
    ax_r.yaxis.set_tick_params(labelleft=False)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.yaxis.grid(color=PALETTE["border"], lw=0.9)
    ax_r.xaxis.grid(color=PALETTE["border"], lw=0.9)

    all_t   = tim_m.flatten()
    all_sig = std_m.flatten()
    for ai, (arm, col, lbl) in enumerate(zip(arms, ARM_COLORS, ARM_LABELS)):
        bi2 = int(np.argmax([r["mean_auc"] for r in arm]))
        br  = arm[bi2]
        na  = np.clip((br["mean_auc"] - 0.5) / 0.5, 0, 1)
        ns  = 1 - (np.log1p(br["training_time_s"]) - np.log1p(all_t.min())) / \
                  (np.log1p(all_t.max()) - np.log1p(all_t.min()) + 1e-9)
        nst = 1 - (br["std_auc"] - all_sig.min()) / \
                  (all_sig.max() - all_sig.min() + 1e-9)
        vals = [na, ns, nst, na]
        ax_r.plot(angles, vals, color=col, lw=2.5,
                  label=f"{lbl}  [{MODEL_NAMES[bi2]}  {br['mean_auc']:.3f}]")
        ax_r.fill(angles, vals, color=col, alpha=0.13)

    ax_r.legend(loc="upper right", bbox_to_anchor=(1.65, 1.18),
                framealpha=0, labelcolor=PALETTE["text"], fontsize=12)

    # ── (F) feature importance ────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs3[1])
    _style_ax(ax_f, "(F)  Top-20 Feature Importances  (Combined Arm · RF MDI)",
              grid_axis="x")

    fi_colors = [_feat_color(n, clin_set) for n in fi_names]
    yp        = np.arange(len(fi_names))
    ax_f.barh(yp, fi_vals, color=fi_colors, alpha=0.85, edgecolor="none")
    ax_f.set_yticks(yp)
    ax_f.set_yticklabels(fi_names, fontsize=12.5, color=PALETTE["text"])
    ax_f.set_xlabel("MDI Importance", color=PALETTE["muted"])

    cp = mpatches.Patch(color=PALETTE["armA"], alpha=0.85, label="Clinical feature")
    ip = mpatches.Patch(color=PALETTE["armB"], alpha=0.85,
                        label="MRI feature  (shape / intensity / texture / asymmetry / ratio)")
    ax_f.legend(handles=[cp, ip], framealpha=0,
                labelcolor=PALETTE["text"], fontsize=12, loc="lower right")
    ax_f.text(fi_vals[-1] + 0.0003, len(fi_names) - 1,
              f"{fi_vals[-1]:.4f}", va="center",
              fontsize=11.5, color=PALETTE["text"])

    plt.savefig(str(out), dpi=155, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    log.info("Dashboard saved → %s", out)
    return str(out)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    image_dir:       str | Path,
    label_dir:       str | Path,
    clinical_csv:    str | Path,
    mri_csv:         str | Path = "mri_features.csv",
    merged_csv:      str | Path = "merged_features.csv",
    results_csv:     str | Path = "results_summary.csv",
    dashboard_path:  str | Path = "ml_combined_dashboard.png",
    shap_plot:       str | Path = "shap_summary.png",
    shap_csv:        str | Path = "shap_importance.csv",
    n_workers:       int = 1,
    skip_extraction: bool = False,
    merge_how:       str = "inner",
    cas_threshold:   int = 2,
) -> Dict:
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    image_dir        : NIfTI image directory
    label_dir        : NIfTI label directory
    clinical_csv     : clinical_characteristics.csv path
    mri_csv          : output / cache path for MRI features
    merged_csv       : output path for merged dataset
    results_csv      : output path for ML results table
    dashboard_path   : output path for the dashboard PNG
    shap_plot        : output path for the SHAP summary PNG  (Stage 4)
    shap_csv         : output path for the SHAP importance CSV  (Stage 4)
    n_workers        : parallel workers for feature extraction
    skip_extraction  : if True and mri_csv exists, skip Stage 1
    merge_how        : pandas merge strategy
    cas_threshold    : CAS binarisation threshold

    Returns
    -------
    dict with keys: df_mri, df_merged, res_A, res_B, res_C, df_shap
    """
    sep = "═" * 65

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"\n{sep}\n  STAGE 1  —  MRI FEATURE EXTRACTION\n{sep}")
    mri_csv = Path(mri_csv)

    if skip_extraction and mri_csv.exists():
        log.info("Skipping extraction — loading %s", mri_csv)
        df_mri = pd.read_csv(mri_csv)
    else:
        df_mri = build_features_dataset(
            image_dir = image_dir,
            label_dir = label_dir,
            n_workers = n_workers,
            out_csv   = mri_csv,
        )

    mri_feature_cols = [c for c in df_mri.columns if c != "case_id"]
    log.info("MRI features: %d cases × %d features",
             len(df_mri), len(mri_feature_cols))

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print(f"\n{sep}\n  STAGE 2  —  DATA FUSION\n{sep}")
    df_merged = merge_with_clinical(
        clinical_csv = clinical_csv,
        mri_csv      = mri_csv,
        how          = merge_how,
        threshold    = cas_threshold,
        out_csv      = merged_csv,
    )

    if df_merged.empty or "target" not in df_merged.columns:
        raise RuntimeError("Merge produced an empty or targetless DataFrame. "
                           "Check case_id alignment between clinical and MRI files.")

    X_clin, X_mri, X_comb, y, feat_clin, feat_mri, feat_comb = \
        prepare_arm_arrays(df_merged, mri_feature_cols)

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    print(f"\n{sep}\n  STAGE 3  —  THREE-ARM ML COMPARISON\n{sep}")

    res_A = run_arm(X_clin, y, ARM_LABELS[0])
    res_B = run_arm(X_mri,  y, ARM_LABELS[1])
    res_C = run_arm(X_comb, y, ARM_LABELS[2])

    # tabular summary
    print(f"\n{'─'*72}")
    print(f"{'Arm':<16} {'Model':<14} {'AUC':>8} {'±σ':>8} {'Time(s)':>10}")
    print(f"{'─'*72}")
    for lbl, res in zip(ARM_LABELS, [res_A, res_B, res_C]):
        for r in res:
            print(f"{lbl:<16} {r['model']:<14} "
                  f"{r['mean_auc']:>8.4f} {r['std_auc']:>8.4f} "
                  f"{r['training_time_s']:>10.2f}")
        print(f"{'─'*72}")

    save_results_csv(res_A, res_B, res_C, results_csv)
    make_dashboard(res_A, res_B, res_C,
                   X_comb, y, feat_comb, feat_clin,
                   out=dashboard_path)

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    print(f"\n{sep}\n  STAGE 4  —  SHAP EXPLAINABILITY\n{sep}")
    df_shap = run_shap_analysis(
        X              = X_comb,
        y              = y,
        feature_names  = feat_comb,
        output_path    = shap_plot,
        importance_csv = shap_csv,
    )

    print(f"\n✓  All outputs written.")
    return dict(
        df_mri=df_mri, df_merged=df_merged,
        res_A=res_A, res_B=res_B, res_C=res_C,
        df_shap=df_shap,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TOM500 full ML pipeline (extraction → fusion → comparison)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image-dir",  required=True,
                   help="Directory containing <case_id>.nii.gz MRI volumes")
    p.add_argument("--label-dir",  required=True,
                   help="Directory containing <case_id>.nii.gz label masks")
    p.add_argument("--clinical",   required=True,
                   help="Path to clinical_characteristics.csv")
    p.add_argument("--mri-csv",    default="mri_features.csv",
                   help="Output (or cache) path for MRI features")
    p.add_argument("--merged-csv", default="merged_features.csv",
                   help="Output path for merged clinical + MRI dataset")
    p.add_argument("--results-csv",default="results_summary.csv",
                   help="Output path for ML results table")
    p.add_argument("--dashboard",  default="ml_combined_dashboard.png",
                   help="Output path for comparison dashboard PNG")
    p.add_argument("--shap-plot",  default="shap_summary.png",
                   help="Output path for SHAP beeswarm summary PNG (Stage 4)")
    p.add_argument("--shap-csv",   default="shap_importance.csv",
                   help="Output path for SHAP mean |SHAP| importance CSV (Stage 4)")
    p.add_argument("--workers",    type=int, default=1,
                   help="Parallel worker processes for feature extraction")
    p.add_argument("--skip-extraction", action="store_true",
                   help="Skip Stage 1 if --mri-csv already exists")
    p.add_argument("--merge-how",  default="inner",
                   choices=["inner", "left", "right", "outer"],
                   help="Pandas merge strategy for clinical + MRI join")
    p.add_argument("--cas-threshold", type=int, default=2,
                   help="CAS binarisation cutoff (active = CAS >= threshold)")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.getLogger().setLevel(args.log_level)

    run_pipeline(
        image_dir       = args.image_dir,
        label_dir       = args.label_dir,
        clinical_csv    = args.clinical,
        mri_csv         = args.mri_csv,
        merged_csv      = args.merged_csv,
        results_csv     = args.results_csv,
        dashboard_path  = args.dashboard,
        shap_plot       = args.shap_plot,
        shap_csv        = args.shap_csv,
        n_workers       = args.workers,
        skip_extraction = args.skip_extraction,
        merge_how       = args.merge_how,
        cas_threshold   = args.cas_threshold,
    )


if __name__ == "__main__":
    image_dir = "data/train/image"
    label_dir = "data/train/label"
    clinical = "clinical_characteristics.csv"

    run_pipeline(image_dir, label_dir, clinical)