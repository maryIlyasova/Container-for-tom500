"""
SHAP Importance Analysis Pipeline
==================================
Loads SHAP importance data, visualises feature importance, performs
feature selection, groups features by anatomical structure, and
produces a suite of publication-ready plots and summary tables.

Outputs
-------
  shap_top20_bar.png           Top-20 features bar chart
  shap_cumulative.png          Cumulative importance curve
  shap_full_distribution.png   Full ranked feature distribution
  shap_grouped_importance.png  Importance aggregated by anatomy group
  shap_filtered_features.csv   Features surviving both selection filters
  shap_summary_table.csv       Per-feature summary with group labels
  shap_group_summary.csv       Per-group aggregated statistics

Usage
-----
  python shap_analysis_pipeline.py                          # default paths
  python shap_analysis_pipeline.py --csv shap_importance.csv --out-dir results/
  python shap_analysis_pipeline.py --threshold 0.0005 --cumulative-cutoff 0.90
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shap_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# PALETTE  (matches tom500_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = dict(
    bg       = "#ffffff",
    panel    = "#e6e6ff",
    border   = "#2C3048",
    text     = "#000000",
    muted    = "#45506a",
    gridl    = "#484F67",
    clinical = "#0361f8",   # blue
    mri      = "#f7b500",   # amber
    combined = "#00fc8b",   # teal
    danger   = "#fb0244",
    highlight= "#f8f800",   # yellow accent
)

# One colour per anatomical group (order matches GROUP_MAP keys)
GROUP_COLORS = [
    "#4d8df5",  # eyeball       – blue
    "#f5c94d",  # fat           – amber
    "#4df5a8",  # optic nerve   – teal
    "#f54d7a",  # muscle        – pink
    "#b45ef4",  # lacrimal      – purple
    "#f5804d",  # superior rect – orange
    "#4df5f5",  # inferior rect – cyan
    "#f54df5",  # medial rect   – magenta
    "#8df54d",  # lateral rect  – lime
    "#f5f54d",  # clinical      – yellow
]


# ─────────────────────────────────────────────────────────────────────────────
# ANATOMICAL GROUP DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Maps group label → list of feature-name prefixes (checked with startswith)
GROUP_MAP: Dict[str, List[str]] = {
    "Eyeball":           ["eyeball_"],
    "Fat":               ["fat_"],
    "Optic Nerve":       ["optic_nerve_"],
    "Muscle (generic)":  ["muscle_"],
    "Lacrimal Gland":    ["lacrimal_gland_"],
    "Superior Rectus":   ["superior_rectus_"],
    "Inferior Rectus":   ["inferior_rectus_"],
    "Medial Rectus":     ["medial_rectus_"],
    "Lateral Rectus":    ["lateral_rectus_"],
    "Clinical":          ["Age", "Sex_enc", "SmokingIndex", "LogSmoke",
                          "DiseaseDuration", "IsSmoker",
                          "StudyYear", "StudyMonth",
                          "ratio_"],  # ratio features are derived from MRI
}

# Features that don't match any prefix land in "Other"
OTHER_LABEL = "Other / Ratio"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_shap_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load and validate the SHAP importance CSV.

    Expected columns: ``feature``, ``mean_abs_shap``
    Rows are returned sorted by importance descending.

    Parameters
    ----------
    csv_path : path to shap_importance.csv

    Returns
    -------
    pd.DataFrame with columns [feature, mean_abs_shap, rank]
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"SHAP CSV not found: {path}")

    df = pd.read_csv(path)

    required = {"feature", "mean_abs_shap"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce")
    n_nan = df["mean_abs_shap"].isna().sum()
    if n_nan:
        log.warning("Dropping %d rows with non-numeric mean_abs_shap.", n_nan)
        df = df.dropna(subset=["mean_abs_shap"])

    df = (df
          .sort_values("mean_abs_shap", ascending=False)
          .reset_index(drop=True))
    df["rank"] = df.index + 1

    log.info("Loaded %d features from %s", len(df), path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE GROUPING
# ─────────────────────────────────────────────────────────────────────────────

def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``group`` column by matching each feature name against GROUP_MAP
    prefixes.  Unmatched features are labelled OTHER_LABEL.

    Parameters
    ----------
    df : DataFrame with a ``feature`` column

    Returns
    -------
    df with new ``group`` column (in-place copy)
    """
    df = df.copy()

    def _label(feat: str) -> str:
        for group, prefixes in GROUP_MAP.items():
            for pfx in prefixes:
                if feat.startswith(pfx) or feat == pfx:
                    return group
        return OTHER_LABEL

    df["group"] = df["feature"].apply(_label)
    return df


def group_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate importance statistics per anatomical group.

    Returns
    -------
    pd.DataFrame with columns:
        group, n_features, total_shap, mean_shap, max_shap, pct_total
    """
    total = df["mean_abs_shap"].sum()
    agg   = (df.groupby("group")["mean_abs_shap"]
               .agg(n_features="count",
                    total_shap="sum",
                    mean_shap="mean",
                    max_shap="max")
               .reset_index())
    agg["pct_total"] = agg["total_shap"] / total * 100
    agg = agg.sort_values("total_shap", ascending=False).reset_index(drop=True)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_by_threshold(
    df:        pd.DataFrame,
    threshold: float = 0.001,
) -> pd.DataFrame:
    """
    Keep features whose mean |SHAP| value ≥ ``threshold``.

    Parameters
    ----------
    df        : full ranked feature DataFrame
    threshold : minimum mean |SHAP| to retain

    Returns
    -------
    Filtered DataFrame
    """
    mask   = df["mean_abs_shap"] >= threshold
    result = df[mask].copy()
    log.info(
        "Threshold filter (≥ %.4f): %d / %d features retained (%.1f%% reduction).",
        threshold, len(result), len(df),
        (1 - len(result) / len(df)) * 100,
    )
    return result


def select_by_cumulative(
    df:     pd.DataFrame,
    cutoff: float = 0.95,
) -> pd.DataFrame:
    """
    Keep the smallest set of top features that together account for
    at least ``cutoff`` fraction of total cumulative importance.

    Parameters
    ----------
    df     : full ranked feature DataFrame (must be sorted descending)
    cutoff : cumulative importance fraction to preserve  (0 < cutoff ≤ 1)

    Returns
    -------
    Filtered DataFrame with an added ``cumulative_pct`` column
    """
    total = df["mean_abs_shap"].sum()
    df    = df.copy()
    df["cumulative_pct"] = df["mean_abs_shap"].cumsum() / total

    mask   = df["cumulative_pct"].shift(1, fill_value=0) < cutoff
    result = df[mask].copy()
    log.info(
        "Cumulative filter (%.0f%%): %d / %d features retained (%.1f%% reduction).",
        cutoff * 100, len(result), len(df),
        (1 - len(result) / len(df)) * 100,
    )
    return result


def combined_selection(
    df:        pd.DataFrame,
    threshold: float = 0.001,
    cutoff:    float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply both selection strategies and return their intersection.

    Returns
    -------
    df_thresh  : threshold-filtered
    df_cumul   : cumulative-filtered
    df_combined: intersection (features surviving both)
    """
    df_thresh  = select_by_threshold(df, threshold)
    df_cumul   = select_by_cumulative(df, cutoff)

    combined_feats = set(df_thresh["feature"]) & set(df_cumul["feature"])
    df_combined    = df[df["feature"].isin(combined_feats)].copy()

    log.info(
        "Combined selection: %d features survive both filters "
        "(%.1f%% reduction from original %d).",
        len(df_combined),
        (1 - len(df_combined) / len(df)) * 100,
        len(df),
    )
    return df_thresh, df_cumul, df_combined


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COMPARISON STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compare_feature_sets(
    df_full:     pd.DataFrame,
    df_filtered: pd.DataFrame,
    label:       str = "Combined filter",
) -> pd.DataFrame:
    """
    Print and return a comparison table of the original vs filtered set.

    Parameters
    ----------
    df_full     : original full feature DataFrame
    df_filtered : post-selection DataFrame
    label       : name of the selection strategy for display

    Returns
    -------
    pd.DataFrame with one row per metric
    """
    total_shap   = df_full["mean_abs_shap"].sum()
    kept_shap    = df_filtered["mean_abs_shap"].sum()

    rows = {
        "Original features":          len(df_full),
        "Retained features":          len(df_filtered),
        "Features removed":           len(df_full) - len(df_filtered),
        "% feature reduction":        round((1 - len(df_filtered) / len(df_full)) * 100, 2),
        "Total SHAP (original)":      round(total_shap, 6),
        "Total SHAP (retained)":      round(kept_shap, 6),
        "% SHAP captured":            round(kept_shap / total_shap * 100, 2),
        "Mean SHAP (original)":       round(df_full["mean_abs_shap"].mean(), 6),
        "Mean SHAP (retained)":       round(df_filtered["mean_abs_shap"].mean(), 6),
        "Min SHAP (retained)":        round(df_filtered["mean_abs_shap"].min(), 6),
    }

    df_cmp = pd.DataFrame({"metric": list(rows.keys()),
                            "value":  list(rows.values())})

    print(f"\n{'═'*58}")
    print(f"  FEATURE SET COMPARISON  —  {label}")
    print(f"{'═'*58}")
    for _, row in df_cmp.iterrows():
        print(f"  {row['metric']:<36} {row['value']:>16}")
    print(f"{'═'*58}\n")

    return df_cmp


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark_style(ax: plt.Axes, title: str = "", grid_axis: str = "x") -> None:
    """Apply the standard dark dashboard style to an Axes object."""
    ax.set_facecolor(PALETTE["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["border"])
        sp.set_linewidth(1.1)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=14,
                     fontweight="bold", pad=9, fontfamily="monospace")
    if grid_axis:
        ax.grid(axis=grid_axis, color=PALETTE["gridl"], lw=0.85, alpha=0.9)
        ax.set_axisbelow(True)


def _savefig(fig: plt.Figure, path: Path, dpi: int = 160) -> None:
    """Save figure with consistent dark background and close it."""
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    log.info("Plot saved → %s", path)


def _figure(width: float = 14, height: float = 8) -> plt.Figure:
    """Create a pre-styled dark figure."""
    fig = plt.figure(figsize=(width, height), facecolor=PALETTE["bg"])
    fig.patch.set_facecolor(PALETTE["bg"])
    return fig


def _group_color(group: str, group_list: List[str]) -> str:
    idx = group_list.index(group) if group in group_list else -1
    return GROUP_COLORS[idx % len(GROUP_COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOT A  —  TOP-20 BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_top20(
    df:       pd.DataFrame,
    out_path: Path,
    top_n:    int = 20,
) -> None:
    """
    Horizontal bar chart of the top ``top_n`` features, coloured by
    anatomical group.

    Parameters
    ----------
    df       : full ranked DataFrame (must have ``group`` column)
    out_path : destination PNG path
    top_n    : number of features to display
    """
    top      = df.head(top_n).iloc[::-1].reset_index(drop=True)  # plot bottom→top
    groups   = sorted(df["group"].unique().tolist())
    colors   = [_group_color(g, groups) for g in top["group"]]

    fig  = _figure(14, max(7, top_n * 0.38))
    ax   = fig.add_subplot(111)
    _apply_dark_style(ax, f"Top {top_n} Features by Mean |SHAP| Value", grid_axis="x")

    ypos = np.arange(len(top))
    bars = ax.barh(ypos, top["mean_abs_shap"], color=colors,
                   alpha=0.88, edgecolor="none", height=0.7)

    # Value labels
    x_max = top["mean_abs_shap"].max()
    for bar, val in zip(bars, top["mean_abs_shap"]):
        ax.text(bar.get_width() + x_max * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", ha="left",
                fontsize=11.5, color=PALETTE["text"])

    ax.set_yticks(ypos)
    ax.set_yticklabels(top["feature"], fontsize=12, color=PALETTE["text"])
    ax.set_xlabel("Mean |SHAP| value", color=PALETTE["muted"])
    ax.set_xlim(0, x_max * 1.22)

    # Group legend
    unique_groups = top["group"].unique().tolist()
    patches = [mpatches.Patch(color=_group_color(g, groups), label=g, alpha=0.88)
               for g in unique_groups]
    ax.legend(handles=patches, framealpha=0, labelcolor=PALETTE["text"],
              fontsize=12.5, loc="lower right")

    fig.text(0.5, 0.01,
             "Feature importance derived from Random Forest · SHAP TreeExplainer",
             ha="center", fontsize=11.5, color=PALETTE["muted"],
             fontfamily="monospace")

    _savefig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOT B  —  CUMULATIVE IMPORTANCE CURVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative(
    df:        pd.DataFrame,
    out_path:  Path,
    threshold: float = 0.001,
    cutoff:    float = 0.95,
) -> None:
    """
    Line chart showing cumulative importance fraction as features are
    added in descending order.  Annotates both selection thresholds.

    Parameters
    ----------
    df        : full ranked DataFrame
    out_path  : destination PNG path
    threshold : mean |SHAP| threshold (for vertical marker)
    cutoff    : cumulative fraction cutoff (for horizontal marker)
    """
    total   = df["mean_abs_shap"].sum()
    cum_pct = df["mean_abs_shap"].cumsum() / total * 100

    # Index where threshold filter kicks in (first rank below threshold)
    thresh_mask = df["mean_abs_shap"] >= threshold
    n_thresh    = thresh_mask.sum()

    # Index where cumulative cutoff is reached
    n_cumul = int((cum_pct / 100 < cutoff).sum()) + 1
    n_cumul = min(n_cumul, len(df))

    fig = _figure(13, 7)
    ax  = fig.add_subplot(111)
    _apply_dark_style(ax, "Cumulative Feature Importance", grid_axis="y")

    x = np.arange(1, len(df) + 1)
    ax.plot(x, cum_pct, color=PALETTE["combined"], lw=2.2, zorder=4)
    ax.fill_between(x, cum_pct, alpha=0.12, color=PALETTE["combined"], zorder=3)

    # Horizontal reference line — cumulative cutoff
    ax.axhline(cutoff * 100, color=PALETTE["clinical"], lw=1.4, ls="--",
               alpha=0.8, label=f"{cutoff*100:.0f}% importance cutoff")
    ax.axvline(n_cumul, color=PALETTE["clinical"], lw=1.2, ls=":",
               alpha=0.7)
    ax.text(n_cumul + len(df) * 0.01, cutoff * 100 - 4,
            f"{n_cumul} features\n→ {cutoff*100:.0f}% importance",
            color=PALETTE["clinical"], fontsize=12, fontfamily="monospace")

    # Vertical reference line — threshold filter
    if n_thresh < len(df):
        ax.axvline(n_thresh, color=PALETTE["mri"], lw=1.2, ls="--",
                   alpha=0.8, label=f"Threshold filter (≥ {threshold})")
        cum_at_thresh = float(cum_pct.iloc[n_thresh - 1])
        ax.text(n_thresh + len(df) * 0.01, cum_at_thresh - 6,
                f"{n_thresh} features\n→ {cum_at_thresh:.1f}% importance",
                color=PALETTE["mri"], fontsize=12, fontfamily="monospace")

    ax.set_xlabel("Number of features (ranked by importance)", color=PALETTE["muted"])
    ax.set_ylabel("Cumulative importance (%)", color=PALETTE["muted"])
    ax.set_ylim(0, 103)
    ax.set_xlim(0, len(df) + 1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(framealpha=0, labelcolor=PALETTE["text"], fontsize=12.5,
              loc="lower right")

    _savefig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOT C  —  FULL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """
    Two-panel plot: (left) full ranked bar distribution with log y-axis;
    (right) histogram of importance values.

    Parameters
    ----------
    df       : full ranked DataFrame (must have ``group`` column)
    out_path : destination PNG path
    """
    groups = sorted(df["group"].unique().tolist())
    colors = [_group_color(g, groups) for g in df["group"]]

    fig = _figure(18, 7)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.32, left=0.06, right=0.97,
                   top=0.88, bottom=0.10)

    # ── Left: ranked bar ──────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    _apply_dark_style(ax_bar, "Full Feature Importance Distribution (log scale)",
                      grid_axis="y")
    ax_bar.bar(np.arange(len(df)), df["mean_abs_shap"],
               color=colors, alpha=0.80, edgecolor="none", width=1.0)
    ax_bar.set_yscale("log")
    ax_bar.set_xlabel("Feature rank (1 = most important)", color=PALETTE["muted"])
    ax_bar.set_ylabel("Mean |SHAP| value (log)", color=PALETTE["muted"])
    ax_bar.set_xlim(-1, len(df))

    patches = [mpatches.Patch(color=_group_color(g, groups), label=g, alpha=0.85)
               for g in groups]
    ax_bar.legend(handles=patches, framealpha=0, labelcolor=PALETTE["text"],
                  fontsize=11, loc="upper right", ncol=2)

    # ── Right: histogram ──────────────────────────────────────────────────────
    ax_hist = fig.add_subplot(gs[1])
    _apply_dark_style(ax_hist, "Distribution of Mean |SHAP| Values", grid_axis="y")

    vals = df["mean_abs_shap"].values
    bins = np.logspace(np.log10(max(vals.min(), 1e-9)), np.log10(vals.max()), 40)
    ax_hist.hist(vals, bins=bins, color=PALETTE["combined"],
                 alpha=0.80, edgecolor=PALETTE["border"], lw=0.5)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Mean |SHAP| value (log)", color=PALETTE["muted"])
    ax_hist.set_ylabel("Number of features", color=PALETTE["muted"])

    # Median & mean annotation
    med = np.median(vals)
    mn  = vals.mean()
    ax_hist.axvline(med, color=PALETTE["mri"], lw=1.5, ls="--",
                    label=f"Median: {med:.5f}")
    ax_hist.axvline(mn,  color=PALETTE["danger"], lw=1.5, ls="--",
                    label=f"Mean: {mn:.5f}")
    ax_hist.legend(framealpha=0, labelcolor=PALETTE["text"], fontsize=12)

    fig.suptitle("SHAP Importance · Full Feature Set",
                 color=PALETTE["text"], fontsize=15, fontweight="bold",
                 fontfamily="monospace", y=0.96)

    _savefig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLOT D  —  GROUPED IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_grouped_importance(
    df_groups: pd.DataFrame,
    out_path:  Path,
) -> None:
    """
    Three-panel plot showing group-level importance:
      (A) Total SHAP per group (horizontal bar)
      (B) % of total importance per group (pie-like bar)
      (C) Feature count per group

    Parameters
    ----------
    df_groups : output of group_importance()
    out_path  : destination PNG path
    """
    groups  = df_groups["group"].tolist()
    n       = len(groups)
    colors  = [_group_color(g, groups) for g in groups]
    ypos    = np.arange(n)

    fig = _figure(20, max(6, n * 0.55))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.36,
                   left=0.22, right=0.97, top=0.88, bottom=0.08)

    # ── (A) Total SHAP ────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    _apply_dark_style(ax_a, "(A)  Total SHAP per Anatomy Group", grid_axis="x")
    bars_a = ax_a.barh(ypos, df_groups["total_shap"],
                       color=colors, alpha=0.88, edgecolor="none", height=0.65)
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels(groups, fontsize=12.5, color=PALETTE["text"])
    ax_a.set_xlabel("Sum of mean |SHAP|", color=PALETTE["muted"])
    x_max = df_groups["total_shap"].max()
    for bar, val in zip(bars_a, df_groups["total_shap"]):
        ax_a.text(bar.get_width() + x_max * 0.02,
                  bar.get_y() + bar.get_height() / 2,
                  f"{val:.4f}", va="center", fontsize=11.5,
                  color=PALETTE["text"])
    ax_a.set_xlim(0, x_max * 1.28)

    # ── (B) % of total ────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    _apply_dark_style(ax_b, "(B)  % of Total Importance", grid_axis="x")
    bars_b = ax_b.barh(ypos, df_groups["pct_total"],
                       color=colors, alpha=0.88, edgecolor="none", height=0.65)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([], fontsize=12)
    ax_b.set_xlabel("% of total SHAP importance", color=PALETTE["muted"])
    ax_b.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    p_max = df_groups["pct_total"].max()
    for bar, val in zip(bars_b, df_groups["pct_total"]):
        ax_b.text(bar.get_width() + p_max * 0.02,
                  bar.get_y() + bar.get_height() / 2,
                  f"{val:.1f}%", va="center", fontsize=11.5,
                  color=PALETTE["text"])
    ax_b.set_xlim(0, p_max * 1.28)

    # ── (C) Feature count ─────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[2])
    _apply_dark_style(ax_c, "(C)  Feature Count per Group", grid_axis="x")
    bars_c = ax_c.barh(ypos, df_groups["n_features"],
                       color=colors, alpha=0.65, edgecolor="none", height=0.65)
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels([], fontsize=12)
    ax_c.set_xlabel("Number of features", color=PALETTE["muted"])
    n_max = df_groups["n_features"].max()
    for bar, val in zip(bars_c, df_groups["n_features"]):
        ax_c.text(bar.get_width() + n_max * 0.02,
                  bar.get_y() + bar.get_height() / 2,
                  str(int(val)), va="center", fontsize=12,
                  color=PALETTE["text"], fontweight="bold")
    ax_c.set_xlim(0, n_max * 1.25)

    fig.suptitle("SHAP Importance · Anatomical Group Summary",
                 color=PALETTE["text"], fontsize=15, fontweight="bold",
                 fontfamily="monospace", y=0.96)

    _savefig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_filtered_features(df_filtered: pd.DataFrame, out_path: Path) -> None:
    """
    Save the filtered feature list with rank, group, and importance value.

    Parameters
    ----------
    df_filtered : post-selection DataFrame
    out_path    : destination CSV path
    """
    cols = ["rank", "feature", "group", "mean_abs_shap"]
    cols = [c for c in cols if c in df_filtered.columns]
    df_filtered[cols].to_csv(str(out_path), index=False)
    log.info("Filtered features saved → %s  (%d rows)", out_path, len(df_filtered))


def save_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a full per-feature summary table including cumulative importance,
    group label, and selection flags.

    Parameters
    ----------
    df       : full ranked DataFrame with ``group`` column
    out_path : destination CSV path
    """
    df = df.copy()
    total       = df["mean_abs_shap"].sum()
    df["cumulative_pct"] = df["mean_abs_shap"].cumsum() / total * 100
    df["pct_of_total"]   = df["mean_abs_shap"] / total * 100

    cols = ["rank", "feature", "group", "mean_abs_shap",
            "pct_of_total", "cumulative_pct"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(str(out_path), index=False)
    log.info("Summary table saved → %s  (%d rows)", out_path, len(df))


def save_group_summary(df_groups: pd.DataFrame, out_path: Path) -> None:
    """Save the group-level aggregated statistics."""
    df_groups.to_csv(str(out_path), index=False)
    log.info("Group summary saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    csv_path:      str | Path = "shap_importance.csv",
    out_dir:       str | Path = ".",
    threshold:     float = 0.001,
    cumul_cutoff:  float = 0.95,
    top_n:         int   = 20,
    full_dist:     bool  = True,
) -> Dict:
    """
    Execute the complete SHAP analysis pipeline.

    Parameters
    ----------
    csv_path     : path to shap_importance.csv
    out_dir      : directory for all outputs
    threshold    : minimum mean |SHAP| for threshold-based filter
    cumul_cutoff : cumulative importance fraction to retain
    top_n        : number of top features for bar chart
    full_dist    : whether to produce the full distribution plot

    Returns
    -------
    dict with keys: df, df_groups, df_thresh, df_cumul, df_combined
    """
    sep = "═" * 60
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n{sep}\n  STAGE 1  —  LOADING DATA\n{sep}")
    df = load_shap_csv(csv_path)
    df = assign_groups(df)

    print(f"  Total features: {len(df)}")
    print(f"  Total SHAP:     {df['mean_abs_shap'].sum():.6f}")
    print(f"  Groups found:   {sorted(df['group'].unique())}")

    # ── 2. Visualise ─────────────────────────────────────────────────────────
    print(f"\n{sep}\n  STAGE 2  —  VISUALISATION\n{sep}")
    plot_top20(df, out / "shap_top20_bar.png", top_n=top_n)
    plot_cumulative(df, out / "shap_cumulative.png",
                    threshold=threshold, cutoff=cumul_cutoff)
    if full_dist:
        plot_full_distribution(df, out / "shap_full_distribution.png")

    # ── 3. Feature selection ──────────────────────────────────────────────────
    print(f"\n{sep}\n  STAGE 3  —  FEATURE SELECTION\n{sep}")
    df_thresh, df_cumul, df_combined = combined_selection(
        df, threshold=threshold, cutoff=cumul_cutoff)

    compare_feature_sets(df, df_thresh,   label=f"Threshold (≥ {threshold})")
    compare_feature_sets(df, df_cumul,    label=f"Cumulative ({cumul_cutoff*100:.0f}%)")
    compare_feature_sets(df, df_combined, label="Combined (intersection)")

    # ── 4. Group analysis ─────────────────────────────────────────────────────
    print(f"\n{sep}\n  STAGE 4  —  ANATOMICAL GROUP ANALYSIS\n{sep}")
    df_groups = group_importance(df)

    print(f"  {'Group':<24} {'N':>5} {'Total SHAP':>12} {'% Total':>9}")
    print(f"  {'─'*24} {'─'*5} {'─'*12} {'─'*9}")
    for _, row in df_groups.iterrows():
        print(f"  {row['group']:<24} {int(row['n_features']):>5} "
              f"{row['total_shap']:>12.5f} {row['pct_total']:>8.1f}%")

    plot_grouped_importance(df_groups, out / "shap_grouped_importance.png")

    # ── 5. Save CSVs ──────────────────────────────────────────────────────────
    print(f"\n{sep}\n  STAGE 5  —  SAVING OUTPUTS\n{sep}")
    df_combined_annotated = df_combined.copy()
    if "group" not in df_combined_annotated.columns:
        df_combined_annotated = assign_groups(df_combined_annotated)

    save_filtered_features(df_combined_annotated, out / "shap_filtered_features.csv")
    save_summary_table(df,        out / "shap_summary_table.csv")
    save_group_summary(df_groups, out / "shap_group_summary.csv")

    print(f"\n✓  Pipeline complete.  All outputs written to: {out.resolve()}")
    print(f"   Plots:  shap_top20_bar.png  |  shap_cumulative.png"
          + ("  |  shap_full_distribution.png" if full_dist else "")
          + "  |  shap_grouped_importance.png")
    print(f"   CSVs:   shap_filtered_features.csv  |  shap_summary_table.csv"
          "  |  shap_group_summary.csv\n")

    return dict(
        df          = df,
        df_groups   = df_groups,
        df_thresh   = df_thresh,
        df_cumul    = df_cumul,
        df_combined = df_combined,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SHAP importance analysis pipeline for TOM500",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",              default="shap_importance.csv",
                   help="Path to shap_importance.csv")
    p.add_argument("--out-dir",          default=".",
                   help="Output directory for all plots and CSVs")
    p.add_argument("--threshold",        type=float, default=0.001,
                   help="Minimum mean |SHAP| for threshold filter")
    p.add_argument("--cumulative-cutoff",type=float, default=0.95,
                   help="Cumulative importance fraction to retain (0–1)")
    p.add_argument("--top-n",            type=int,   default=20,
                   help="Number of top features to show in bar chart")
    p.add_argument("--no-full-dist",     action="store_true",
                   help="Skip the full distribution plot (faster)")
    p.add_argument("--log-level",        default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.getLogger().setLevel(args.log_level)

    run_pipeline(
        csv_path     = args.csv,
        out_dir      = args.out_dir,
        threshold    = args.threshold,
        cumul_cutoff = args.cumulative_cutoff,
        top_n        = args.top_n,
        full_dist    = not args.no_full_dist,
    )


if __name__ == "__main__":
    main()
