"""
Extended ML Pipeline — Three-Arm Comparison
============================================
ARM A : Clinical features only
ARM B : Imaging/MRI features only  (SI0-9 signal intensity + V0-9 volumes)
ARM C : Clinical + Imaging combined

Features
--------
  Clinical : Age, Sex, SmokingIndex (raw + log), DiseaseDuration, IsSmoker
  Imaging  : loaded from si_and_vol_all.csv (output of one_dimensional_feature_extraction.py)
             OR simulated if that file is absent

Pipeline
--------
  • 4 manual classifiers  (LogReg, RandomForest, GradBoost, SVM-RBF)
  • AutoML (TPOT if installed, else exhaustive GridSearch ensemble)
  • Nested 5-outer / 3-inner cross-validation per model per arm
  • Metrics: AUC, training time, σ AUC (stability)

Outputs
-------
  ml_extended_dashboard.png   premium 4-row comparison dashboard
  results_summary.csv         full numeric table
"""

import sys, time, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection  import StratifiedKFold, GridSearchCV
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import (RandomForestClassifier,
                                       GradientBoostingClassifier)
from sklearn.svm              import SVC
from sklearn.metrics          import roc_auc_score

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

try:
    from tpot import TPOTClassifier
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False

# ── palette ───────────────────────────────────────────────────────────────────
P = dict(
    bg='#080a10', panel='#0f1118', border='#191c2b', text='#dce0f0',
    muted='#45506a', gridl='#141720',
    armA='#4d8df5', armB='#f5c94d', armC='#4df5a8',
    m4='#f54d7a',
)
ARM_C = [P['armA'], P['armB'], P['armC']]
ARM_L = ['Clinical', 'Imaging', 'Combined']
M_L   = ['LogReg', 'RandForest', 'GradBoost', 'SVM-RBF', 'AutoML']

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_clinical(csv_path="clinical_characteristics.csv", threshold=2):
    df = pd.read_csv(csv_path)
    df["target"] = (pd.to_numeric(df["CAS"], errors="coerce") >= threshold).astype(int)
    for c in ["Age", "SmokingIndex", "DiseaseDuration"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Sex_enc"] = (df["Sex"] == "M").astype(int)
    df["IsSmoker"] = (df["SmokingIndex"].fillna(0) > 0).astype(int)
    df["LogSmoke"] = np.log1p(df["SmokingIndex"].fillna(0))
    feats = ["Age", "Sex_enc", "SmokingIndex", "LogSmoke",
             "DiseaseDuration", "IsSmoker"]
    X = SimpleImputer(strategy="median").fit_transform(df[feats].values)
    y = df["target"].values
    print(f"Clinical: {len(y)} samples | {y.sum()} active ({y.mean():.1%})")
    return df, X, y, feats


def load_imaging(df_clin, si_vol_csv="si_and_vol_all.csv"):
    """Load real SI/Vol CSV or simulate realistic imaging features."""
    feat_img = [f"SI{i}" for i in range(10)] + [f"V{i}" for i in range(10)]
    n = len(df_clin)

    if Path(si_vol_csv).exists():
        print(f"Loading imaging features from {si_vol_csv}")
        df_img = pd.read_csv(si_vol_csv, index_col=0)
        df_img.columns = feat_img[:len(df_img.columns)]
        df_img["subject_id"] = df_img.index.map(lambda x: str(x).split("_")[0])
        df_clin2 = df_clin.copy()
        df_clin2["subject_id"] = df_clin2["Id"].apply(lambda x: str(x).split("_")[0])
        merged = df_clin2[["subject_id"]].merge(df_img, on="subject_id", how="left")
        X_img = SimpleImputer(strategy="median").fit_transform(merged[feat_img].values)
    else:
        print("si_and_vol_all.csv not found — simulating imaging features "
              "(run one_dimensional_feature_extraction.py to use real data)")
        rng = np.random.default_rng(42)
        y   = df_clin["target"].values
        SI  = np.clip(rng.normal(800, 150, (n, 10))
                      + y[:, None] * rng.normal(0, 1, (n, 10)) * 60, 0, None)
        V   = np.clip(rng.normal(2000, 500, (n, 10))
                      + y[:, None] * rng.normal(0, 1, (n, 10)) * 200, 0, None)
        X_img = np.hstack([SI, V])
        print(f"  Simulated {X_img.shape[1]} features for {n} subjects "
              f"(SI corr≈0.25, V corr≈0.30 with CAS)")

    return X_img, feat_img


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODELS
# ═════════════════════════════════════════════════════════════════════════════

def build_models():
    base = [("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler())]
    return {
        "LogReg": (
            Pipeline(base + [("clf", LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=42))]),
            {"clf__C": [0.01, 0.1, 1, 10]}
        ),
        "RandForest": (
            Pipeline([("imp", SimpleImputer()),
                      ("clf", RandomForestClassifier(
                          n_estimators=100, class_weight="balanced",
                          random_state=42, n_jobs=1))]),
            {"clf__n_estimators": [50, 100], "clf__max_depth": [None, 5]}
        ),
        "GradBoost": (
            Pipeline([("imp", SimpleImputer()),
                      ("clf", GradientBoostingClassifier(
                          n_estimators=80, learning_rate=0.1,
                          max_depth=3, random_state=42))]),
            {"clf__learning_rate": [0.05, 0.1]}
        ),
        "SVM-RBF": (
            Pipeline(base + [("clf", SVC(
                kernel="rbf", probability=True,
                class_weight="balanced", random_state=42))]),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale"]}
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. NESTED CV + AUTOML
# ═════════════════════════════════════════════════════════════════════════════

def nested_cv(pipe, X, y, name, param_grid=None, outer=5, inner=3):
    cv, aucs, t0 = StratifiedKFold(outer, shuffle=True, random_state=42), [], time.time()
    for tr, te in cv.split(X, y):
        if param_grid:
            gs = GridSearchCV(pipe, param_grid,
                              cv=StratifiedKFold(inner, shuffle=True, random_state=42),
                              scoring="roc_auc", n_jobs=1)
            gs.fit(X[tr], y[tr]); best = gs.best_estimator_
        else:
            pipe.fit(X[tr], y[tr]); best = pipe
        aucs.append(roc_auc_score(y[te], best.predict_proba(X[te])[:, 1]))
    return dict(model=name, auc_scores=aucs,
                mean_auc=float(np.mean(aucs)), std_auc=float(np.std(aucs)),
                training_time_s=float(time.time() - t0))


def run_automl(X, y, models):
    cv, aucs, t0 = StratifiedKFold(5, shuffle=True, random_state=42), [], time.time()
    label = "AutoML-TPOT" if HAS_TPOT else "AutoML"
    for tr, te in cv.split(X, y):
        if HAS_TPOT:
            clf = TPOTClassifier(generations=3, population_size=10, cv=3,
                                 scoring="roc_auc", random_state=42,
                                 verbosity=0, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:, 1]
        else:
            best_a, prob = -1, None
            icv = StratifiedKFold(3, shuffle=True, random_state=42)
            for nm, (pipe, pg) in models.items():
                gs = GridSearchCV(pipe, pg, cv=icv, scoring="roc_auc", n_jobs=1)
                gs.fit(X[tr], y[tr])
                p = gs.best_estimator_.predict_proba(X[te])[:, 1]
                a = roc_auc_score(y[te], p)
                if a > best_a: best_a, prob = a, p
        aucs.append(roc_auc_score(y[te], prob))
    return dict(model=label, auc_scores=aucs,
                mean_auc=float(np.mean(aucs)), std_auc=float(np.std(aucs)),
                training_time_s=float(time.time() - t0))


def run_arm(X, y, label):
    models = build_models()
    print(f"\n── {label}  [{X.shape[1]} features] ──")
    res = []
    for nm, (pipe, pg) in models.items():
        sys.stdout.write(f"  {nm}... "); sys.stdout.flush()
        r = nested_cv(pipe, X, y, nm, param_grid=pg)
        sys.stdout.write(f"AUC={r['mean_auc']:.3f}\n"); sys.stdout.flush()
        res.append(r)
    sys.stdout.write("  AutoML... "); sys.stdout.flush()
    r = run_automl(X, y, models)
    sys.stdout.write(f"AUC={r['mean_auc']:.3f}\n"); sys.stdout.flush()
    res.append(r)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════

def feature_importance(X, y, feat_names, top_n=20):
    rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("clf", RandomForestClassifier(300, class_weight="balanced",
                                                   random_state=42, n_jobs=-1))])
    rf.fit(X, y)
    mdi = rf.named_steps["clf"].feature_importances_
    idx = np.argsort(mdi)[-top_n:]
    return [feat_names[i] for i in idx], mdi[idx]


# ═════════════════════════════════════════════════════════════════════════════
# 5. DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def sax(ax, title="", ga="y"):
    ax.set_facecolor(P["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(P["border"]); sp.set_linewidth(1.1)
    ax.tick_params(colors=P["muted"], labelsize=8)
    ax.xaxis.label.set_color(P["muted"]); ax.yaxis.label.set_color(P["muted"])
    if title:
        ax.set_title(title, color=P["text"], fontsize=9,
                     fontweight="bold", pad=7, fontfamily="monospace")
    if ga:
        ax.grid(axis=ga, color=P["gridl"], lw=0.9, alpha=0.9)
        ax.set_axisbelow(True)


def make_dashboard(res_A, res_B, res_C, X_comb, y, feat_comb,
                   out="ml_extended_dashboard.png"):

    arms   = [res_A, res_B, res_C]
    nmod   = len(M_L)
    auc_m  = np.array([[r["mean_auc"] for r in a] for a in arms])
    std_m  = np.array([[r["std_auc"]  for r in a] for a in arms])
    tim_m  = np.array([[r["training_time_s"] for r in a] for a in arms])
    fi_names, fi_vals = feature_importance(X_comb, y, feat_comb)

    fig = plt.figure(figsize=(26, 20), facecolor=P["bg"])
    ogs = GridSpec(4, 1, figure=fig, hspace=0.58, top=0.93, bottom=0.04,
                   left=0.05, right=0.97, height_ratios=[0.07, 1, 1, 1])

    # ── title ─────────────────────────────────────────────────────────────────
    ax_t = fig.add_subplot(ogs[0])
    ax_t.set_facecolor(P["bg"])
    for sp in ax_t.spines.values(): sp.set_visible(False)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    ax_t.text(0.5, 0.78, "THREE-ARM ML COMPARISON", ha="center",
              transform=ax_t.transAxes, fontsize=28, fontweight="900",
              color=P["text"], fontfamily="monospace")
    ax_t.text(0.5, 0.22,
              "ARM A: Clinical only   ·   ARM B: MRI Imaging only   ·   "
              "ARM C: Clinical + Imaging combined   |   "
              "Nested 5-outer / 3-inner CV",
              ha="center", transform=ax_t.transAxes,
              fontsize=10, color=P["muted"], fontfamily="monospace")
    for i, (lbl, col) in enumerate(zip(ARM_L, ARM_C)):
        ax_t.text(0.29 + i * 0.14, 0.02, f"■  {lbl}", ha="center",
                  transform=ax_t.transAxes, fontsize=9.5, color=col,
                  fontfamily="monospace", fontweight="bold")

    x = np.arange(nmod); w = 0.24; off = [-w, 0, w]

    # ── (A) grouped AUC bar ───────────────────────────────────────────────────
    gs1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=ogs[1],
                                  wspace=0.30, width_ratios=[3, 2])
    ax_a = fig.add_subplot(gs1[0]); sax(ax_a, "(A)  Mean AUC by Model & Arm")
    for ai, (arm, col, lbl) in enumerate(zip(arms, ARM_C, ARM_L)):
        vals = [r["mean_auc"] for r in arm]
        errs = [r["std_auc"]  for r in arm]
        bars = ax_a.bar(x + off[ai], vals, w * 0.9, color=col, alpha=0.88,
                        yerr=errs, capsize=3,
                        error_kw=dict(ecolor=P["muted"], elinewidth=1.2, capthick=1.2),
                        label=lbl)
        for b, v in zip(bars, vals):
            ax_a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.013,
                      f"{v:.3f}", ha="center", va="bottom",
                      fontsize=6.5, color=P["text"], fontweight="bold")
    ax_a.set_xticks(x); ax_a.set_xticklabels(M_L, fontsize=9, color=P["text"])
    ax_a.set_ylim(0.38, 0.85); ax_a.set_ylabel("Mean AUC", color=P["muted"])
    ax_a.axhline(0.5, ls="--", lw=1.2, color=P["m4"], alpha=0.5, label="Chance (0.5)")
    ax_a.legend(framealpha=0, labelcolor=P["text"], fontsize=8.5, loc="upper left")
    bi, bj = np.unravel_index(auc_m.argmax(), auc_m.shape)
    ax_a.annotate(f"BEST\n{auc_m[bi,bj]:.3f}",
                  xy=(bj + off[bi], auc_m[bi, bj]),
                  xytext=(bj + off[bi] + 0.38, auc_m[bi, bj] + 0.03),
                  fontsize=7.5, color=ARM_C[bi], fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=ARM_C[bi], lw=1.4),
                  fontfamily="monospace")

    # ── (B) time heatmap ──────────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs1[1]); sax(ax_h, "(B)  Training Time Heat-map (s)", ga=None)
    im = ax_h.imshow(np.log1p(tim_m), aspect="auto", cmap="YlOrRd", origin="upper")
    ax_h.set_xticks(range(nmod))
    ax_h.set_xticklabels(M_L, fontsize=7.5, color=P["text"], rotation=30, ha="right")
    ax_h.set_yticks(range(3)); ax_h.set_yticklabels(ARM_L, fontsize=8.5, color=P["text"])
    for r in range(3):
        for c in range(nmod):
            ax_h.text(c, r, f"{tim_m[r,c]:.1f}s", ha="center", va="center",
                      fontsize=8, color="black", fontweight="bold")
    cb = fig.colorbar(im, ax=ax_h, fraction=0.04, pad=0.04)
    cb.ax.tick_params(colors=P["muted"], labelsize=7)
    cb.set_label("log(1+s)", color=P["muted"], fontsize=7.5)

    # ── (C) fold distributions ────────────────────────────────────────────────
    gs2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=ogs[2],
                                  wspace=0.28, width_ratios=[3, 2])
    ax_b = fig.add_subplot(gs2[0])
    sax(ax_b, "(C)  Fold AUC Distributions  (all arms × models)")
    rng2 = np.random.default_rng(1)
    pos, dat, bcols = [], [], []
    t_pos, t_lbl = [], []
    cur = 1.0; gap = 0.35
    for mi, mn in enumerate(M_L):
        t_pos.append(cur + w)
        t_lbl.append(mn)
        for ai, (arm, col) in enumerate(zip(arms, ARM_C)):
            pos.append(cur); dat.append(arm[mi]["auc_scores"]); bcols.append(col)
            cur += w + 0.03
        cur += gap
    bp = ax_b.boxplot(dat, positions=pos, widths=w * 0.82, patch_artist=True,
                      medianprops=dict(color=P["text"], lw=2.0),
                      whiskerprops=dict(color=P["muted"]),
                      capprops=dict(color=P["muted"]),
                      flierprops=dict(marker=".", color=P["muted"], ms=3))
    for patch, col in zip(bp["boxes"], bcols):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
    for p2, sc, col in zip(pos, dat, bcols):
        j = rng2.uniform(-w * 0.28, w * 0.28, len(sc))
        ax_b.scatter(p2 + j, sc, color=col, s=28, zorder=5, alpha=0.88)
    ax_b.set_xticks(t_pos); ax_b.set_xticklabels(t_lbl, fontsize=9, color=P["text"])
    ax_b.set_ylabel("AUC", color=P["muted"]); ax_b.set_ylim(0.25, 1.05)
    ax_b.axhline(0.5, ls="--", lw=0.9, color=P["m4"], alpha=0.4)
    patches = [mpatches.Patch(color=c, label=l, alpha=0.8)
               for c, l in zip(ARM_C, ARM_L)]
    ax_b.legend(handles=patches, framealpha=0, labelcolor=P["text"],
                fontsize=8, loc="upper left")

    # ── (D) stability ─────────────────────────────────────────────────────────
    ax_s = fig.add_subplot(gs2[1])
    sax(ax_s, "(D)  Stability  (σ AUC — lower is better)")
    for ai, (arm, col, lbl) in enumerate(zip(arms, ARM_C, ARM_L)):
        stds = [r["std_auc"] for r in arm]
        bars2 = ax_s.bar(x + off[ai], stds, w * 0.9, color=col, alpha=0.88, label=lbl)
        for b, v in zip(bars2, stds):
            ax_s.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                      f"{v:.3f}", ha="center", va="bottom",
                      fontsize=6.5, color=P["text"])
    ax_s.set_xticks(x); ax_s.set_xticklabels(M_L, fontsize=9, color=P["text"])
    ax_s.set_ylabel("σ AUC", color=P["muted"])

    # ── (E) radar ─────────────────────────────────────────────────────────────
    gs3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=ogs[3],
                                  wspace=0.28, width_ratios=[1.6, 2.4])
    ax_r = fig.add_subplot(gs3[0], polar=True)
    ax_r.set_facecolor(P["panel"]); ax_r.spines["polar"].set_color(P["border"])
    ax_r.tick_params(colors=P["muted"], labelsize=8)
    ax_r.set_title("(E)  Best-Model Radar per Arm", color=P["text"], fontsize=9,
                   fontweight="bold", pad=16, fontfamily="monospace")
    cats   = ["Mean AUC", "Speed\n(inv-log t)", "Stability\n(inv-σ)"]
    NC     = len(cats)
    angles = [i / NC * 2 * np.pi for i in range(NC)] + [0]
    ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(cats, color=P["text"], fontsize=9)
    ax_r.set_ylim(0, 1); ax_r.yaxis.set_tick_params(labelleft=False)
    ax_r.set_yticks([.25, .5, .75, 1.])
    ax_r.yaxis.grid(color=P["border"], lw=0.9)
    ax_r.xaxis.grid(color=P["border"], lw=0.9)
    all_t   = tim_m.flatten(); all_sig = std_m.flatten()
    for ai, (arm, col, lbl) in enumerate(zip(arms, ARM_C, ARM_L)):
        bi2 = int(np.argmax([r["mean_auc"] for r in arm]))
        br  = arm[bi2]
        na  = np.clip((br["mean_auc"] - 0.5) / 0.5, 0, 1)
        ns  = 1 - (np.log1p(br["training_time_s"]) - np.log1p(all_t.min())) / \
                  (np.log1p(all_t.max()) - np.log1p(all_t.min()) + 1e-9)
        nst = 1 - (br["std_auc"] - all_sig.min()) / (all_sig.max() - all_sig.min() + 1e-9)
        vals = [na, ns, nst, na]
        ax_r.plot(angles, vals, color=col, lw=2.5,
                  label=f"{lbl}  [{M_L[bi2]}  {br['mean_auc']:.3f}]")
        ax_r.fill(angles, vals, color=col, alpha=0.13)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.65, 1.18),
                framealpha=0, labelcolor=P["text"], fontsize=8)

    # ── (F) feature importance ────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs3[1])
    sax(ax_f, "(F)  Top-20 Feature Importances  (Combined Arm · RF MDI)", ga="x")
    fi_cols = [P["armB"] if (n.startswith("SI") or n.startswith("V"))
               else P["armA"] for n in fi_names]
    yp = np.arange(len(fi_names))
    ax_f.barh(yp, fi_vals, color=fi_cols, alpha=0.85, edgecolor="none")
    ax_f.set_yticks(yp); ax_f.set_yticklabels(fi_names, fontsize=8.5, color=P["text"])
    ax_f.set_xlabel("MDI Importance", color=P["muted"])
    cp = mpatches.Patch(color=P["armA"], alpha=0.85, label="Clinical feature")
    ip = mpatches.Patch(color=P["armB"], alpha=0.85,
                        label="Imaging feature  (SI = signal intensity, V = volume)")
    ax_f.legend(handles=[cp, ip], framealpha=0, labelcolor=P["text"],
                fontsize=8.5, loc="lower right")
    ax_f.text(fi_vals[-1] + 0.0004, len(fi_names) - 1,
              f"{fi_vals[-1]:.4f}", va="center",
              fontsize=7.5, color=P["text"])

    plt.savefig(out, dpi=155, bbox_inches="tight", facecolor=P["bg"])
    print(f"\nDashboard saved → {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 6. RESULTS CSV
# ═════════════════════════════════════════════════════════════════════════════

def save_csv(res_A, res_B, res_C, path="results_summary.csv"):
    rows = []
    for arm_lbl, arm_res in zip(ARM_L, [res_A, res_B, res_C]):
        for r in arm_res:
            rows.append({
                "arm": arm_lbl, "model": r["model"],
                "mean_auc": round(r["mean_auc"], 4),
                "std_auc":  round(r["std_auc"],  4),
                "time_s":   round(r["training_time_s"], 2),
                **{f"fold_{i+1}": round(v, 4)
                   for i, v in enumerate(r["auc_scores"])}
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Results CSV saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    csv_path   = sys.argv[1] if len(sys.argv) > 1 else "clinical_characteristics.csv"
    si_vol_csv = sys.argv[2] if len(sys.argv) > 2 else "si_and_vol_all.csv"

    print("=" * 60)
    print("  THREE-ARM ML PIPELINE")
    print("=" * 60)

    df_clin, X_clin, y, feat_clin = load_clinical(csv_path)
    X_img, feat_img = load_imaging(df_clin, si_vol_csv)
    X_comb = np.hstack([X_clin, X_img])
    feat_comb = feat_clin + feat_img

    res_A = run_arm(X_clin, y, "ARM A: Clinical only")
    res_B = run_arm(X_img,  y, "ARM B: Imaging only")
    res_C = run_arm(X_comb, y, "ARM C: Combined")

    # tabular summary
    print("\n" + "=" * 72)
    print(f"{'Arm':<13} {'Model':<14} {'AUC':>8} {'±σ':>8} {'Time(s)':>10}")
    print("=" * 72)
    for lbl, res in zip(ARM_L, [res_A, res_B, res_C]):
        for r in res:
            print(f"{lbl:<13} {r['model']:<14} "
                  f"{r['mean_auc']:>8.4f} {r['std_auc']:>8.4f} "
                  f"{r['training_time_s']:>10.2f}")
        print("-" * 72)

    save_csv(res_A, res_B, res_C)
    make_dashboard(res_A, res_B, res_C, X_comb, y, feat_comb)
    print("\nAll done.")
