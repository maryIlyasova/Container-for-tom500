import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
SHAP_FILE = "shap_importance.csv"
DATA_FILE = "mri_features.csv"   # replace with your dataset
THRESHOLD = 0.001
TOP_N = 30

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(SHAP_FILE)
df = df.sort_values(by="mean_abs_shap", ascending=False)

print("Top 10 features:")
print(df.head(10))

# =========================
# PLOT: TOP-N FEATURES
# =========================
df_top = df.head(TOP_N)

plt.figure(figsize=(10, 6))
plt.barh(df_top["feature"], df_top["mean_abs_shap"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP value|")
plt.title(f"Top {TOP_N} Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("shap_top_features.png")
plt.close()

# =========================
# PLOT: ALL FEATURES
# =========================
plt.figure(figsize=(10, 20))
plt.barh(df["feature"], df["mean_abs_shap"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP value|")
plt.title("All Features Importance (SHAP)")
plt.tight_layout()
plt.savefig("shap_all_features.png")
plt.close()

# =========================
# FEATURE SELECTION METHODS
# =========================

# 1. Threshold-based
selected_threshold = df[df["mean_abs_shap"] >= THRESHOLD]["feature"]
print(f"Selected (threshold={THRESHOLD}):", len(selected_threshold))

# 2. Top-N
selected_topn = df.head(TOP_N)["feature"]
print(f"Selected (top {TOP_N}):", len(selected_topn))

# 3. Cumulative importance (95%)
df["cumulative"] = df["mean_abs_shap"].cumsum()
total = df["mean_abs_shap"].sum()
df["cumulative_ratio"] = df["cumulative"] / total

selected_cum = df[df["cumulative_ratio"] <= 0.95]["feature"]
print("Selected (95% importance):", len(selected_cum))

# =========================
# SAVE SELECTED FEATURES
# =========================
selected_threshold.to_csv("selected_features_threshold.csv", index=False)
selected_topn.to_csv("selected_features_topn.csv", index=False)
selected_cum.to_csv("selected_features_cumulative.csv", index=False)

# =========================
# APPLY TO DATASET
# =========================
try:
    X = pd.read_csv(DATA_FILE)

    X_selected = X[selected_topn]  # you can change method here

    X_selected.to_csv("dataset_selected.csv", index=False)
    print("Filtered dataset saved as dataset_selected.csv")

except Exception as e:
    print("Dataset not processed:", e)

print("Done.")
