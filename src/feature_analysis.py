# src/feature_analysis.py
"""
Feature analysis for fuel-efficiency project.

Reads:  data/processed_engine_data.csv
Writes: results/correlation.csv
        results/correlation_heatmap.png
        results/top_feature_importances.png
        results/top_features.csv
        results/scatter_TOPn.png
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------
# Paths & folders
# -----------------------
ROOT = Path(".")
DATA_PATH = ROOT / "data" / "processed_engine_data.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------
# Helper: detect target
# -----------------------
def detect_target(columns):
    # prefer columns containing 'efficiency' or 'thermal' or 'gross'
    candidates = [c for c in columns if any(k in c.lower() for k in ["efficiency", "thermal", "gross", "bsfc", "consumption"])]
    if len(candidates) == 0:
        raise RuntimeError("No target-like column found (look for 'efficiency', 'bsfc', 'consumption', 'thermal'). Inspect columns in data file.")
    # prefer the best match order
    for pref in ["efficiency", "gross", "thermal", "bsfc", "consumption"]:
        for c in candidates:
            if pref in c.lower():
                return c
    return candidates[0]

# -----------------------
# Main analysis
# -----------------------
def main():
    print("🔁 Loading processed dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)

    # detect target column (efficiency)
    target_col = detect_target(df.columns)
    print("🎯 Detected target column:", target_col)

    # separate numeric and categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # ensure target is numeric column (if not, convert)
    if target_col not in numeric_cols:
        # try convert
        try:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        except Exception:
            pass

    # compute correlations (numeric only)
    print("📊 Computing correlations (numeric features)...")
    numeric_df = df[numeric_cols].copy()
    corr = numeric_df.corr()
    corr[target_col].sort_values(ascending=False).to_csv(RESULTS_DIR / "correlation_with_target.csv")
    corr.to_csv(RESULTS_DIR / "correlation_matrix.csv")
    print("Saved correlation CSVs to results/")

    # Heatmap (save)
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, linewidths=0.3)
    plt.title("Correlation matrix (numeric features)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()
    print("Saved correlation heatmap:", RESULTS_DIR / "correlation_heatmap.png")

    # Prepare ML features: one-hot encode non-numeric cols (if any)
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # one-hot encode object / categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        print(f"One-hot encoded {len(cat_cols)} categorical columns -> new shape {X.shape}")

    # Keep only numeric columns now
    X = X.select_dtypes(include=[np.number])

    # Impute if any missing after conversion
    if X.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Train/Test split and RandomForest for feature importance
    print("🌲 Training RandomForest for feature importance...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"Model test RMSE: {rmse:.4f}, R2: {r2:.4f}")

    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    feat_imp.head(30).to_csv(RESULTS_DIR / "feature_importances_top30.csv")
    feat_imp.to_csv(RESULTS_DIR / "feature_importances_all.csv")
    print("Saved feature importances CSVs to results/")

    # Plot top features
    top_n = 20
    plt.figure(figsize=(8, max(4, top_n*0.25)))
    feat_imp.head(top_n)[::-1].plot(kind="barh")
    plt.title(f"Top {top_n} feature importances (RandomForest)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_feature_importances.png", dpi=200)
    plt.close()
    print("Saved top feature importances plot:", RESULTS_DIR / "top_feature_importances.png")

    # Save top-n to CSV for GA selection
    top_features = feat_imp.head(15).index.tolist()
    pd.Series(top_features).to_csv(RESULTS_DIR / "top_features.csv", index=False, header=False)
    print("Top features saved (results/top_features.csv):")
    for i, f in enumerate(top_features, 1):
        print(f"  {i}. {f}")

    # Scatter plots for top features vs target (save a grid)
    scatter_dir = RESULTS_DIR / "scatter_plots"
    scatter_dir.mkdir(exist_ok=True)
    for f in top_features[:8]:  # create up to 8 scatter plots
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[f] if f in df.columns else X[f], y=df[target_col], alpha=0.6)
        plt.xlabel(f)
        plt.ylabel(target_col)
        plt.title(f"{f} vs {target_col}")
        plt.tight_layout()
        plt.savefig(scatter_dir / f"scatter_{f}.png", dpi=150)
        plt.close()
    print(f"Saved scatter plots for top features in {scatter_dir}")

    print("\n✅ Feature analysis complete. Check the 'results' folder for CSVs and plots.")

if __name__ == "__main__":
    main()
