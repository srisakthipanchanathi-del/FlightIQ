"""
src/explainability.py
---------------------
FlightIQ — AI Travel Price Intelligence
SHAP-based model explainability.
Uses a random subsample for efficiency — full dataset SHAP would take hours.
Falls back gracefully to native feature importance if SHAP is unavailable.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import json

ASSETS_EXPL = Path(__file__).resolve().parents[1] / "assets" / "explainability"
ASSETS_EXPL.mkdir(parents=True, exist_ok=True)

METADATA_PATH = Path(__file__).resolve().parents[1] / "models" / "model_metadata.json"
MODEL_PATH    = Path(__file__).resolve().parents[1] / "models" / "flight_price_model.joblib"

BG     = "#FAFAFA"
ACCENT = "#2563EB"
DPI    = 160


def _save(fig, fname):
    path = ASSETS_EXPL / fname
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path.name}")
    return path


def plot_native_feature_importance(model, feature_names, top_n=25):
    """Bar chart of native feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print("  Model does not support native feature importance — skipping.")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.32)))
    ax.barh(top_names[::-1], top_vals[::-1], color=ACCENT, alpha=0.85, height=0.7)
    ax.set_xlabel("Feature Importance (MDI)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances\n(Mean Decrease in Impurity — tree-based model)",
                 fontsize=13, fontweight="bold")
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    return _save(fig, "feature_importance_native.png")


def run_shap_analysis(model, X_train_proc, feature_names, sample_n=2000, random_state=42):
    """
    Run SHAP TreeExplainer on a subsample for speed.
    Returns shap_values array or None if SHAP fails.
    """
    try:
        import shap
    except ImportError:
        print("  SHAP not installed — using native feature importance only.")
        return None

    np.random.seed(random_state)
    if X_train_proc.shape[0] > sample_n:
        idx = np.random.choice(X_train_proc.shape[0], sample_n, replace=False)
        X_sample = X_train_proc[idx]
    else:
        X_sample = X_train_proc

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        print(f"  SHAP computed on {X_sample.shape[0]} samples.")

        # Summary plot
        fig, ax = plt.subplots(figsize=(11, max(7, len(feature_names[:25]) * 0.35)))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=feature_names,
                          max_display=25,
                          show=False, plot_type="bar")
        plt.title("SHAP Feature Importance (Mean |SHAP value|)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = ASSETS_EXPL / "shap_summary_bar.png"
        plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved → shap_summary_bar.png")

        # Beeswarm plot
        shap.summary_plot(shap_values, X_sample,
                          feature_names=feature_names,
                          max_display=20,
                          show=False)
        plt.title("SHAP Beeswarm — Feature Impact on Price", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path2 = ASSETS_EXPL / "shap_beeswarm.png"
        plt.savefig(path2, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved → shap_beeswarm.png")

        return shap_values, X_sample

    except Exception as e:
        print(f"  SHAP failed: {e}. Using native feature importance.")
        return None


def run_explainability(model, X_train_proc, feature_names):
    """Main entry: run native importance + SHAP."""
    print("=== Explainability ===")
    plot_native_feature_importance(model, feature_names, top_n=25)
    result = run_shap_analysis(model, X_train_proc, feature_names)
    return result


if __name__ == "__main__":
    import joblib, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from feature_engineering import prepare_data

    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data()
    pipe   = joblib.load(MODEL_PATH)
    model  = pipe.named_steps["model"]
    run_explainability(model, X_train, feature_names)
