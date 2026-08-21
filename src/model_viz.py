"""
Generates model comparison, actual-vs-predicted, residual, and
feature-importance charts. Run AFTER model_training.py completes.
"""
import warnings
warnings.filterwarnings("ignore")
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import prepare_data, load_data, ALL_FEATURES, get_feature_names, build_preprocessor

ASSETS_MODEL = Path(__file__).resolve().parents[1] / "assets" / "model"
MODEL_PATH   = Path(__file__).resolve().parents[1] / "models" / "flight_price_model.joblib"
META_PATH    = Path(__file__).resolve().parents[1] / "models" / "model_metadata.json"
ASSETS_MODEL.mkdir(parents=True, exist_ok=True)

BG     = "#FAFAFA"
ACCENT = "#2563EB"
DPI    = 160

def _fmt_k(x, _):
    if x >= 1_000_000: return f"₹{x/1_000_000:.1f}M"
    if x >= 1_000:     return f"₹{x/1_000:.0f}K"
    return f"₹{x:.0f}"

def _save(fig, fname):
    path = ASSETS_MODEL / fname
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {fname}")
    return path

def plot_model_comparison(results):
    names = [r["model"] for r in results]
    maes  = [r["MAE"]  for r in results]
    r2s   = [r["R2"]   for r in results]
    colors = ["#64748B", "#2563EB", "#16A34A"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Comparison — Test Set Performance", fontsize=13, fontweight="bold")
    fig.patch.set_facecolor(BG)

    # MAE bar
    ax = axes[0]
    ax.set_facecolor(BG)
    bars = ax.bar(names, maes, color=colors, alpha=0.85, width=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_ylabel("MAE (INR)")
    ax.set_title("Mean Absolute Error  (lower = better)")
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"₹{v:,.0f}", ha="center", va="bottom", fontsize=9)

    # R² bar
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    bars2 = ax2.bar(names, r2s, color=colors, alpha=0.85, width=0.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("R² Score")
    ax2.set_title("R² Score  (higher = better, max=1.0)")
    for bar, v in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return _save(fig, "model_comparison.png")


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # Hexbin for 18k points
    hb = ax.hexbin(y_test, y_pred, gridsize=50, cmap="Blues", mincnt=1, bins="log")
    cb = plt.colorbar(hb, ax=ax); cb.set_label("log₁₀(count)", fontsize=9)

    # Perfect line
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_xlabel("Actual Price (INR)")
    ax.set_ylabel("Predicted Price (INR)")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ax.set_title(f"Actual vs Predicted — {model_name}\nR²={r2:.4f}  MAE=₹{mae:,.0f}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _save(fig, "actual_vs_predicted.png")


def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=13, fontweight="bold")
    fig.patch.set_facecolor(BG)

    # Residuals vs predicted
    ax = axes[0]; ax.set_facecolor(BG)
    ax.hexbin(y_pred, residuals, gridsize=45, cmap="RdBu_r", mincnt=1, bins="log")
    ax.axhline(0, color="#DC2626", linestyle="--", linewidth=1.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_xlabel("Predicted Price (INR)")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residuals vs Predicted")

    # Residual histogram
    ax2 = axes[1]; ax2.set_facecolor(BG)
    ax2.hist(residuals, bins=80, color=ACCENT, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="#DC2626", linestyle="--", linewidth=1.5)
    ax2.axvline(residuals.mean(), color="#16A34A", linestyle="-", linewidth=1.5,
                label=f"Mean: ₹{residuals.mean():,.0f}")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax2.set_xlabel("Residual (INR)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, "residual_plot.png")


def plot_feature_importance(model, feature_names, top_n=25):
    if not hasattr(model, "feature_importances_"):
        print("  Skipping — model has no feature_importances_")
        return None
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.33)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.barh(top_names[::-1], top_vals[::-1], color=ACCENT, alpha=0.85, height=0.7)
    ax.set_xlabel("Feature Importance (MDI)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — {type(model).__name__}",
                 fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _save(fig, "feature_importance.png")


def run_model_visualizations():
    print("=== Building model visualizations ===")
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    # Re-train all models quickly to get predictions (same as training script)
    from model_training import get_models, evaluate
    results = []
    trained = {}
    for name, est in get_models():
        est.fit(X_train, y_train)
        res = evaluate(est, X_test, y_test, name)
        results.append(res)
        trained[name] = est

    best = max(results, key=lambda r: r["R2"])
    best_model = trained[best["model"]]
    y_pred = best["predictions"]

    print("\nPlotting...")
    plot_model_comparison(results)
    plot_actual_vs_predicted(y_test.values, y_pred, best["model"])
    plot_residuals(y_test.values, y_pred, best["model"])
    plot_feature_importance(best_model, feature_names)
    print(f"\nDone. Assets saved to assets/model/")
    return results, best


if __name__ == "__main__":
    run_model_visualizations()
