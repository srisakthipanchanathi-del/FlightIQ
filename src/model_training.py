"""
src/model_training.py
---------------------
FlightIQ — AI Travel Price Intelligence
Trains, evaluates, and saves three regression models.
"""

import warnings
warnings.filterwarnings("ignore")

import json, sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import (
    build_preprocessor, prepare_data, load_data,
    ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET,
    get_feature_names
)

MODELS_DIR   = Path(__file__).resolve().parents[1] / "models"
ASSETS_MODEL = Path(__file__).resolve().parents[1] / "assets" / "model"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_MODEL.mkdir(parents=True, exist_ok=True)

MODEL_PATH    = MODELS_DIR / "flight_price_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def evaluate(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2    = r2_score(y_test, preds)
    print(f"  [{name}]  MAE=₹{mae:,.0f}  RMSE=₹{rmse:,.0f}  R²={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "predictions": preds}


def get_models():
    return [
        ("Linear Regression", LinearRegression(n_jobs=-1)),
        ("Random Forest",     RandomForestRegressor(
                                  n_estimators=150,
                                  max_depth=18,
                                  min_samples_leaf=5,
                                  max_features="sqrt",
                                  n_jobs=-1,
                                  random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(
                                  n_estimators=200,
                                  learning_rate=0.1,
                                  max_depth=5,
                                  min_samples_leaf=10,
                                  subsample=0.8,
                                  random_state=42)),
    ]


def run_training(random_state=42):
    print("=" * 60)
    print("FlightIQ — Model Training Pipeline")
    print("=" * 60)

    df = load_data()
    print(f"\nLoaded: {df.shape}")

    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(
        df, test_size=0.20, random_state=random_state
    )

    results      = []
    trained_models = {}

    print("\n--- Training ---")
    for name, estimator in get_models():
        print(f"\n  Training: {name}...")
        estimator.fit(X_train, y_train)
        res = evaluate(estimator, X_test, y_test, name)
        results.append(res)
        trained_models[name] = estimator

    # Best by R²
    best_result = max(results, key=lambda x: x["R2"])
    best_name   = best_result["model"]
    best_model  = trained_models[best_name]
    print(f"\nBest model: {best_name}  R²={best_result['R2']:.4f}")

    # Save full Pipeline: preprocessor (fitted) + best model (fitted)
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        best_model),
    ])
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Saved pipeline → {MODEL_PATH}")

    metadata = {
        "best_model":              best_name,
        "features":                ALL_FEATURES,
        "numerical_features":      NUMERICAL_FEATURES,
        "categorical_features":    CATEGORICAL_FEATURES,
        "target":                  TARGET,
        "train_size":              int(X_train.shape[0]),
        "test_size":               int(X_test.shape[0]),
        "n_features_raw":          len(ALL_FEATURES),
        "n_features_encoded":      int(X_train.shape[1]),
        "feature_names_encoded":   feature_names,
        "metrics": {r["model"]: {"MAE": round(r["MAE"], 2),
                                  "RMSE": round(r["RMSE"], 2),
                                  "R2": round(r["R2"], 4)}
                    for r in results},
        "best_metrics": {
            "MAE":  round(best_result["MAE"], 2),
            "RMSE": round(best_result["RMSE"], 2),
            "R2":   round(best_result["R2"], 4),
        }
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata → {METADATA_PATH}")

    return results, best_result, final_pipeline, preprocessor, feature_names, X_test, y_test


def load_pipeline():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    results, best, pipe, pre, fnames, X_test, y_test = run_training()
    print("\n=== FINAL RESULTS ===")
    for r in results:
        print(f"  {r['model']:<22}: MAE=₹{r['MAE']:>8,.0f}  RMSE=₹{r['RMSE']:>8,.0f}  R²={r['R2']:.4f}")
    print(f"\nBest: {best['model']}  MAE=₹{best['MAE']:,.0f}  RMSE=₹{best['RMSE']:,.0f}  R²={best['R2']:.4f}")
