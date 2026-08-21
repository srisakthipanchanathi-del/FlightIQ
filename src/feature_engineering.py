"""
src/feature_engineering.py
---------------------------
FlightIQ — AI Travel Price Intelligence
Builds and returns the sklearn preprocessing pipeline + feature/target arrays.

Design:
  - All feature decisions are documented inline.
  - No leakage: Price never touches the feature pipeline.
  - Pipeline is fully fitted on train set only.
  - Encoders/scalers are reused at inference time by loading the saved pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleaned_flight_data.csv"

# ── Feature selection ─────────────────────────────────────────────────────────
# NUMERICAL — all zero-missing after cleaning, so imputer is a safety net only
NUMERICAL_FEATURES = [
    "Distance_km",            # strongest single predictor (r=0.652)
    "Duration_Minutes",       # near-equal to distance (r=0.650)
    "Total_Stops_Numeric",    # moderate signal (r=0.113)
    "Days_Before_Departure",  # mild negative signal (r=-0.096)
    "Departure_Time_Minutes", # weak but included for completeness
    "Arrival_Time_Minutes",   # weak but included for completeness
    "Departure_Month",        # seasonal pattern
    "Departure_DayOfWeek_Num",# weekend premium observed
    "Passenger_Count",        # very weak, but known at booking
]

# CATEGORICAL — low-to-moderate cardinality, all clean after Part 1
CATEGORICAL_FEATURES = [
    "Airline",          # 13 categories — strong price signal
    "Source",           # 18 cities — strong signal
    "Destination",      # 18 cities — strong signal
    "Travel_Class",     # 4 classes — very strong signal (4.5× price gap)
    "Season",           # 4 seasons — mild signal
    "Weekday",          # 7 days — small weekend premium
    "Aircraft_Type",    # 8 types — included as proxy for route/operator
    "Booking_Channel",  # 5 channels — minor signal, included for completeness
]

# EXCLUDED columns and reasons (documented for reproducibility):
# Flight_ID          — identifier only, no predictive value
# Price              — TARGET variable, must never be a feature
# Duration           — raw string; Duration_Minutes is the clean version
# Total_Stops        — raw string; Total_Stops_Numeric is the clean version
# Departure_Date     — raw string with ~5K missing; Departure_Month extracted
# Departure_Time     — raw string; Departure_Time_Minutes extracted
# Arrival_Time       — raw string; Arrival_Time_Minutes extracted
# Departure_Date_Parsed — datetime with missing; numeric features already extracted
# Departure_DayOfYear   — high cardinality (365), correlated with Month; dropped
# Weekday_Num        — duplicate of Departure_DayOfWeek_Num (same info)

TARGET = "Price"

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_data(path=PROCESSED_PATH):
    return pd.read_csv(path)


def build_preprocessor():
    """Return an unfitted ColumnTransformer preprocessing pipeline."""
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # safety net
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # safety net
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline,  NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")

    return preprocessor


def get_feature_names(preprocessor):
    """Extract human-readable feature names after fitting."""
    num_names = NUMERICAL_FEATURES.copy()
    cat_names = (preprocessor
                 .named_transformers_["cat"]
                 .named_steps["encoder"]
                 .get_feature_names_out(CATEGORICAL_FEATURES)
                 .tolist())
    return num_names + cat_names


def prepare_data(df=None, test_size=0.20, random_state=42):
    """
    Load, split, and preprocess. Returns:
      X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names
    """
    if df is None:
        df = load_data()

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)   # fit ONLY on train
    X_test_proc  = preprocessor.transform(X_test)        # transform test

    feature_names = get_feature_names(preprocessor)

    print(f"Train size : {X_train_proc.shape[0]:,} rows × {X_train_proc.shape[1]} features")
    print(f"Test size  : {X_test_proc.shape[0]:,} rows × {X_test_proc.shape[1]} features")

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pre, fnames = prepare_data()
    print(f"\nFeatures ({len(fnames)}):")
    for i, f in enumerate(fnames, 1):
        print(f"  {i:3}. {f}")
