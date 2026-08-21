"""
src/data_loader.py
------------------
FlightIQ — AI Travel Price Intelligence
Data loading utilities for the raw flight pricing dataset.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "flight_pricing_dataset.csv"


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw flight pricing dataset without any modification."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dictionary for the loaded dataframe."""
    summary = {
        "shape": df.shape,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_values_per_col": {col: df[col].nunique() for col in df.columns},
        "numerical_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
    }
    return summary


if __name__ == "__main__":
    df = load_raw_data()
    summary = get_data_summary(df)
    print(f"Shape       : {summary['shape']}")
    print(f"Columns     : {summary['columns']}")
    print(f"Missing     : {summary['total_missing']}")
    print(f"Duplicates  : {summary['duplicate_rows']}")
    print(f"Numerical   : {summary['numerical_columns']}")
    print(f"Categorical : {summary['categorical_columns']}")
