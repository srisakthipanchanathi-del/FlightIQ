"""
src/data_preprocessing.py
--------------------------
FlightIQ — AI Travel Price Intelligence
Reusable, deterministic data cleaning and preprocessing pipeline.

Design principles:
  - No target leakage: Price is never used to construct input features.
  - All transformations are deterministic (no random state in cleaning).
  - Raw data is never modified.
  - Every step is a standalone function for reuse during inference.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "flight_pricing_dataset.csv"
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleaned_flight_data.csv"

# ---------------------------------------------------------------------------
# IATA/city canonical mapping  (built from actual dataset inspection)
# ---------------------------------------------------------------------------
CITY_CANONICAL = {
    # Mumbai
    "bom": "Mumbai", "mumbai": "Mumbai", "mumbai airport": "Mumbai",
    # Delhi
    "del": "Delhi", "delhi": "Delhi", "delhi airport": "Delhi",
    # Bangalore
    "blr": "Bangalore", "bangalore": "Bangalore", "bangalore airport": "Bangalore",
    # Hyderabad
    "hyd": "Hyderabad", "hyderabad": "Hyderabad", "hyderabad airport": "Hyderabad",
    # Chennai
    "maa": "Chennai", "chennai": "Chennai", "chennai airport": "Chennai",
    # Kolkata
    "ccu": "Kolkata", "kolkata": "Kolkata", "kolkata airport": "Kolkata",
    # Ahmedabad
    "amd": "Ahmedabad", "ahmedabad": "Ahmedabad", "ahmedabad airport": "Ahmedabad",
    # Pune
    "pnq": "Pune", "pune": "Pune", "pune airport": "Pune",
    # Jaipur
    "jai": "Jaipur", "jaipur": "Jaipur", "jaipur airport": "Jaipur",
    # Goa
    "goi": "Goa", "goa": "Goa", "goa airport": "Goa",
    # Dubai
    "dxb": "Dubai", "dubai": "Dubai", "dubai airport": "Dubai",
    # Doha
    "doh": "Doha", "doha": "Doha", "doha airport": "Doha",
    # Singapore
    "sin": "Singapore", "singapore": "Singapore", "singapore airport": "Singapore",
    # London
    "lhr": "London", "london": "London", "london airport": "London",
    # New York
    "jfk": "New York", "new york": "New York", "new york airport": "New York",
    # Bangkok
    "bkk": "Bangkok", "bangkok": "Bangkok", "bangkok airport": "Bangkok",
    # Sydney
    "syd": "Sydney", "sydney": "Sydney", "sydney airport": "Sydney",
    # Frankfurt
    "fra": "Frankfurt", "frankfurt": "Frankfurt", "frankfurt airport": "Frankfurt",
}

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
}

WEEKDAY_ORDER = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------
def load_raw(path=RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV with no modifications."""
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# Step 2 — Strip whitespace from column names and string values
# ---------------------------------------------------------------------------
def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


# ---------------------------------------------------------------------------
# Step 3 — Clean Price (target)
# ---------------------------------------------------------------------------
def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Price to float.
    Handles formats: plain numbers, 'Rs. 1,23,456.78', comma-separated.
    Drops rows where Price is null or non-parseable.
    Drops rows where Price <= 0.
    """
    df = df.copy()

    def _parse_price(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        # Remove currency prefix like 'Rs.' or 'INR'
        val = re.sub(r'^[A-Za-z\s\.]+', '', val)
        # Remove commas
        val = val.replace(',', '').strip()
        try:
            return float(val)
        except ValueError:
            return np.nan

    df['Price'] = df['Price'].apply(_parse_price)

    before = len(df)
    df = df[df['Price'].notna() & (df['Price'] > 0)]
    dropped = before - len(df)
    print(f"  [Price] Dropped {dropped} rows (null/invalid/non-positive Price).")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Clean Duration → Duration_Minutes
# ---------------------------------------------------------------------------
def _parse_duration_to_minutes(val) -> float:
    """
    Parse Duration strings into total minutes.

    Formats observed in actual dataset:
      - "1.67"   → decimal HOURS  → multiply by 60
        (verified: 0.75 h = 45 min = matches '0h 45m' rows)
      - "0h 45m" → hours + minutes format
      - "177 min" → already in minutes
    Returns NaN for unparseable values.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip()

    # Format: "Xh Ym" e.g. "1h 28m", "0h 45m", "17h 04m"
    m = re.match(r'^(\d+)h\s*(\d+)m$', val)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    # Format: "X min" e.g. "177 min", "900 min"
    m = re.match(r'^(\d+)\s*min$', val)
    if m:
        return float(m.group(1))

    # Format: float/int string e.g. "1.67", "14.80" → decimal HOURS
    m = re.match(r'^\d+(\.\d+)?$', val)
    if m:
        hours = float(val)
        return round(hours * 60, 2)

    return np.nan


def clean_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add Duration_Minutes column. Impute missing with median."""
    df = df.copy()
    df['Duration_Minutes'] = df['Duration'].apply(_parse_duration_to_minutes)

    invalid = (df['Duration_Minutes'].notna()) & (df['Duration_Minutes'] <= 0)
    df.loc[invalid, 'Duration_Minutes'] = np.nan

    median_dur = df['Duration_Minutes'].median()
    missing_count = df['Duration_Minutes'].isna().sum()
    df['Duration_Minutes'] = df['Duration_Minutes'].fillna(median_dur)
    print(f"  [Duration] Parsed to minutes. Imputed {missing_count} missing with median ({median_dur:.1f} min).")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Clean Total_Stops → Total_Stops_Numeric
# ---------------------------------------------------------------------------
def clean_total_stops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Total_Stops to integer.
    'non-stop' → 0, '1 stop' → 1, '2 stops' → 2, '0'/'1'/'2' → as-is.
    """
    df = df.copy()

    def _parse_stops(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        if val in ("non-stop", "nonstop", "0"):
            return 0
        m = re.match(r'^(\d+)\s*stop', val)
        if m:
            return int(m.group(1))
        # plain digit
        if re.match(r'^\d+$', val):
            return int(val)
        return np.nan

    df['Total_Stops_Numeric'] = df['Total_Stops'].apply(_parse_stops)
    median_stops = df['Total_Stops_Numeric'].median()
    missing_count = df['Total_Stops_Numeric'].isna().sum()
    df['Total_Stops_Numeric'] = df['Total_Stops_Numeric'].fillna(median_stops)
    print(f"  [Total_Stops] Converted to numeric. Imputed {missing_count} missing with median ({median_stops}).")
    return df


# ---------------------------------------------------------------------------
# Step 6 — Clean Distance_km, Days_Before_Departure, Passenger_Count
# ---------------------------------------------------------------------------
def clean_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-intended columns to float/int."""
    df = df.copy()

    # Distance_km
    df['Distance_km'] = pd.to_numeric(df['Distance_km'], errors='coerce')
    median_dist = df['Distance_km'].median()
    n = df['Distance_km'].isna().sum()
    df['Distance_km'] = df['Distance_km'].fillna(median_dist)
    print(f"  [Distance_km] Converted. Imputed {n} missing with median ({median_dist:.1f}).")

    # Days_Before_Departure
    df['Days_Before_Departure'] = pd.to_numeric(df['Days_Before_Departure'], errors='coerce')
    median_days = df['Days_Before_Departure'].median()
    n = df['Days_Before_Departure'].isna().sum()
    df['Days_Before_Departure'] = df['Days_Before_Departure'].fillna(median_days)
    print(f"  [Days_Before_Departure] Converted. Imputed {n} missing with median ({median_days}).")

    # Passenger_Count — mixed: "1"/"one"
    def _parse_pax(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        if val in WORD_TO_NUM:
            return WORD_TO_NUM[val]
        try:
            return int(float(val))
        except ValueError:
            return np.nan

    df['Passenger_Count'] = df['Passenger_Count'].apply(_parse_pax)
    mode_pax = df['Passenger_Count'].mode()[0]
    n = df['Passenger_Count'].isna().sum()
    df['Passenger_Count'] = df['Passenger_Count'].fillna(mode_pax)
    print(f"  [Passenger_Count] Converted (word→int). Imputed {n} missing with mode ({mode_pax}).")
    return df


# ---------------------------------------------------------------------------
# Step 7 — Clean Categorical columns
# ---------------------------------------------------------------------------
def clean_airline(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize airline names to Title Case."""
    df = df.copy()
    df['Airline'] = df['Airline'].str.strip().str.title()
    mode_airline = df['Airline'].mode()[0]
    n = df['Airline'].isna().sum()
    df['Airline'] = df['Airline'].fillna(mode_airline)
    print(f"  [Airline] Normalized case. Imputed {n} missing with mode ('{mode_airline}').")
    return df


def clean_city_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize Source and Destination.
    Maps IATA codes, 'City Airport' variants, and mixed-case to a single city name.
    """
    df = df.copy()

    def _canonical(val):
        if pd.isna(val):
            return np.nan
        key = str(val).strip().lower()
        return CITY_CANONICAL.get(key, str(val).strip().title())

    for col in ['Source', 'Destination']:
        df[col] = df[col].apply(_canonical)
        mode_val = df[col].mode()[0]
        n = df[col].isna().sum()
        df[col] = df[col].fillna(mode_val)
        print(f"  [{col}] Canonicalized. Imputed {n} missing.")
    return df


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Travel_Class, Season, Weekday, Aircraft_Type, Booking_Channel."""
    df = df.copy()
    cat_cols = ['Travel_Class', 'Season', 'Weekday', 'Aircraft_Type', 'Booking_Channel']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
            mode_val = df[col].mode()[0]
            n = df[col].isna().sum()
            df[col] = df[col].fillna(mode_val)
            print(f"  [{col}] Standardized. Imputed {n} missing with mode ('{mode_val}').")
    return df


# ---------------------------------------------------------------------------
# Step 8 — Clean Date / Time columns
# ---------------------------------------------------------------------------
def _parse_time_to_minutes(val) -> float:
    """
    Convert time strings to minutes since midnight.
    Handles: '8:10 PM', '13:30', '07:05', '5:45 AM'
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    try:
        t = pd.to_datetime(val, format='%I:%M %p')
        return t.hour * 60 + t.minute
    except Exception:
        pass
    try:
        t = pd.to_datetime(val, format='%H:%M')
        return t.hour * 60 + t.minute
    except Exception:
        pass
    return np.nan


def clean_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Departure_Date → Departure_DayOfYear, Departure_Month, Departure_DayOfWeek_Num.
    Parse Departure_Time and Arrival_Time → minutes since midnight.
    """
    df = df.copy()

    # Departure_Date
    df['Departure_Date_Parsed'] = pd.to_datetime(df['Departure_Date'], errors='coerce')
    df['Departure_Month'] = df['Departure_Date_Parsed'].dt.month
    df['Departure_DayOfYear'] = df['Departure_Date_Parsed'].dt.dayofyear
    df['Departure_DayOfWeek_Num'] = df['Departure_Date_Parsed'].dt.dayofweek

    # Fill missing date features with median
    for feat in ['Departure_Month', 'Departure_DayOfYear', 'Departure_DayOfWeek_Num']:
        n = df[feat].isna().sum()
        df[feat] = df[feat].fillna(df[feat].median())
        print(f"  [{feat}] Extracted. Imputed {n} missing.")

    # Departure_Time → minutes since midnight
    df['Departure_Time_Minutes'] = df['Departure_Time'].apply(_parse_time_to_minutes)
    n = df['Departure_Time_Minutes'].isna().sum()
    df['Departure_Time_Minutes'] = df['Departure_Time_Minutes'].fillna(df['Departure_Time_Minutes'].median())
    print(f"  [Departure_Time_Minutes] Converted. Imputed {n} missing.")

    # Arrival_Time → minutes since midnight
    df['Arrival_Time_Minutes'] = df['Arrival_Time'].apply(_parse_time_to_minutes)
    n = df['Arrival_Time_Minutes'].isna().sum()
    df['Arrival_Time_Minutes'] = df['Arrival_Time_Minutes'].fillna(df['Arrival_Time_Minutes'].median())
    print(f"  [Arrival_Time_Minutes] Converted. Imputed {n} missing.")

    # Weekday numeric from actual Weekday column (as backup/validation)
    df['Weekday_Num'] = df['Weekday'].map(WEEKDAY_ORDER)
    n = df['Weekday_Num'].isna().sum()
    df['Weekday_Num'] = df['Weekday_Num'].fillna(df['Weekday_Num'].median())
    print(f"  [Weekday_Num] Mapped. Imputed {n} missing.")

    return df


# ---------------------------------------------------------------------------
# Step 9 — Outlier handling (invalid records only)
# ---------------------------------------------------------------------------
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove only clearly invalid records.
    - Duration_Minutes < 10 or > 1800 (30 hours): impossible flight
    - Price < 100: not a realistic flight fare in INR
    - Distance_km <= 0
    """
    df = df.copy()
    before = len(df)

    df = df[df['Duration_Minutes'] >= 10]
    df = df[df['Duration_Minutes'] <= 1800]
    df = df[df['Price'] >= 100]
    df = df[df['Distance_km'] > 0]

    after = len(df)
    print(f"  [Outliers] Removed {before - after} clearly invalid records.")
    return df


# ---------------------------------------------------------------------------
# Step 10 — Drop duplicates
# ---------------------------------------------------------------------------
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"  [Duplicates] Removed {before - len(df)} exact duplicate rows.")
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def run_cleaning_pipeline(raw_path=RAW_PATH, save_path=PROCESSED_PATH) -> pd.DataFrame:
    """
    Full deterministic cleaning pipeline.
    Returns the cleaned DataFrame and saves to processed/.
    """
    print("=" * 60)
    print("FlightIQ — Data Cleaning Pipeline")
    print("=" * 60)

    df_raw = load_raw(raw_path)
    original_shape = df_raw.shape
    original_missing = int(df_raw.isnull().sum().sum())
    print(f"\nRaw shape     : {original_shape}")
    print(f"Raw missing   : {original_missing:,}")
    print(f"Raw dupes     : {df_raw.duplicated().sum():,}\n")

    print("--- Step 1: Strip whitespace ---")
    df = strip_whitespace(df_raw)

    print("--- Step 2: Clean Price (target) ---")
    df = clean_price(df)

    print("--- Step 3: Clean Duration ---")
    df = clean_duration(df)

    print("--- Step 4: Clean Total_Stops ---")
    df = clean_total_stops(df)

    print("--- Step 5: Clean numeric features ---")
    df = clean_numeric_features(df)

    print("--- Step 6: Clean Airline ---")
    df = clean_airline(df)

    print("--- Step 7: Canonicalize Source/Destination ---")
    df = clean_city_columns(df)

    print("--- Step 8: Standardize other categoricals ---")
    df = clean_categorical_columns(df)

    print("--- Step 9: Parse date/time columns ---")
    df = clean_datetime_columns(df)

    print("--- Step 10: Remove outliers ---")
    df = handle_outliers(df)

    print("--- Step 11: Drop duplicates ---")
    df = drop_duplicates(df)

    cleaned_missing = int(df.isnull().sum().sum())
    print(f"\n{'=' * 60}")
    print(f"Cleaned shape   : {df.shape}")
    print(f"Cleaned missing : {cleaned_missing:,}")
    print(f"{'=' * 60}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\nSaved → {save_path}")

    return df, original_shape, original_missing


if __name__ == "__main__":
    df, orig_shape, orig_missing = run_cleaning_pipeline()
    print(f"\nDone. Final shape: {df.shape}")
