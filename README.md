# FlightIQ — AI Travel Price Intelligence

FlightIQ is an end-to-end flight price intelligence platform that combines exploratory data analysis, machine learning, model explainability, and personalized flight recommendations using historical flight pricing data.

---

## Project Overview

Flight pricing is non-linear and dynamic, fluctuating based on complex interactions between route distance, carrier, travel class, lead time, aircraft type, seasonality, and booking channels. For travelers and business analysts, understanding why flight prices differ and predicting future ticket costs requires structured analytical modeling rather than simple historical averages.

FlightIQ addresses this challenge by transforming raw flight data into empirical insights, real-time Machine Learning (ML) fare estimates, and multi-criteria ranked flight recommendations. The system integrates full data cleaning pipelines, interactive Exploratory Data Analysis (EDA), Gradient Boosting predictive models, SHAP (SHapley Additive exPlanations) model explainability, and an interactive web interface.

---

## Problem Statement

Flight fares vary widely across different travel classes, booking lead times, airlines, and journey durations. Consumers and corporate travel managers face several core challenges:

1. **Price Transparency**: Lack of clarity regarding which factors (such as airline brand vs. flight duration vs. booking window) drive price premiums.
2. **Fare Uncertainty**: High dispersion in flight costs makes it difficult to ascertain whether a quoted price represents a value deal or an over-priced fare.
3. **Multi-Criteria Selection**: Balancing price constraints against preferred flight durations, maximum allowable stops, and carrier reputation when searching for flight options.

FlightIQ provides a unified analytical engine that predicts prices, quantifies pricing drivers using explainable AI, and recommends optimal flight options based on user budget and travel preferences.

---

## System Architecture & Solution Pipeline

FlightIQ is designed as a modular, end-to-end data science and web engineering pipeline:

```text
Raw Flight Data (93,083 rows)
        │
        ▼
Data Preprocessing & Cleaning (src/data_preprocessing.py)
        │
        ▼
Feature Engineering & Transformation (src/feature_engineering.py)
        │
        ├───► Exploratory Data Analysis & Visualizations (src/visualization.py)
        │
        ▼
Model Training & Evaluation (src/model_training.py)
        │     ├── Linear Regression (R²: 0.6217)
        │     ├── Random Forest (R²: 0.6902)
        │     └── Gradient Boosting (Selected | R²: 0.7049, MAE: ₹14,262)
        │
        ├───► Model Explainability via SHAP (src/explainability.py)
        │
        ├───► Recommendation Engine (src/recommender.py)
        │
        ▼
FastAPI REST Application (app.py)
        │
        ▼
Interactive Light-Theme Web UI (web/index.html, web/styles.css, web/app.js)
```

---

## Key Features

### Data Processing & Pipeline Architecture
- **Duplicate Removal & Validation**: Automated detection and elimination of redundant flight entries.
- **Missing Value Imputation**: Median and mode imputation based on domain hierarchy.
- **Categorical Normalization**: Text standardization across airline names, sources, destinations, and booking channels.
- **Derived Time Features**: Parsing departure/arrival timestamps into minutes from midnight, departure month, and day-of-week indicators.
- **Duration & Stops Transformation**: Conversion of text duration formats into continuous numeric minutes and categorical stops into numeric counts.
- **Target Leakage Prevention**: Strict separation of training and testing data splits prior to preprocessing and encoding.

### Exploratory Data Analysis
- **Price Distribution Analysis**: Empirical density and log-normal pricing distributions.
- **Travel Class Price Gap**: Quantification of price multipliers between Economy, Premium Economy, Business, and First Class.
- **Lead Time Trajectory**: Bivariate analysis of booking lead time (days before departure) vs. fare inflation.
- **Airline Carrier Comparison**: Benchmark of mean pricing across 13 domestic and international airlines.
- **Route & Distance Correlation**: Evaluation of flight distance and journey duration against observed fares.

### Machine Learning Engine
- **Multi-Model Benchmark**: Evaluation of Linear Regression, Random Forest, and Gradient Boosting Regressors.
- **Scikit-Learn Pipeline**: Integrated preprocessing pipeline handling One-Hot Encoding for categorical features and StandardScaler for numerical features.
- **Model Artifact Persistence**: Model state saved as `models/flight_price_model.joblib` with complete metadata stored in `models/model_metadata.json`.

### Explainable AI (XAI)
- **Global Feature Importance**: Direct quantification of predictive weight per feature.
- **SHAP Beeswarm Analysis**: Directional impact analysis showing how specific feature values (e.g., high flight duration or early booking window) push price predictions up or down.

### Flight Recommendation Engine
- **Multi-Attribute Filtering**: Hard filtering by origin, destination, travel class, and maximum budget.
- **Composite Scoring**: Ranked recommendation algorithm evaluating price fit, flight duration, and stop counts.
- **Explanation Generation**: Transparent "Why it fits" justifications for top-ranked flight options.

### Interactive Web Application
- **Real-Time Fare Estimator**: Instant price prediction with LOW / TYPICAL / HIGH price indicator tags.
- **Interactive Chart Detail Views**: Click-to-expand modal visualizations featuring enlarged charts, key figures, and statistical notes.
- **Translucent Fixed Parallax UI**: Aviation-themed user experience with fixed background composition and responsive controls.

---

## Dataset Specifications

FlightIQ operates on historical flight data stored in `data/raw/flight_pricing_dataset.csv`. Preprocessing outputs `data/processed/cleaned_flight_data.csv`.

- **Total Records**: 93,083 flights
- **Train Split**: 74,466 records (80%)
- **Test Split**: 18,617 records (20%)
- **Raw Features**: 17 features (9 numerical, 8 categorical)
- **One-Hot Encoded Dimensions**: 86 total features
- **Target Variable**: `Price` (Continuous, INR ₹)
- **Price Range**: ₹152 to ₹999,306
- **Mean Price**: ₹72,990.45
- **Airlines Included**: 13 carriers (IndiGo, SpiceJet, Air India, Vistara, AirAsia India, Go First, Emirates, Lufthansa, Qatar Airways, Etihad Airways, Singapore Airlines, Thai Airways, British Airways)
- **Unique Routes**: 308 origin-destination routes

---

## Data Preprocessing Workflow

1. **Schema Validation**: Ensures all 17 raw columns match expected data types and structural bounds.
2. **Cleaning Operations**:
   - Strips whitespace and normalizes string casing for categorical variables.
   - Cleans numeric distance fields by removing non-numeric units (`km`).
   - Converts duration strings (e.g., `2h 10m`) into total minutes (`130`).
   - Converts total stops strings (e.g., `1 Stop`, `Non-Stop`) into integer counts (`1`, `0`).
3. **Timestamp Processing**:
   - Extract departure month (`1-12`) and day of week (`0-6`).
   - Compute departure and arrival times as minutes from midnight (`0-1439`).
4. **Data Preservation**: The raw dataset in `data/raw/flight_pricing_dataset.csv` is preserved unchanged. All cleaning routines write output to `data/processed/cleaned_flight_data.csv`.

---

## Feature Engineering

The feature engineering pipeline (`src/feature_engineering.py`) constructs domain-specific predictive indicators:

- `Distance_km`: Numerical continuous flight distance between source and destination.
- `Duration_Minutes`: Total elapsed flight time in minutes.
- `Total_Stops_Numeric`: Integer count of layovers.
- `Days_Before_Departure`: Advance booking window in days.
- `Departure_Time_Minutes`: Time of departure normalized to minutes from midnight.
- `Arrival_Time_Minutes`: Time of arrival normalized to minutes from midnight.
- `Departure_Month`: Calendar month of departure.
- `Departure_DayOfWeek_Num`: Numeric index of the day of the week.
- `Passenger_Count`: Number of passengers associated with the booking instance.
- `Travel_Class`: Categorical encoding (Economy, Premium Economy, Business, First).
- `Season`: Seasonal categorization (Summer, Monsoon, Autumn, Winter).
- `Weekday`: Named day of the week.
- `Aircraft_Type`: Commercial aircraft family (e.g., Airbus A320, Boeing 777).
- `Booking_Channel`: Channel through which ticket was bought (e.g., Website, Mobile App, Travel Agent).

---

## Machine Learning & Model Performance

Three regression algorithms were evaluated using identical 80/20 train/test splits (`74,466` training rows, `18,617` testing rows). Performance was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$).

### Model Performance Comparison

| Model | MAE (₹) | RMSE (₹) | $R^2$ Score | Selection Status |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | ₹23,200.86 | ₹45,587.54 | 0.6217 | Baseline Model |
| **Random Forest Regressor** | ₹16,531.95 | ₹41,252.87 | 0.6902 | Candidate Model |
| **Gradient Boosting Regressor** | **₹14,262.47** | **₹40,261.86** | **0.7049** | **Selected Final Model** |

### Evaluation Metrics Breakdown

- **Mean Absolute Error (MAE)**: Measures the average absolute magnitude of prediction errors. The Gradient Boosting model achieves an MAE of ₹14,262.47.
- **Root Mean Squared Error (RMSE)**: Penalizes larger prediction errors more heavily. Gradient Boosting achieves an RMSE of ₹40,261.86.
- **$R^2$ Score**: Indicates that the Gradient Boosting model explains **70.49% of the total variance** in flight ticket prices.

---

## Model Explainability (SHAP & Feature Importance)

To ensure model transparency, FlightIQ incorporates SHAP tree explainability (`src/explainability.py`):

1. **Top Predictive Drivers**:
   - **Distance & Duration**: Account for >60% of overall model decision weight.
   - **Travel Class**: First and Business class cabin options exert a strong positive push on predicted prices.
   - **Lead Time**: Advance booking (`Days_Before_Departure`) demonstrates a negative coefficient impact, lowering predicted fares for bookings made 21–45 days ahead.
2. **Interpretability Disclaimer**: Feature importance and SHAP values quantify statistical contribution within the trained pipeline and do not imply direct causal relationships.

---

## Flight Recommendation Engine

The recommendation engine (`src/recommender.py`) translates user inputs into ranked flight selections:

1. **Filtering Layer**: Filters raw flights matching source, destination, and travel class constraints.
2. **Budget Constraint**: Excludes options exceeding the user's declared maximum budget.
3. **Scoring Function**:
   $$\text{Score} = 100 - \left( 0.5 \times \frac{\text{Price}}{\text{Max Budget}} + 0.3 \times \frac{\text{Duration}}{\text{Max Duration}} + 0.2 \times \frac{\text{Stops}}{\text{Max Stops}} \right) \times 100$$
4. **Explanation Generator**: Dynamically outputs structured bullet points explaining why a recommended flight fits the user's criteria.

---

## Web Application & REST API

The backend is built with FastAPI (`app.py`) and served via Uvicorn.

### API Endpoints

- `GET /`: Serves the primary single-page web interface (`web/index.html`).
- `GET /api/kpi`: Returns overall dataset metrics (total flights, average price, total airlines, total routes).
- `GET /api/options`: Returns available dropdown choices (airlines, origins, destinations, travel classes, aircraft types, channels).
- `POST /api/predict`: Computes real-time price prediction and category (`LOW`, `TYPICAL`, `HIGH`).
- `POST /api/recommend`: Executes recommendation engine and returns ranked flight candidates with match scores.
- `GET /api/metadata`: Returns trained model metadata, feature names, and benchmark evaluation scores.

---

## Technology Stack

- **Core Language**: Python 3.10+
- **Data Manipulation & Analytics**: Pandas, NumPy
- **Machine Learning & Pipeline**: Scikit-Learn, Joblib
- **Model Explainability**: SHAP, Matplotlib, Seaborn
- **Backend Framework**: FastAPI, Uvicorn, Pydantic
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (ES6+)
- **Testing**: Python `unittest` framework

---

## Repository Structure

```text
FlightIQ/
├── app.py                     # FastAPI server and endpoint definitions
├── flight_pricing_dataset.csv # Official raw dataset (13.8 MB)
├── requirements.txt           # Python dependency requirements
├── README.md                  # Project documentation
├── .gitignore                 # Version control exclusion configuration
├── src/                       # Source modules
│   ├── data_loader.py         # Data loading utilities
│   ├── data_preprocessing.py  # Cleaning and validation routines
│   ├── feature_engineering.py # Feature extraction functions
│   ├── visualization.py       # EDA plot generation
│   ├── model_training.py      # ML model training pipeline
│   ├── model_viz.py           # Model performance visualizations
│   ├── explainability.py      # SHAP feature explainability
│   └── recommender.py         # Flight recommendation engine
├── notebooks/                 # Jupyter exploratory notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_explainability.ipynb
│   └── 07_flight_recommendation.ipynb
├── data/                      # Dataset directories
│   ├── raw/
│   │   └── flight_pricing_dataset.csv
│   └── processed/
│       └── cleaned_flight_data.csv
├── models/                    # Serialized model artifacts
│   ├── flight_price_model.joblib
│   └── model_metadata.json
├── assets/                    # Generated project plots & media
│   ├── eda/                   # 12 EDA chart images
│   ├── model/                 # 4 model performance plots
│   ├── explainability/        # 3 SHAP plots
│   └── flightiq-hero.png      # Hero visual asset
├── web/                       # Web UI static directory
│   ├── index.html             # Main HTML5 document
│   ├── styles.css             # Light-theme CSS styling
│   ├── app.js                 # Frontend JS controller
│   └── assets/                # Static asset copies
└── tests/                     # Automated unit test suite
    └── test_recommender.py    # Recommendation system unit tests
```

---

## Installation & Setup Instructions

### 1. Prerequisites
Ensure Python 3.10+ and Git are installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/srisakthipanchanathi-del/FlightIQ.git
cd FlightIQ
```

### 3. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Web Application
```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 6. Access the Application
Open your web browser and navigate to:
`http://127.0.0.1:8000`

---

## Automated Verification

To run the automated unit test suite:

```bash
python3 -m unittest discover tests/
```

Expected output:
```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.450s

OK
```

---

## License & Citation

FlightIQ is released under an open software development project license. All historical flight dataset attributes and trained model artifacts are documented for analytical and educational application.
