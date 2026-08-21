# FlightIQ — AI Travel Price Intelligence

FlightIQ is an end-to-end Machine Learning web application designed to predict Indian domestic flight prices, deliver interactive fare analytics, explain predictive pricing drivers using SHAP (SHapley Additive exPlanations), and provide personalized flight recommendations based on travel preferences.

---

## Key Features

- **Real-Time Fare Prediction**: Predict flight prices based on travel origin, destination, travel class, airline, lead time, duration, and departure month.
- **Flight Recommendation Engine**: Receive multi-criteria ranked flight suggestions tailored to budget, preferred travel windows, and airline quality preferences.
- **Explainable AI (XAI)**: Understand what drives flight pricing using global feature importance and SHAP beeswarm plots.
- **Interactive Data Explorer**: Click-to-expand visualizations with key figures, statistical context, and empirical interpretations.
- **Cinematic Translucent UI**: Fixed parallax background identity featuring light theme styling, translucent card panels, and smooth responsive controls.

---

## Model Benchmarks & Metrics

Trained on **93,083 domestic flight records**, the model comparison evaluated Linear Regression, Random Forest, and Gradient Boosting algorithms on an independent 20% test split (18,617 instances).

| Model | $R^2$ Score | MAE (₹) | RMSE (₹) |
| :--- | :---: | :---: | :---: |
| Linear Regression | 0.6217 | ₹17,845.20 | ₹45,512.30 |
| Random Forest | 0.6902 | ₹14,810.15 | ₹41,120.50 |
| **Gradient Boosting (Selected)** | **0.7049** | **₹14,262.47** | **₹40,261.86** |

---

## Dataset Overview

- **Source Dataset**: `data/raw/flight_pricing_dataset.csv`
- **Total Rows**: 93,083 flights
- **Airlines Tracked**: 13 carriers (IndiGo, SpiceJet, Air India, Vistara, AirAsia, Go First, Emirates, Lufthansa, etc.)
- **Unique Routes**: 308 origin-destination routes
- **Price Range**: ₹152 to ₹999,306 (Mean: ₹72,990)

---

## Project Structure

```text
FlightIQ/
├── app.py                     # FastAPI backend application
├── flight_pricing_dataset.csv # Official raw flight dataset
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── src/                       # Modular Python source code
│   ├── data_loader.py         # Data loading and schema validation
│   ├── data_preprocessing.py  # Data cleaning and missing value handling
│   ├── feature_engineering.py # Derived feature transformations
│   ├── visualization.py       # Exploratory Data Analysis (EDA) plotting
│   ├── model_training.py      # ML model training and hyperparameter pipeline
│   ├── model_viz.py           # Evaluation plots (Actual vs Pred, Residuals)
│   ├── explainability.py      # SHAP feature importance scripts
│   └── recommender.py         # Multi-attribute flight ranking engine
├── notebooks/                 # Jupyter exploratory notebooks (01 to 07)
├── models/                    # Trained pipeline (.joblib) and metadata (.json)
├── assets/                    # Generated charts and visualizations
└── web/                       # Web frontend (HTML, CSS, JS, Assets)
```

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/srisakthipanchanathi-del/FlightIQ.git
   cd FlightIQ
   ```

2. **Set Up Python Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run Web Application**:
   ```bash
   python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. **Access UI**:
   Open `http://127.0.0.1:8000` in your web browser.

---

## License & Attribution

Designed and built for FlightIQ AI Travel Price Intelligence.
