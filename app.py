"""
app.py
------
FlightIQ — AI Travel Price Intelligence
FastAPI backend server for serving API endpoints, static assets, and ML model inference.
"""

import sys
from pathlib import Path
import json
import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlightIQ")

# Add workspace to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.recommender import FlightRecommender, load_recommender_dataset, get_flight_recommendations

app = FastAPI(title="FlightIQ API", description="AI Travel Price Intelligence Engine API", version="2.0.0")

# Paths
MODEL_PATH = BASE_DIR / "models" / "flight_price_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATASET_PATH = BASE_DIR / "data" / "processed" / "cleaned_flight_data.csv"
RAW_DATASET_PATH = BASE_DIR / "data" / "raw" / "flight_pricing_dataset.csv"
ROOT_DATASET_PATH = BASE_DIR / "flight_pricing_dataset.csv"

# Load dataset safely with fallbacks
df_raw = pd.DataFrame()
recommender_engine = None

for path in [DATASET_PATH, RAW_DATASET_PATH, ROOT_DATASET_PATH]:
    if path.exists():
        try:
            df_raw = load_recommender_dataset(path)
            recommender_engine = FlightRecommender(data=df_raw)
            logger.info(f"Successfully loaded dataset from {path} ({len(df_raw)} records)")
            break
        except Exception as e:
            logger.warning(f"Warning loading dataset from {path}: {e}")

# Diagnostic logging & model initialization
logger.info(f"MODEL_PATH: {MODEL_PATH}")
logger.info(f"MODEL_EXISTS: {MODEL_PATH.exists()}")
logger.info(f"DATA_PATH: {DATASET_PATH}")
logger.info(f"DATA_EXISTS: {DATASET_PATH.exists()}")

model_pipeline = None
if MODEL_PATH.exists():
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        logger.info("FlightIQ model loaded successfully")
        logger.info(f"MODEL_PIPELINE_TYPE: {type(model_pipeline)}")
        logger.info("MODEL_PIPELINE_LOADED: True")
    except Exception as e:
        logger.error(f"Error loading model pipeline from {MODEL_PATH}: {e}", exc_info=True)
        logger.info("MODEL_PIPELINE_LOADED: False")
else:
    logger.error(f"Model file does not exist at expected path: {MODEL_PATH}")
    logger.info("MODEL_PIPELINE_LOADED: False")


class PredictionRequest(BaseModel):
    source: str = Field(default="Mumbai")
    destination: str = Field(default="Delhi")
    travel_class: str = Field(default="Economy")
    airline: str = Field(default="Indigo")
    days_before_departure: int = Field(default=14)
    duration_minutes: float = Field(default=130.0)
    total_stops: int = Field(default=0)
    departure_month: int = Field(default=6)
    departure_day_of_week_num: int = Field(default=3)
    departure_time_minutes: int = Field(default=600)
    arrival_time_minutes: int = Field(default=730)
    passenger_count: int = Field(default=1)
    season: str = Field(default="Summer")
    weekday: str = Field(default="Wednesday")
    aircraft_type: str = Field(default="Airbus A320")
    booking_channel: str = Field(default="Website")
    distance_km: Optional[float] = Field(default=None)


class RecommendationRequest(BaseModel):
    source: Optional[str] = None
    destination: Optional[str] = None
    travel_class: Optional[str] = None
    max_budget: Optional[float] = None
    preferred_airline: Optional[str] = None
    max_stops: Optional[int] = None
    max_duration_minutes: Optional[float] = None
    season: Optional[str] = None
    top_k: int = 5


@app.api_route("/", methods=["GET", "HEAD"])
def serve_root_index():
    """Serve index.html at root route for GET and HEAD requests."""
    index_file = BASE_DIR / "web" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/api/debug/model")
def debug_model():
    """Temporary diagnostic endpoint for inspecting loaded global model pipeline state."""
    return {
        "loaded": model_pipeline is not None,
        "type": str(type(model_pipeline)),
        "steps": list(model_pipeline.named_steps.keys()) if model_pipeline is not None and hasattr(model_pipeline, "named_steps") else []
    }


@app.get("/api/kpi")
def get_kpi_metrics():
    """Return dataset KPI metrics including required KPIs."""
    if df_raw.empty:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")
    
    total_flights = len(df_raw)
    avg_price = float(df_raw["Price"].mean())
    min_price = float(df_raw["Price"].min())
    max_price = float(df_raw["Price"].max())
    num_airlines = int(df_raw["Airline"].nunique())
    routes_series = df_raw["Source"].astype(str) + " → " + df_raw["Destination"].astype(str)
    num_routes = int(routes_series.nunique())

    return {
        "status": "success",
        "kpis": {
            "flights_analyzed": total_flights,
            "flights_analyzed_formatted": f"{total_flights:,}",
            "average_price": round(avg_price, 2),
            "average_price_formatted": f"₹{round(avg_price):,}",
            "lowest_price": round(min_price, 2),
            "lowest_price_formatted": f"₹{round(min_price):,}",
            "highest_price": round(max_price, 2),
            "highest_price_formatted": f"₹{round(max_price):,}",
            "airlines_count": num_airlines,
            "routes_count": num_routes,
        }
    }


@app.get("/api/options")
def get_dropdown_options():
    """Return available unique options for form dropdowns."""
    if df_raw.empty:
        return {}
    
    sources = sorted([str(x) for x in df_raw["Source"].dropna().unique()])
    destinations = sorted([str(x) for x in df_raw["Destination"].dropna().unique()])
    airlines = sorted([str(x) for x in df_raw["Airline"].dropna().unique()])
    classes = sorted([str(x) for x in df_raw["Travel_Class"].dropna().unique()])
    seasons = sorted([str(x) for x in df_raw["Season"].dropna().unique()]) if "Season" in df_raw.columns else []
    aircrafts = sorted([str(x) for x in df_raw["Aircraft_Type"].dropna().unique()]) if "Aircraft_Type" in df_raw.columns else []
    channels = sorted([str(x) for x in df_raw["Booking_Channel"].dropna().unique()]) if "Booking_Channel" in df_raw.columns else []
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    return {
        "sources": sources,
        "destinations": destinations,
        "airlines": airlines,
        "travel_classes": classes,
        "seasons": seasons,
        "aircraft_types": aircrafts,
        "booking_channels": channels,
        "weekdays": weekdays
    }


@app.post("/api/predict")
def predict_flight_price(req: PredictionRequest):
    """Predict flight price using saved trained scikit-learn model pipeline."""
    logger.info(f"PREDICT model_pipeline: {model_pipeline}")
    logger.info(f"PREDICT model_pipeline type: {type(model_pipeline)}")
    logger.info(f"PREDICT model_pipeline is None: {model_pipeline is None}")
    if model_pipeline is not None and hasattr(model_pipeline, "named_steps"):
        logger.info(f"PREDICT model steps: {list(model_pipeline.named_steps.keys())}")

    if model_pipeline is None:
        logger.error("Prediction request failed: model_pipeline is None")
        raise HTTPException(status_code=500, detail="Model pipeline not available.")

    dist = req.distance_km
    if dist is None:
        if not df_raw.empty:
            match = df_raw[(df_raw["Source"].str.title() == req.source.title()) & 
                           (df_raw["Destination"].str.title() == req.destination.title())]
            if not match.empty and "Distance_km" in match.columns:
                dist = float(match["Distance_km"].mean())
        if dist is None or np.isnan(dist):
            dist = max(300.0, req.duration_minutes * 8.5)

    input_dict = {
        "Distance_km": float(dist),
        "Duration_Minutes": float(req.duration_minutes),
        "Total_Stops_Numeric": float(req.total_stops),
        "Days_Before_Departure": float(req.days_before_departure),
        "Departure_Time_Minutes": float(req.departure_time_minutes),
        "Arrival_Time_Minutes": float(req.arrival_time_minutes),
        "Departure_Month": float(req.departure_month),
        "Departure_DayOfWeek_Num": float(req.departure_day_of_week_num),
        "Passenger_Count": float(req.passenger_count),
        "Airline": req.airline,
        "Source": req.source,
        "Destination": req.destination,
        "Travel_Class": req.travel_class,
        "Season": req.season,
        "Weekday": req.weekday,
        "Aircraft_Type": req.aircraft_type,
        "Booking_Channel": req.booking_channel,
    }

    input_df = pd.DataFrame([input_dict])

    try:
        pred_val = float(model_pipeline.predict(input_df)[0])
        pred_val = max(0.0, round(pred_val, 2))

        # Calculate category (LOW / TYPICAL / HIGH) based on class / route / dataset distribution
        subset = pd.DataFrame()
        if not df_raw.empty:
            subset = df_raw[df_raw["Travel_Class"].str.title() == req.travel_class.title()]
        
        if not subset.empty:
            p25 = float(subset["Price"].quantile(0.25))
            p75 = float(subset["Price"].quantile(0.75))
        else:
            p25 = 13258.0
            p75 = 113660.0

        if pred_val <= p25:
            category = "LOW"
        elif pred_val >= p75:
            category = "HIGH"
        else:
            category = "TYPICAL"

        return {
            "status": "success",
            "predicted_price": pred_val,
            "predicted_price_formatted": f"₹{round(pred_val):,}",
            "price_category": category,
            "p25_formatted": f"₹{round(p25):,}",
            "p75_formatted": f"₹{round(p75):,}",
            "currency": "INR",
            "disclaimer": "Prediction generated from historical flight pricing patterns."
        }
    except Exception as e:
        logger.error(f"Inference execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference execution error: {str(e)}")


@app.post("/api/recommend")
def recommend_flights(req: RecommendationRequest):
    """Run two-stage flight recommendation engine."""
    global recommender_engine
    if recommender_engine is None and not df_raw.empty:
        recommender_engine = FlightRecommender(data=df_raw)
    if recommender_engine is None:
        raise HTTPException(status_code=500, detail="Recommender engine not available.")

    res = recommender_engine.recommend(
        source=req.source,
        destination=req.destination,
        travel_class=req.travel_class,
        max_budget=req.max_budget,
        preferred_airline=req.preferred_airline,
        max_stops=req.max_stops,
        max_duration_minutes=req.max_duration_minutes,
        season=req.season,
        top_k=req.top_k,
    )
    return res


@app.get("/api/metadata")
def get_model_metadata():
    """Return model performance metrics and feature metadata."""
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    return meta


# Static file mounts
web_path = BASE_DIR / "web"
if web_path.exists():
    app.mount("/static", StaticFiles(directory=str(web_path)), name="static")

app.mount("/assets", StaticFiles(directory=str(BASE_DIR / "assets")), name="assets")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
