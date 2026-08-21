"""
src/recommender.py
------------------
FlightIQ — AI Travel Price Intelligence
Two-Stage Flight Recommendation System:
1. Stage 1 — Multi-criteria Filtering with Graceful Constraint Relaxation
2. Stage 2 — Normalized Multi-factor Transparent Scoring & Ranking
3. Fact-based "Why this flight?" Explanation Generator
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleaned_flight_data.csv"

# Default scoring weights (configurable)
DEFAULT_WEIGHTS = {
    "price": 0.35,            # Lower price gets higher score
    "budget_efficiency": 0.15,# How well it fits within budget headroom
    "stops": 0.20,            # Fewer stops gets higher score
    "duration": 0.15,         # Shorter flight gets higher score
    "preferred_airline": 0.10,# Match gets bonus
    "timing_preference": 0.05,# Closeness to preferred departure/arrival window
}


def load_recommender_dataset(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the processed flight dataset safely."""
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {path}")
    return pd.read_csv(path)


def format_duration(minutes: float) -> str:
    """Convert minutes into readable 'Xh Ym' string."""
    if pd.isna(minutes) or minutes <= 0:
        return "N/A"
    hrs = int(minutes // 60)
    mins = int(round(minutes % 60))
    if hrs == 0:
        return f"{mins}m"
    return f"{hrs}h {mins:02d}m"


def format_time_minutes(minutes: float) -> str:
    """Convert minutes from midnight to HH:MM format."""
    if pd.isna(minutes):
        return "N/A"
    hrs = int((minutes // 60) % 24)
    mins = int(round(minutes % 60))
    return f"{hrs:02d}:{mins:02d}"


def parse_time_to_minutes(val: Any) -> Optional[float]:
    """Parse time string or numeric value into minutes since midnight."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val) % 1440
    val_str = str(val).strip().upper()
    try:
        t = pd.to_datetime(val_str, format="%I:%M %p")
        return float(t.hour * 60 + t.minute)
    except Exception:
        pass
    try:
        t = pd.to_datetime(val_str, format="%H:%M")
        return float(t.hour * 60 + t.minute)
    except Exception:
        pass
    # Maybe it's a number string
    try:
        return float(val_str) % 1440
    except ValueError:
        return None


class FlightRecommender:
    def __init__(self, data: Optional[pd.DataFrame] = None, weights: Optional[Dict[str, float]] = None):
        self.df = data if data is not None else load_recommender_dataset()
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        
        # Available choices in dataset for validation
        self.available_sources = sorted(self.df["Source"].dropna().unique().tolist())
        self.available_destinations = sorted(self.df["Destination"].dropna().unique().tolist())
        self.available_classes = sorted(self.df["Travel_Class"].dropna().unique().tolist())
        self.available_airlines = sorted(self.df["Airline"].dropna().unique().tolist())
        self.available_seasons = sorted(self.df["Season"].dropna().unique().tolist()) if "Season" in self.df.columns else []

    def filter_flights(
        self,
        source: Optional[str] = None,
        destination: Optional[str] = None,
        travel_class: Optional[str] = None,
        max_budget: Optional[float] = None,
        max_stops: Optional[int] = None,
        max_duration_minutes: Optional[float] = None,
        preferred_airline: Optional[str] = None,
        season: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Stage 1: Multi-criteria Filtering with Graceful Constraint Relaxation.
        
        Hardest constraints (Origin, Destination, Travel Class) are kept fixed if possible.
        If zero flights match, optional filters (budget, duration, stops, airline, season)
        are relaxed step-by-step with informative notes.
        """
        relaxed_notes: List[str] = []
        if self.df.empty:
            return pd.DataFrame(), ["Dataset is empty."]

        filtered = self.df.copy()

        # 1. Source (case-insensitive)
        if source:
            src_clean = str(source).strip().title()
            matches = filtered[filtered["Source"].str.title() == src_clean]
            if not matches.empty:
                filtered = matches
            else:
                relaxed_notes.append(f"Origin '{source}' had no matches; showing all origins.")

        # 2. Destination (case-insensitive)
        if destination:
            dest_clean = str(destination).strip().title()
            matches = filtered[filtered["Destination"].str.title() == dest_clean]
            if not matches.empty:
                filtered = matches
            else:
                relaxed_notes.append(f"Destination '{destination}' had no matches; showing all destinations.")

        # 3. Travel Class (case-insensitive)
        if travel_class:
            class_clean = str(travel_class).strip().title()
            matches = filtered[filtered["Travel_Class"].str.title() == class_clean]
            if not matches.empty:
                filtered = matches
            else:
                relaxed_notes.append(f"Travel class '{travel_class}' not found; relaxed to all travel classes.")

        # 4. Season
        if season and "Season" in filtered.columns:
            season_clean = str(season).strip().title()
            matches = filtered[filtered["Season"].str.title() == season_clean]
            if not matches.empty:
                filtered = matches
            else:
                relaxed_notes.append(f"Season '{season}' filter relaxed due to no flights matching.")

        # 5. Preferred Airline (Soft filter in Stage 1 only if results remain)
        if preferred_airline:
            al_clean = str(preferred_airline).strip().title()
            matches = filtered[filtered["Airline"].str.title() == al_clean]
            if not matches.empty:
                # We keep preferred airline bonus in ranking
                pass

        # 6. Max Stops
        if max_stops is not None and "Total_Stops_Numeric" in filtered.columns:
            try:
                ms = float(max_stops)
                matches = filtered[filtered["Total_Stops_Numeric"] <= ms]
                if not matches.empty:
                    filtered = matches
                else:
                    relaxed_notes.append(f"Relaxed max stops constraint ({int(ms)} stops) as no flights matched.")
            except (ValueError, TypeError):
                pass

        # 7. Max Duration
        if max_duration_minutes is not None and "Duration_Minutes" in filtered.columns:
            try:
                md = float(max_duration_minutes)
                matches = filtered[filtered["Duration_Minutes"] <= md]
                if not matches.empty:
                    filtered = matches
                else:
                    relaxed_notes.append(f"Relaxed max journey duration ({format_duration(md)}) as no flights matched.")
            except (ValueError, TypeError):
                pass

        # 8. Max Budget
        if max_budget is not None and "Price" in filtered.columns:
            try:
                mb = float(max_budget)
                if mb > 0:
                    matches = filtered[filtered["Price"] <= mb]
                    if not matches.empty:
                        filtered = matches
                    else:
                        lowest_price = filtered["Price"].min()
                        relaxed_notes.append(
                            f"Budget constraint (₹{mb:,.0f}) was too low (lowest available: ₹{lowest_price:,.0f}); relaxed budget filter."
                        )
            except (ValueError, TypeError):
                pass

        return filtered, relaxed_notes

    def rank_flights(
        self,
        candidates: pd.DataFrame,
        max_budget: Optional[float] = None,
        preferred_airline: Optional[str] = None,
        pref_dep_time_minutes: Optional[float] = None,
        pref_arr_time_minutes: Optional[float] = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Stage 2: Normalized Multi-factor Transparent Scoring & Ranking.
        
        Calculates a composite match score [0 to 100] for each candidate row:
        - Price Score (normalized inverse of price)
        - Budget Headroom Score (savings relative to budget)
        - Stops Score (0 stops = 1.0, 1 stop = 0.5, 2 stops = 0.0)
        - Duration Score (normalized inverse of duration)
        - Preferred Airline Bonus (1.0 if match, 0.0 otherwise)
        - Time Preference Score (proximity to preferred departure/arrival window)
        """
        if candidates.empty:
            return pd.DataFrame()

        scored = candidates.copy()
        n = len(scored)

        # 1. Normalized Price Score [0, 1] (Lower price -> Higher score)
        min_p = scored["Price"].min()
        max_p = scored["Price"].max()
        if max_p > min_p:
            scored["score_price"] = 1.0 - ((scored["Price"] - min_p) / (max_p - min_p))
        else:
            scored["score_price"] = 1.0

        # 2. Budget Efficiency Score [0, 1]
        if max_budget and max_budget > 0:
            # Score how much below budget it is (0 if above budget, 1 if significantly below)
            scored["score_budget"] = scored["Price"].apply(
                lambda p: max(0.0, min(1.0, (max_budget - p) / max_budget)) if p <= max_budget else 0.0
            )
        else:
            scored["score_budget"] = scored["score_price"]

        # 3. Stops Score [0, 1]
        if "Total_Stops_Numeric" in scored.columns:
            # 0 stops -> 1.0, 1 stop -> 0.5, 2+ stops -> 0.0
            scored["score_stops"] = scored["Total_Stops_Numeric"].apply(
                lambda s: 1.0 if s == 0 else (0.5 if s == 1 else 0.0)
            )
        else:
            scored["score_stops"] = 0.5

        # 4. Duration Score [0, 1] (Shorter duration -> Higher score)
        if "Duration_Minutes" in scored.columns:
            min_d = scored["Duration_Minutes"].min()
            max_d = scored["Duration_Minutes"].max()
            if max_d > min_d:
                scored["score_duration"] = 1.0 - ((scored["Duration_Minutes"] - min_d) / (max_d - min_d))
            else:
                scored["score_duration"] = 1.0
        else:
            scored["score_duration"] = 0.5

        # 5. Preferred Airline Bonus [0, 1]
        if preferred_airline:
            al_clean = str(preferred_airline).strip().title()
            scored["score_airline"] = scored["Airline"].apply(
                lambda a: 1.0 if str(a).strip().title() == al_clean else 0.0
            )
        else:
            scored["score_airline"] = 0.0

        # 6. Timing Preference Score [0, 1]
        time_scores = np.ones(n)
        if pref_dep_time_minutes is not None and "Departure_Time_Minutes" in scored.columns:
            # Difference in minutes circular across 24 hours
            diffs = scored["Departure_Time_Minutes"].apply(
                lambda t: min(abs(t - pref_dep_time_minutes), 1440 - abs(t - pref_dep_time_minutes))
            )
            # Max difference is 720 minutes (12h). Closer is higher score.
            time_scores = (1.0 - (diffs / 720.0)).clip(0.0, 1.0).values

        if pref_arr_time_minutes is not None and "Arrival_Time_Minutes" in scored.columns:
            arr_diffs = scored["Arrival_Time_Minutes"].apply(
                lambda t: min(abs(t - pref_arr_time_minutes), 1440 - abs(t - pref_arr_time_minutes))
            )
            arr_scores = (1.0 - (arr_diffs / 720.0)).clip(0.0, 1.0).values
            time_scores = (time_scores + arr_scores) / 2.0

        scored["score_timing"] = time_scores

        # Weighted Composite Score Calculation
        w = self.weights
        scored["Total_Score"] = (
            w.get("price", 0.35) * scored["score_price"] +
            w.get("budget_efficiency", 0.15) * scored["score_budget"] +
            w.get("stops", 0.20) * scored["score_stops"] +
            w.get("duration", 0.15) * scored["score_duration"] +
            w.get("preferred_airline", 0.10) * scored["score_airline"] +
            w.get("timing_preference", 0.05) * scored["score_timing"]
        ) * 100.0  # Scale to 0-100

        # Sort descending by Total_Score, break ties by lowest price
        ranked = scored.sort_values(by=["Total_Score", "Price"], ascending=[False, True]).head(top_k)
        return ranked

    def generate_explanations(
        self,
        flight_row: pd.Series,
        all_candidates: pd.DataFrame,
        max_budget: Optional[float] = None,
        preferred_airline: Optional[str] = None,
        max_stops: Optional[int] = None,
        max_duration_minutes: Optional[float] = None,
        pref_dep_time_minutes: Optional[float] = None,
    ) -> List[str]:
        """
        Generate a factual, truthful "Why this flight?" explanation list.
        Only claims benefits that are genuinely satisfied by this specific flight record.
        """
        reasons: List[str] = []

        price = flight_row.get("Price", 0)
        stops = flight_row.get("Total_Stops_Numeric", None)
        duration = flight_row.get("Duration_Minutes", None)
        airline = str(flight_row.get("Airline", ""))
        dep_min = flight_row.get("Departure_Time_Minutes", None)

        # 1. Budget Reasons
        if max_budget is not None and max_budget > 0:
            if price <= max_budget:
                savings = max_budget - price
                if savings > 0.15 * max_budget:
                    reasons.append(f"Within your budget (saves ₹{savings:,.0f} from max budget)")
                else:
                    reasons.append(f"Within your specified budget of ₹{max_budget:,.0f}")
        
        # 2. Competitive / Lowest Price Reason
        if not all_candidates.empty and "Price" in all_candidates.columns:
            min_cand_price = all_candidates["Price"].min()
            p25_cand_price = all_candidates["Price"].quantile(0.25)
            if price == min_cand_price:
                reasons.append("Lowest price flight among all available options on this route")
            elif price <= p25_cand_price:
                reasons.append("Highly competitive fare (in the cheapest 25% of available flights)")

        # 3. Stops Reason
        if stops == 0:
            reasons.append("Direct non-stop flight for maximum convenience")
        elif stops == 1:
            if max_stops is not None and max_stops >= 1:
                reasons.append("1 stop connecting flight meeting your stops preference")
            else:
                reasons.append("Single stop connecting route")

        # 4. Duration Reason
        if duration is not None and not all_candidates.empty and "Duration_Minutes" in all_candidates.columns:
            min_cand_dur = all_candidates["Duration_Minutes"].min()
            if duration == min_cand_dur:
                reasons.append(f"Fastest journey on this route ({format_duration(duration)})")
            elif duration <= all_candidates["Duration_Minutes"].quantile(0.30):
                reasons.append(f"Shorter journey time ({format_duration(duration)})")

        # 5. Preferred Airline Reason
        if preferred_airline and airline.strip().title() == str(preferred_airline).strip().title():
            reasons.append(f"Operated by your preferred airline ({airline})")

        # 6. Preferred Departure Time Reason
        if pref_dep_time_minutes is not None and dep_min is not None:
            diff = min(abs(dep_min - pref_dep_time_minutes), 1440 - abs(dep_min - pref_dep_time_minutes))
            if diff <= 60:
                reasons.append(f"Departs close to your preferred time ({format_time_minutes(dep_min)})")

        # Fallback if no specific reason triggered
        if not reasons:
            reasons.append("Balanced option optimizing overall price, timing, and travel duration")

        return reasons

    def recommend(
        self,
        source: Optional[str] = None,
        destination: Optional[str] = None,
        travel_class: Optional[str] = None,
        max_budget: Optional[float] = None,
        preferred_airline: Optional[str] = None,
        max_stops: Optional[int] = None,
        max_duration_minutes: Optional[float] = None,
        preferred_departure_time: Optional[Any] = None,
        preferred_arrival_time: Optional[Any] = None,
        season: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        End-to-End Safe Recommendation Pipeline:
        1. Parse and sanitize all user preferences
        2. Stage 1: Filter flights (with constraint relaxation if needed)
        3. Stage 2: Rank filtered candidate flights with composite scoring
        4. Generate fact-based 'Why this flight?' reasons
        5. Return structured response with top recommendations and relaxation metadata
        """
        # Parse time preferences safely
        dep_time_min = parse_time_to_minutes(preferred_departure_time)
        arr_time_min = parse_time_to_minutes(preferred_arrival_time)

        # Stage 1: Filter
        candidates, relaxation_notes = self.filter_flights(
            source=source,
            destination=destination,
            travel_class=travel_class,
            max_budget=max_budget,
            max_stops=max_stops,
            max_duration_minutes=max_duration_minutes,
            preferred_airline=preferred_airline,
            season=season,
        )

        if candidates.empty:
            return {
                "status": "NO_MATCH",
                "count": 0,
                "recommendations": [],
                "relaxation_notes": relaxation_notes or ["No flights in dataset matching the requested query."],
                "query": {
                    "source": source,
                    "destination": destination,
                    "travel_class": travel_class,
                    "max_budget": max_budget,
                    "preferred_airline": preferred_airline,
                    "max_stops": max_stops,
                    "max_duration_minutes": max_duration_minutes,
                }
            }

        # Stage 2: Rank
        ranked_df = self.rank_flights(
            candidates=candidates,
            max_budget=max_budget,
            preferred_airline=preferred_airline,
            pref_dep_time_minutes=dep_time_min,
            pref_arr_time_minutes=arr_time_min,
            top_k=top_k,
        )

        recommendations_list: List[Dict[str, Any]] = []
        for rank_idx, (_, row) in enumerate(ranked_df.iterrows(), start=1):
            explanations = self.generate_explanations(
                flight_row=row,
                all_candidates=candidates,
                max_budget=max_budget,
                preferred_airline=preferred_airline,
                max_stops=max_stops,
                max_duration_minutes=max_duration_minutes,
                pref_dep_time_minutes=dep_time_min,
            )

            rec_item = {
                "rank": rank_idx,
                "flight_id": str(row.get("Flight_ID", f"FL_{rank_idx}")),
                "airline": str(row.get("Airline", "Unknown")),
                "source": str(row.get("Source", "Unknown")),
                "destination": str(row.get("Destination", "Unknown")),
                "travel_class": str(row.get("Travel_Class", "Unknown")),
                "price": float(row.get("Price", 0.0)),
                "price_formatted": f"₹{float(row.get('Price', 0.0)):,.2f}",
                "stops": int(row.get("Total_Stops_Numeric", 0)),
                "stops_formatted": "Non-Stop" if row.get("Total_Stops_Numeric", 0) == 0 else f"{int(row.get('Total_Stops_Numeric', 0))} Stop(s)",
                "duration_minutes": float(row.get("Duration_Minutes", 0.0)),
                "duration_formatted": format_duration(float(row.get("Duration_Minutes", 0.0))),
                "departure_time": str(row.get("Departure_Time", format_time_minutes(row.get("Departure_Time_Minutes", 0)))),
                "arrival_time": str(row.get("Arrival_Time", format_time_minutes(row.get("Arrival_Time_Minutes", 0)))),
                "match_score": round(float(row.get("Total_Score", 0.0)), 1),
                "why_this_flight": explanations,
            }
            recommendations_list.append(rec_item)

        return {
            "status": "SUCCESS",
            "count": len(recommendations_list),
            "total_matching_candidates": len(candidates),
            "recommendations": recommendations_list,
            "relaxation_notes": relaxation_notes,
            "query": {
                "source": source,
                "destination": destination,
                "travel_class": travel_class,
                "max_budget": max_budget,
                "preferred_airline": preferred_airline,
                "max_stops": max_stops,
                "max_duration_minutes": max_duration_minutes,
                "preferred_departure_time": preferred_departure_time,
                "preferred_arrival_time": preferred_arrival_time,
                "season": season,
            }
        }


def get_flight_recommendations(
    source: Optional[str] = None,
    destination: Optional[str] = None,
    travel_class: Optional[str] = None,
    max_budget: Optional[float] = None,
    preferred_airline: Optional[str] = None,
    max_stops: Optional[int] = None,
    max_duration_minutes: Optional[float] = None,
    preferred_departure_time: Optional[Any] = None,
    preferred_arrival_time: Optional[Any] = None,
    season: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Functional convenience entrypoint for the recommender engine."""
    recommender = FlightRecommender()
    return recommender.recommend(
        source=source,
        destination=destination,
        travel_class=travel_class,
        max_budget=max_budget,
        preferred_airline=preferred_airline,
        max_stops=max_stops,
        max_duration_minutes=max_duration_minutes,
        preferred_departure_time=preferred_departure_time,
        preferred_arrival_time=preferred_arrival_time,
        season=season,
        top_k=top_k,
    )


if __name__ == "__main__":
    rec = FlightRecommender()
    print("Testing recommendation engine...")
    res = rec.recommend(
        source="Mumbai",
        destination="Goa",
        travel_class="Economy",
        max_budget=10000,
        max_stops=0,
        preferred_airline="Indigo",
    )
    print(f"Status: {res['status']}, Found: {res['count']} recommendations")
    for r in res["recommendations"]:
        print(f"\n#{r['rank']}: {r['airline']} | {r['source']} → {r['destination']} | {r['travel_class']} | {r['price_formatted']} | {r['stops_formatted']} | {r['duration_formatted']}")
        print(f"  Score: {r['match_score']}/100")
        print(f"  Why: {'; '.join(r['why_this_flight'])}")
