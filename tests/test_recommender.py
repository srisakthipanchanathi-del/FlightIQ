"""
tests/test_recommender.py
-------------------------
Unit tests for the FlightIQ Recommendation Engine.
Covers:
- Basic route & class filtering
- Budget constraint handling & relaxation
- Stops constraint handling
- Ranking & match score order
- Empty result handling / invalid inputs
- Output schema & explanation generation
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root / src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recommender import FlightRecommender, get_flight_recommendations


class TestFlightRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = FlightRecommender()
        cls.df = cls.recommender.df
        assert not cls.df.empty, "Dataset is empty for tests"

    def test_basic_filtering(self):
        """Test that filtering returns valid flights between source and destination."""
        source = "Mumbai"
        dest = "Goa"
        res = self.recommender.recommend(source=source, destination=dest, top_k=5)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["count"], 0)
        for flight in res["recommendations"]:
            self.assertEqual(flight["source"], source)
            self.assertEqual(flight["destination"], dest)

    def test_budget_constraint(self):
        """Test that recommendation respects maximum budget when available."""
        source = "Mumbai"
        dest = "Goa"
        budget = 5000.0
        res = self.recommender.recommend(source=source, destination=dest, max_budget=budget, top_k=5)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["count"], 0)
        for flight in res["recommendations"]:
            self.assertLessEqual(flight["price"], budget)

    def test_stops_constraint(self):
        """Test that max stops constraint is strictly adhered to."""
        res = self.recommender.recommend(source="Delhi", destination="London", max_stops=0, top_k=5)
        self.assertIn(res["status"], ["SUCCESS", "NO_MATCH"])
        if res["status"] == "SUCCESS":
            for flight in res["recommendations"]:
                # If direct flights exist and weren't relaxed
                if not any("Relaxed max stops" in n for n in res["relaxation_notes"]):
                    self.assertEqual(flight["stops"], 0)

    def test_ranking_order(self):
        """Test that recommended flights are sorted in descending order of match_score."""
        res = self.recommender.recommend(source="Bangalore", destination="Delhi", travel_class="Economy", top_k=5)
        self.assertEqual(res["status"], "SUCCESS")
        scores = [f["match_score"] for f in res["recommendations"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_or_invalid_inputs(self):
        """Test that system gracefully handles non-existent or impossible queries without crashing."""
        # Non-existent route
        res = self.recommender.recommend(source="Atlantis", destination="Moon", max_budget=10.0, top_k=5)
        self.assertIn(res["status"], ["SUCCESS", "NO_MATCH"])
        self.assertIsInstance(res["recommendations"], list)
        self.assertIsInstance(res["relaxation_notes"], list)

    def test_output_schema_and_explanations(self):
        """Test that output contains all required fields and factual 'Why this flight?' reasons."""
        res = self.recommender.recommend(
            source="Hyderabad",
            destination="Ahmedabad",
            travel_class="Economy",
            max_budget=20000.0,
            preferred_airline="Indigo",
            top_k=5
        )
        self.assertEqual(res["status"], "SUCCESS")
        required_fields = [
            "rank", "flight_id", "airline", "source", "destination",
            "travel_class", "price", "price_formatted", "stops", "stops_formatted",
            "duration_minutes", "duration_formatted", "departure_time", "arrival_time",
            "match_score", "why_this_flight"
        ]
        for flight in res["recommendations"]:
            for field in required_fields:
                self.assertIn(field, flight)
            self.assertIsInstance(flight["why_this_flight"], list)
            self.assertGreater(len(flight["why_this_flight"]), 0)


if __name__ == "__main__":
    unittest.main()
