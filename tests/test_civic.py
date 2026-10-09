"""
NagrikAI - Unit Tests: Civic Intelligence & Hotspot Engine
"""

import unittest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.civic.risk import get_civic_profile, CIVIC_KNOWLEDGE_BASE
from src.civic.recommendations import generate_action_plan
from src.civic.hotspot import calculate_hotspot_priority_score, generate_root_cause_hypothesis
from src.utils.config import CLASSES

class TestCivicIntelligence(unittest.TestCase):
    
    def test_all_classes_mapped(self):
        for c in CLASSES:
            profile = get_civic_profile(c)
            self.assertIn("civic_category", profile)
            self.assertIn("environmental_risk", profile)
            self.assertIn("drain_risk", profile)
            self.assertIn("recommended_action", profile)
            
    def test_battery_critical_hazard(self):
        battery_profile = get_civic_profile("battery")
        self.assertEqual(battery_profile["environmental_risk"], "CRITICAL")
        self.assertEqual(battery_profile["segregation_stream"], "Hazardous Waste (Red Bin / Special Collection)")
        
    def test_action_plan_generation(self):
        plan = generate_action_plan("plastic", probability=0.92)
        self.assertEqual(plan["class"], "Plastic")
        self.assertIn("citizen", plan)
        self.assertIn("sanitation_workers", plan)
        self.assertIn("municipal_authority", plan)
        
    def test_hotspot_priority_scoring(self):
        score_data = calculate_hotspot_priority_score(
            recurrence_count=8,
            severity_level="CRITICAL",
            frequency_reports=10,
            drain_risk_level="CRITICAL"
        )
        self.assertGreaterEqual(score_data["score"], 80)
        self.assertEqual(score_data["priority_level"], "CRITICAL")
        self.assertIn("components", score_data)
        
    def test_root_cause_hypothesis(self):
        sample_hotspot = {
            "ward": "Ward 23 - Koramangala",
            "primary_waste": "Plastic",
            "drain_risk": "CRITICAL",
            "recurrence_count": 5
        }
        hypothesis = generate_root_cause_hypothesis(sample_hotspot)
        self.assertIn("contributing_factors", hypothesis)
        self.assertIn("hypothesis", hypothesis)
        self.assertTrue(len(hypothesis["contributing_factors"]) > 0)

if __name__ == "__main__":
    unittest.main()

