"""
NagrikAI - Unit Tests: Application Startup & Integrity
"""

import unittest
import sys
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.config import (
    CLASSES, PALETTE, RAW_DATA_DIR, CIVIC_DATA_DIR,
    DATA_QUALITY_REPORT_PATH
)

class TestAppIntegrity(unittest.TestCase):
    
    def test_classes_integrity(self):
        self.assertEqual(len(CLASSES), 10)
        self.assertIn("plastic", CLASSES)
        self.assertIn("battery", CLASSES)
        self.assertIn("trash", CLASSES)
        
    def test_palette_styling(self):
        self.assertIn("cream_bg", PALETTE)
        self.assertIn("deep_brown", PALETTE)
        self.assertIn("earth_accent", PALETTE)
        
    def test_civic_data_files_exist(self):
        reports_file = CIVIC_DATA_DIR / "demo_reports.csv"
        hotspots_file = CIVIC_DATA_DIR / "hotspots.csv"
        
        self.assertTrue(reports_file.exists(), "demo_reports.csv should exist")
        self.assertTrue(hotspots_file.exists(), "hotspots.csv should exist")
        
        with open(reports_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertGreaterEqual(len(rows), 20)
            fieldnames = reader.fieldnames or []
            self.assertIn("waste_category", fieldnames)
            self.assertIn("severity", fieldnames)
            self.assertIn("drain_risk", fieldnames)
            
        with open(hotspots_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertGreaterEqual(len(rows), 5)
            fieldnames = reader.fieldnames or []
            self.assertIn("priority_score", fieldnames)
        
    def test_quality_report_exists(self):
        self.assertTrue(DATA_QUALITY_REPORT_PATH.exists(), "Data quality report should exist")

if __name__ == "__main__":
    unittest.main()
