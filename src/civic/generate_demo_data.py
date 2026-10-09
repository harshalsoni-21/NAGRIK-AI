"""
NagrikAI - Synthetic Civic Demo Data Generator
Creates realistic synthetic civic reports and hotspots for demonstration.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..utils.config import CIVIC_DATA_DIR
from .hotspot import calculate_hotspot_priority_score

# Sample wards with baseline Indian city coordinates (Bengaluru / Pune / Delhi hybrid civic coords)
WARDS = [
    {"ward": "Ward 23 - Koramangala", "lat": 12.9352, "lon": 77.6245, "recurring": True},
    {"ward": "Ward 12 - Indiranagar", "lat": 12.9784, "lon": 77.6408, "recurring": True},
    {"ward": "Ward 18 - Malleshwaram", "lat": 13.0031, "lon": 77.5643, "recurring": True},
    {"ward": "Ward 45 - Shivajinagar", "lat": 12.9866, "lon": 77.6012, "recurring": True},
    {"ward": "Ward 07 - Jayanagar", "lat": 12.9299, "lon": 77.5824, "recurring": False},
    {"ward": "Ward 31 - Whitefield", "lat": 12.9698, "lon": 77.7499, "recurring": True},
    {"ward": "Ward 15 - HSR Layout", "lat": 12.9121, "lon": 77.6446, "recurring": False},
    {"ward": "Ward 28 - Rajajinagar", "lat": 12.9982, "lon": 77.5530, "recurring": False},
    {"ward": "Ward 52 - Basavanagudi", "lat": 12.9421, "lon": 77.5752, "recurring": False},
    {"ward": "Ward 09 - Electronic City", "lat": 12.8452, "lon": 77.6602, "recurring": True},
]

WASTE_TYPES = ["Plastic", "Trash", "Biological", "Cardboard", "Clothes", "Metal", "Glass", "Battery"]
SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
DRAIN_RISKS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
STATUSES = ["Pending", "Assigned", "In Progress", "Resolved"]

def generate_civic_data():
    CIVIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    
    # Generate 45 reports
    reports = []
    base_time = datetime.now() - timedelta(days=14)
    
    for i in range(1, 46):
        ward_info = np.random.choice(WARDS)
        # Add slight jitter to coordinates (within 100-300m)
        lat = ward_info["lat"] + np.random.uniform(-0.003, 0.003)
        lon = ward_info["lon"] + np.random.uniform(-0.003, 0.003)
        
        waste = np.random.choice(WASTE_TYPES, p=[0.30, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05])
        
        if waste in ["Plastic", "Clothes"]:
            drain_risk = np.random.choice(["HIGH", "CRITICAL"], p=[0.4, 0.6])
        elif waste == "Battery":
            drain_risk = "HIGH"
        else:
            drain_risk = np.random.choice(DRAIN_RISKS, p=[0.3, 0.3, 0.3, 0.1])
            
        severity = "CRITICAL" if waste in ["Battery"] else np.random.choice(SEVERITIES, p=[0.2, 0.35, 0.35, 0.1])
        status = np.random.choice(STATUSES, p=[0.25, 0.20, 0.20, 0.35])
        
        t_offset = np.random.uniform(0, 14)
        rep_time = base_time + timedelta(days=t_offset)
        
        reports.append({
            "report_id": f"NGK-{1000 + i}",
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "ward": ward_info["ward"],
            "waste_category": waste,
            "severity": severity,
            "drain_risk": drain_risk,
            "status": status,
            "timestamp": rep_time.strftime("%Y-%m-%d %H:%M"),
            "recurring": ward_info["recurring"]
        })
        
    df_reports = pd.DataFrame(reports)
    reports_csv = CIVIC_DATA_DIR / "demo_reports.csv"
    df_reports.to_csv(reports_csv, index=False)
    
    # Generate Hotspots Summary Table
    hotspots = []
    for idx, w in enumerate(WARDS, 1):
        ward_reps = df_reports[df_reports["ward"] == w["ward"]]
        report_count = len(ward_reps)
        recurring_count = int(report_count * 1.5) if w["recurring"] else report_count
        
        top_waste = ward_reps["waste_category"].mode()[0] if not ward_reps.empty else "Plastic"
        max_severity = "CRITICAL" if "CRITICAL" in ward_reps["severity"].values else ("HIGH" if "HIGH" in ward_reps["severity"].values else "MEDIUM")
        max_drain = "CRITICAL" if "CRITICAL" in ward_reps["drain_risk"].values else ("HIGH" if "HIGH" in ward_reps["drain_risk"].values else "MEDIUM")
        
        score_info = calculate_hotspot_priority_score(
            recurrence_count=recurring_count,
            severity_level=max_severity,
            frequency_reports=report_count,
            drain_risk_level=max_drain,
            exposure_level="HIGH" if w["recurring"] else "MEDIUM"
        )
        
        hotspots.append({
            "hotspot_id": f"HOT-{idx:02d}",
            "ward": w["ward"],
            "latitude": w["lat"],
            "longitude": w["lon"],
            "priority_score": score_info["score"],
            "priority_level": score_info["priority_level"],
            "primary_waste": top_waste,
            "recurrence_count": recurring_count,
            "drain_risk": max_drain,
            "active_reports": report_count,
            "status": "Action Required" if score_info["score"] > 60 else "Monitoring"
        })
        
    df_hotspots = pd.DataFrame(hotspots)
    df_hotspots = df_hotspots.sort_values(by="priority_score", ascending=False)
    hotspots_csv = CIVIC_DATA_DIR / "hotspots.csv"
    df_hotspots.to_csv(hotspots_csv, index=False)
    
    print(f"Generated {len(df_reports)} synthetic reports -> {reports_csv}")
    print(f"Generated {len(df_hotspots)} synthetic hotspots -> {hotspots_csv}")

if __name__ == "__main__":
    generate_civic_data()

