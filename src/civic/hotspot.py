"""
NagrikAI - Civic Hotspot Intelligence & Root-Cause Engine
Calculates explainable multi-factor municipal priority scores, identifies
recurring illegal dumping hotspots, and synthesizes root-cause hypotheses.
"""

from typing import Dict, Any, List
import numpy as np

from ..utils.config import HOTSPOT_WEIGHTS

def calculate_hotspot_priority_score(
    recurrence_count: int,
    severity_level: str,     # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    frequency_reports: int,
    drain_risk_level: str,   # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    exposure_level: str = "HIGH", # "LOW", "MEDIUM", "HIGH"
    weights: Dict[str, float] = HOTSPOT_WEIGHTS
) -> Dict[str, Any]:
    """
    Computes an explainable, deterministic civic priority score (0 to 100).
    Formula incorporates:
    - 30% Recurrence factor (habitual dumping)
    - 25% Severity factor (toxic/bulky/sharp volume)
    - 20% Report frequency (citizen complaints density)
    - 15% Drain blockage risk (flood/drain vulnerability)
    - 10% Public exposure (pedestrian/market footfall)
    """
    severity_map = {"LOW": 25, "MEDIUM": 50, "HIGH": 80, "CRITICAL": 100}
    drain_map = {"LOW": 20, "MEDIUM": 50, "HIGH": 80, "CRITICAL": 100}
    exposure_map = {"LOW": 30, "MEDIUM": 60, "HIGH": 100}
    
    # Normalize recurrence (cap at 10 events)
    norm_recurrence = min(recurrence_count * 10, 100)
    norm_severity = severity_map.get(severity_level.upper(), 50)
    norm_frequency = min(frequency_reports * 12, 100)
    norm_drain = drain_map.get(drain_risk_level.upper(), 50)
    norm_exposure = exposure_map.get(exposure_level.upper(), 60)
    
    score = (
        (norm_recurrence * weights["recurrence"]) +
        (norm_severity * weights["severity"]) +
        (norm_frequency * weights["frequency"]) +
        (norm_drain * weights["drain_risk"]) +
        (norm_exposure * weights["exposure"])
    )
    
    priority_level = "CRITICAL" if score >= 80 else ("HIGH" if score >= 65 else ("MEDIUM" if score >= 45 else "LOW"))
    
    return {
        "score": round(score, 1),
        "priority_level": priority_level,
        "components": {
            "recurrence_component": round(norm_recurrence * weights["recurrence"], 1),
            "severity_component": round(norm_severity * weights["severity"], 1),
            "frequency_component": round(norm_frequency * weights["frequency"], 1),
            "drain_risk_component": round(norm_drain * weights["drain_risk"], 1),
            "exposure_component": round(norm_exposure * weights["exposure"], 1)
        },
        "weights": weights
    }

def generate_root_cause_hypothesis(hotspot_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesizes an explainable, data-driven hypothesis explaining why a specific
    location continues to experience recurring waste accumulation.
    Explicitly marked as: AI-Assisted Hypothesis (Requires Field Verification).
    """
    ward = hotspot_record.get("ward", "Ward Area")
    category = hotspot_record.get("primary_waste", "Mixed Waste")
    drain_risk = hotspot_record.get("drain_risk", "MEDIUM")
    recurrence = hotspot_record.get("recurrence_count", 3)
    
    factors = []
    if category.lower() in ["plastic", "clothes", "trash"]:
        factors.append("Absence of segregated dry-waste collection bins within a 200m radius")
    if drain_risk in ["HIGH", "CRITICAL"]:
        factors.append("Direct proximity to an open roadside storm drain or stormwater culvert")
    if recurrence >= 4:
        factors.append("Habitual nighttime dumping due to poor street illumination and lack of enforcement")
    factors.append("Commercial or marketplace overflow during evening business hours")
    
    hypothesis_text = (
        f"This hotspot in {ward} has accumulated {recurrence} recurring incidents primarily consisting of "
        f"{category}. The high {drain_risk.lower()} drain vulnerability combined with lack of dedicated "
        f"secondary segregation infrastructure encourages habitual dumping. Immediate placement of containerized "
        f"twin-bins and CCTV deterrence recommended."
    )
    
    return {
        "contributing_factors": factors,
        "hypothesis": hypothesis_text,
        "recommended_civic_intervention": "Install twin-bin infrastructure + weekly ward marshal patrol",
        "verification_note": "AI-assisted demo hypothesis. Field validation by Ward Sanitation Inspector required."
    }
