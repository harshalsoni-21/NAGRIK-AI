"""
NagrikAI - Civic Action Recommendations Module
Generates actionable, role-based recommendations for Citizens, Waste Pickers/Sanitation Staff,
and Municipal Urban Local Bodies (ULBs).
"""

from typing import Dict, Any
from .risk import get_civic_profile

def generate_action_plan(
    waste_class: str,
    probability: float,
    is_cluttered_scene: bool = False
) -> Dict[str, Any]:
    """
    Generate an integrated multi-stakeholder action plan.
    Includes honest note on cluttered/mixed waste scenes.
    """
    profile = get_civic_profile(waste_class)
    
    # Mixed / cluttered scene transparency
    mixed_note = None
    if is_cluttered_scene or probability < 0.65 or waste_class.lower() == "trash":
        mixed_note = (
            "Mixed / cluttered waste pattern detected. Single-label classifier identifies the dominant "
            "feature, but multi-stream elements may coexist. Secondary manual segregation advised."
        )
        
    citizen_guide = {
        "bin_color": "Blue Bin (Dry)" if "Dry" in profile["segregation_stream"] else (
            "Green Bin (Wet)" if "Wet" in profile["segregation_stream"] else "Red / Domestic Hazardous Box"
        ),
        "primary_action": profile["recommended_action"],
        "safety_tip": "Handle with gloves if sharp or hazardous." if profile["environmental_risk"] in ["HIGH", "CRITICAL"] else "Ensure waste is free of moisture."
    }
    
    sanitation_guide = {
        "ppe_required": "Heavy puncture-resistant gloves and safety boots" if waste_class.lower() in ["glass", "battery", "metal"] else "Standard sanitization apron and nitrile gloves",
        "handling_priority": "Immediate containment" if profile["drain_risk"] in ["HIGH", "CRITICAL"] else "Regular scheduled clearance",
        "route_channel": profile["municipal_action"]
    }
    
    ulb_intelligence = {
        "sla_target_hours": 12 if profile["drain_risk"] == "CRITICAL" else (24 if profile["environmental_risk"] == "HIGH" else 48),
        "recycling_potential": "High Value Resource" if profile["recyclable"] else "Non-Recyclable Residue / RDF Destination",
        "monsoon_vulnerability": "High vulnerability to urban waterlogging" if profile["drain_risk"] in ["HIGH", "CRITICAL"] else "Standard drainage profile"
    }
    
    return {
        "class": waste_class.capitalize(),
        "probability": float(probability),
        "profile": profile,
        "mixed_scene_note": mixed_note,
        "citizen": citizen_guide,
        "sanitation_workers": sanitation_guide,
        "municipal_authority": ulb_intelligence
    }

