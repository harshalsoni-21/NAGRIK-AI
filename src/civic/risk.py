"""
NagrikAI - Civic Risk & Segregation Intelligence Engine
Maps raw computer vision classifications to actionable civic waste streams,
environmental risk profiles, drain vulnerability, and municipal routing.
"""

from typing import Dict, Any

# Knowledge base for Indian municipal solid waste rules (SWM Rules 2016 alignment)
CIVIC_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "battery": {
        "civic_category": "Domestic Hazardous Waste / E-Waste",
        "segregation_stream": "Hazardous Waste (Red Bin / Special Collection)",
        "recyclable": True,
        "special_recycler": True,
        "environmental_risk": "CRITICAL",
        "environmental_risk_score": 95,
        "drain_risk": "HIGH",
        "drain_risk_score": 75,
        "hazard_summary": "Heavy metals (lead, cadmium, lithium) leach into ground soil and stormwater runoff, contaminating water tables.",
        "recommended_action": "Do NOT mix with municipal trash. Seal contacts with tape and drop at designated e-waste drop-off center or battery retailer take-back kiosk.",
        "municipal_action": "Route to authorized hazardous waste treatment, storage, and disposal facility (TSDF)."
    },
    "biological": {
        "civic_category": "Biodegradable / Organic Wet Waste",
        "segregation_stream": "Wet Waste (Green Bin)",
        "recyclable": True, # Compostable
        "special_recycler": False,
        "environmental_risk": "MEDIUM",
        "environmental_risk_score": 55,
        "drain_risk": "HIGH",
        "drain_risk_score": 70,
        "hazard_summary": "Rapid anaerobic decomposition generates foul odor, leachate, and attracts disease vectors if left uncollected.",
        "recommended_action": "Deposit directly in green bin for decentralized biomethanation or ward composting within 24 hours.",
        "municipal_action": "Prioritize rapid same-day daily organic waste collection for municipal composting facility."
    },
    "cardboard": {
        "civic_category": "Dry Recyclable Fibre",
        "segregation_stream": "Dry Waste (Blue Bin)",
        "recyclable": True,
        "special_recycler": False,
        "environmental_risk": "LOW",
        "environmental_risk_score": 25,
        "drain_risk": "MEDIUM",
        "drain_risk_score": 60,
        "hazard_summary": "When wet, collapses and mats over storm sewer grates, accelerating blockage during rainfall.",
        "recommended_action": "Flatten to save volume. Keep dry and deposit in dry waste bin or hand over to registered informal waste pickers (Kabadiwala).",
        "municipal_action": "Route to Material Recovery Facility (MRF) for baling and paper mill recycling."
    },
    "clothes": {
        "civic_category": "Textile Waste",
        "segregation_stream": "Dry Waste / Textile Take-back",
        "recyclable": True,
        "special_recycler": True,
        "environmental_risk": "LOW",
        "environmental_risk_score": 35,
        "drain_risk": "CRITICAL",
        "drain_risk_score": 90,
        "hazard_summary": "Synthetic microfibers resist breakdown; large textiles form tough blockages in open drains and culverts.",
        "recommended_action": "If reusable, donate to community cloth drives. Non-reusable cloth should be bagged dry and routed to textile shredding.",
        "municipal_action": "Channel to NGO partnerships or municipal textile upcycling aggregation hubs."
    },
    "glass": {
        "civic_category": "Inert Dry Recyclable / Sharp Hazard",
        "segregation_stream": "Dry Waste (Wrapped / Safe Containment)",
        "recyclable": True,
        "special_recycler": False,
        "environmental_risk": "LOW",
        "environmental_risk_score": 30,
        "drain_risk": "LOW",
        "drain_risk_score": 20,
        "hazard_summary": "Physical laceration hazard to sanitation workers and stray animals. Non-biodegradable but inert.",
        "recommended_action": "Wrap broken shards securely in newspaper or thick cardboard; hand directly to dry waste collector to protect workers.",
        "municipal_action": "Segregate color-wise cullet for glass manufacturing furnaces."
    },
    "metal": {
        "civic_category": "Dry Recyclable / Scrap Metal",
        "segregation_stream": "Dry Waste (Blue Bin)",
        "recyclable": True,
        "special_recycler": False,
        "environmental_risk": "LOW",
        "environmental_risk_score": 20,
        "drain_risk": "MEDIUM",
        "drain_risk_score": 45,
        "hazard_summary": "Valuable secondary resource; sharp edges pose puncture danger to sanitation workers.",
        "recommended_action": "Rinse if food containers. Store dry and exchange via local scrap dealer network.",
        "municipal_action": "High market resale value; route through MRF magnetic separators."
    },
    "paper": {
        "civic_category": "Dry Recyclable Fibre",
        "segregation_stream": "Dry Waste (Blue Bin)",
        "recyclable": True,
        "special_recycler": False,
        "environmental_risk": "LOW",
        "environmental_risk_score": 15,
        "drain_risk": "MEDIUM",
        "drain_risk_score": 50,
        "hazard_summary": "Degrades quickly if uncontaminated; contaminated soiled paper cannot be recycled.",
        "recommended_action": "Keep dry and unsoiled. Separate newsprint and office paper for dry waste collection.",
        "municipal_action": "MRF pulping batch channel."
    },
    "plastic": {
        "civic_category": "Dry Recyclable / Polymer Waste",
        "segregation_stream": "Dry Waste (Blue Bin)",
        "recyclable": True,
        "special_recycler": False,
        "environmental_risk": "HIGH",
        "environmental_risk_score": 85,
        "drain_risk": "CRITICAL",
        "drain_risk_score": 95,
        "hazard_summary": "Severe choking risk for city storm drains; photodegrades into microplastics; non-biodegradable for centuries.",
        "recommended_action": "Rinse single-use containers. Crush bottles. Never burn or dump into open drainage channels.",
        "municipal_action": "Classify polymers (PET, HDPE, MLP). Route MLPs to cement kiln co-processing or road bitumen mixing."
    },
    "shoes": {
        "civic_category": "Composite Footwear Waste",
        "segregation_stream": "Dry Waste / Bulky Collection",
        "recyclable": False,
        "special_recycler": True,
        "environmental_risk": "MEDIUM",
        "environmental_risk_score": 50,
        "drain_risk": "HIGH",
        "drain_risk_score": 80,
        "hazard_summary": "Composite rubber/foam/synthetic materials do not decompose and cause severe hydraulic clogging in drains.",
        "recommended_action": "Separate reusable shoes for donation. Non-wearable shoes should be routed through bulky municipal pickup.",
        "municipal_action": "Separate rubber soles for pyrolysis/rubber reclaim; non-recyclable composites to refuse-derived fuel (RDF)."
    },
    "trash": {
        "civic_category": "Mixed Non-Recyclable / Inert Residue",
        "segregation_stream": "Residual Solid Waste (Black / Red Bin)",
        "recyclable": False,
        "special_recycler": False,
        "environmental_risk": "HIGH",
        "environmental_risk_score": 75,
        "drain_risk": "HIGH",
        "drain_risk_score": 85,
        "hazard_summary": "Mixed unsegregated refuse produces toxic leachate, greenhouse gases, and obstructs civic sanitation systems.",
        "recommended_action": "Attempt segregation before disposal. Deposit residue in designated neighborhood dhalao or containerized compactor.",
        "municipal_action": "Mechanized secondary segregation at transfer station; non-recyclable combustible fraction to waste-to-energy plant."
    }
}

def get_civic_profile(waste_class: str) -> Dict[str, Any]:
    """Retrieve complete civic intelligence profile for a classified waste class."""
    norm_class = str(waste_class).lower().strip()
    profile = CIVIC_KNOWLEDGE_BASE.get(norm_class)
    if not profile:
        return {
            "civic_category": "Unclassified Mixed Waste",
            "segregation_stream": "Residual Waste",
            "recyclable": False,
            "special_recycler": False,
            "environmental_risk": "MEDIUM",
            "environmental_risk_score": 50,
            "drain_risk": "MEDIUM",
            "drain_risk_score": 50,
            "hazard_summary": "Unidentified waste matter requiring manual sanitation inspection.",
            "recommended_action": "Inspect for hazardous substances and segregate dry recyclables from wet waste.",
            "municipal_action": "Dispatch ward sanitation team for on-site assessment."
        }
    return profile

