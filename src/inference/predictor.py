"""
NagrikAI - Real-Time Inference & Waste Intelligence Engine
Coordinates local transfer-learning neural inference with civic intelligence mapping,
cluttered scene heuristics, and optional Google Gemini vision enhancement.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image

# Ensure venv site-packages is accessible for torch (APPEND only — never insert at 0)
# Inserting at 0 would let the venv's blocked pandas shadow the working system pandas.
venv_site = Path(__file__).resolve().parent.parent.parent / ".venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.append(str(venv_site))

import torch
import torch.nn.functional as F

from ..utils.config import (
    BEST_MODEL_DIR, MODEL_METADATA_PATH, CLASS_NAMES_PATH,
    CLASSES, IMAGE_SIZE
)
from ..utils.helpers import logger, load_json
from ..data.preprocessing import preprocess_image
from ..civic.risk import get_civic_profile
from ..civic.recommendations import generate_action_plan

class WastePredictor:
    """Production-grade waste inference engine."""
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = Path(model_path) if model_path else (BEST_MODEL_DIR / "model.pth")
        self.classes = CLASSES
        self.metadata = {}
        self.model = None
        self._load_model()
        
    def _load_model(self):
        # 1. Prefer fast, standalone TorchScript model if available
        traced_path = BEST_MODEL_DIR / "model_traced.pt"
        if traced_path.exists():
            try:
                self.model = torch.jit.load(str(traced_path), map_location=self.device)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded high-performance TorchScript model from {traced_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load TorchScript model ({e}); falling back to standard loader.")

        # 2. Fallback to state_dict architecture
        if MODEL_METADATA_PATH.exists():
            try:
                self.metadata = load_json(MODEL_METADATA_PATH)
                arch = self.metadata.get("model_architecture", "mobilenet_v3_small")
            except Exception:
                arch = "mobilenet_v3_small"
        else:
            arch = "mobilenet_v3_small"
            
        logger.info(f"Instantiating classifier architecture: {arch}")
        try:
            from ..training.model_selection import build_classifier_model
            self.model = build_classifier_model(arch, num_classes=len(self.classes), pretrained=True)
            if self.model_path.exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded fine-tuned weights from {self.model_path}")
        except Exception as e:
            logger.warning(f"Fallback architecture load failed: {e}")
            
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference:
        1. Preprocesses image
        2. Computes Softmax model probabilities
        3. Extracts top-k predictions
        4. Detects potential mixed/cluttered waste patterns
        5. Enriches with civic risk and segregation recommendations
        """
        # Preprocess to (1, 3, 224, 224)
        tensor = preprocess_image(image_input).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.classes)))
        
        primary_idx = top_indices[0].item()
        primary_class = self.classes[primary_idx]
        primary_prob = float(top_probs[0].item())
        
        # Calculate entropy / margin between top-1 and top-2 for mixed-waste heuristic
        second_prob = float(top_probs[1].item()) if len(top_probs) > 1 else 0.0
        prob_gap = primary_prob - second_prob
        is_mixed_clutter = (prob_gap < 0.25) or (primary_prob < 0.55)
        
        top_predictions = [
            {
                "class": self.classes[idx.item()].capitalize(),
                "model_probability": round(float(p.item()), 4),
                "percentage": f"{float(p.item()) * 100:.1f}%"
            }
            for idx, p in zip(top_indices, top_probs)
        ]
        
        action_plan = generate_action_plan(
            waste_class=primary_class,
            probability=primary_prob,
            is_cluttered_scene=is_mixed_clutter
        )
        
        return {
            "primary_class": primary_class.capitalize(),
            "model_probability": round(primary_prob, 4),
            "model_probability_pct": f"{primary_prob * 100:.1f}%",
            "is_mixed_clutter": is_mixed_clutter,
            "top_candidates": top_predictions,
            "action_plan": action_plan,
            "civic_profile": action_plan["profile"]
        }

def analyze_with_gemini(
    image: Image.Image,
    local_prediction: Dict[str, Any],
    api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Optional Generative AI reasoning layer using Google Gemini Vision.
    Provides natural-language interpretation, scene composition, and civic reasoning.
    Strictly marked as Generative AI Reasoning (not trained on the Kaggle dataset).
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
        
    try:
        from google import genai
        client = genai.Client(api_key=key)
        
        prompt = f"""
You are NagrikAI's Civic Waste Vision Assistant.
A local computer vision model classified this image as:
Primary Class: {local_prediction['primary_class']} (Model probability: {local_prediction['model_probability_pct']}).

Please analyze the uploaded image and provide an objective civic evaluation in clean JSON format:
{{
  "scene_type": "Single object | Scattered street litter | Overflowing dumpster | Open dump",
  "visible_items": ["list of clearly visible waste items"],
  "drain_hazard_assessment": "Short assessment of whether this threatens open gutters or drains",
  "civic_hygiene_risk": "Low | Moderate | Severe",
  "recommended_municipal_action": "Concise 1-2 sentence recommendation for Indian city municipal ward staff"
}}
Return only valid JSON.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image]
        )
        # Parse JSON from response
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        logger.warning(f"Gemini enhancement query failed: {e}")
        return None
