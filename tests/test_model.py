"""
NagrikAI - Unit Tests: Model Architecture & Predictor Pipeline
"""

import unittest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

venv_site = BASE_DIR / ".venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.append(str(venv_site))

import torch
from PIL import Image
import numpy as np

from src.inference.predictor import WastePredictor
from src.utils.config import CLASSES

class TestModelPipeline(unittest.TestCase):
    
    def test_model_architecture_output_dim(self):
        predictor = WastePredictor()
        dummy_tensor = torch.randn(2, 3, 224, 224)
        out = predictor.model(dummy_tensor)
        self.assertEqual(out.shape, (2, len(CLASSES)))
        
    def test_predictor_inference(self):
        predictor = WastePredictor()
        # Test on dummy generated RGB image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = predictor.predict(dummy_img)
        
        self.assertIn("primary_class", result)
        self.assertIn("model_probability", result)
        self.assertIn("action_plan", result)
        self.assertIn(result["primary_class"].lower(), [c.lower() for c in CLASSES])
        self.assertGreaterEqual(result["model_probability"], 0.0)
        self.assertLessEqual(result["model_probability"], 1.0)

if __name__ == "__main__":
    unittest.main()

