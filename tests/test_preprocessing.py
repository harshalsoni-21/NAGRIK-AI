"""
NagrikAI - Unit Tests: Preprocessing Pipeline
"""

import unittest
from PIL import Image
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

venv_site = BASE_DIR / ".venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.append(str(venv_site))

from src.data.preprocessing import preprocess_image, get_label_mappings
from src.utils.config import CLASSES, IMAGE_SIZE

class TestPreprocessing(unittest.TestCase):
    
    def test_label_mappings(self):
        class_to_idx, idx_to_class = get_label_mappings()
        self.assertEqual(len(class_to_idx), len(CLASSES))
        self.assertEqual(len(idx_to_class), len(CLASSES))
        for c in CLASSES:
            self.assertIn(c, class_to_idx)
            self.assertEqual(idx_to_class[class_to_idx[c]], c)
            
    def test_tensor_shape_and_dtype(self):
        dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        tensor = preprocess_image(dummy_img)
        self.assertEqual(tensor.shape, (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        
    def test_grayscale_conversion(self):
        dummy_gray = Image.fromarray(np.random.randint(0, 255, (200, 200), dtype=np.uint8), mode="L")
        tensor = preprocess_image(dummy_gray)
        self.assertEqual(tensor.shape, (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

if __name__ == "__main__":
    unittest.main()

