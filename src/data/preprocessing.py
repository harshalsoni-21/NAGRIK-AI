"""
NagrikAI - Preprocessing & Pipeline Module
Ensures 100% parity between training pipeline and live Streamlit/API inference.
"""

from typing import Union, Tuple, Dict
from pathlib import Path
from PIL import Image
import numpy as np
import sys

# Ensure torch can be found
venv_site = Path(__file__).resolve().parent.parent.parent / ".venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.append(str(venv_site))

import torch

from ..utils.config import CLASSES, IMAGE_SIZE, NORM_MEAN, NORM_STD

def get_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Generates deterministic class-to-index and index-to-class mappings."""
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
    idx_to_class = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}
    return class_to_idx, idx_to_class

def preprocess_image(image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
    """
    Standardizes any input image (file path, PIL Image) into a normalized tensor:
    Shape: (1, 3, 224, 224)
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
    # Ensure RGB (strip alpha / grayscale)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Resize to 224x224
    img_resized = img.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
    
    # Convert to float numpy array (H, W, C) in [0, 1]
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Standard ImageNet normalization: (x - mean) / std
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std = np.array(NORM_STD, dtype=np.float32)
    arr = (arr - mean) / std
    
    # Transpose to (C, H, W)
    arr = np.transpose(arr, (2, 0, 1))
    
    # Convert to PyTorch Tensor with batch dimension -> (1, 3, 224, 224)
    tensor = torch.from_numpy(arr).float().unsqueeze(0)
    return tensor

