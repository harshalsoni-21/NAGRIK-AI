"""
NagrikAI - Data Augmentation Module
Applies realistic, waste-preserving visual transformations to simulate
real-world civic camera conditions (varying daylight, angles, shadows, orientation)
without distorting waste semantics. Supports both Torchvision and Pure PIL/NumPy pipelines.
"""

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from ..utils.config import IMAGE_SIZE, NORM_MEAN, NORM_STD

def augment_pil_image(img: Image.Image, seed: int = None) -> Image.Image:
    """
    Applies realistic PIL-based semantic augmentation:
    - Random Horizontal Flip
    - Subtle rotation (-15 to +15 deg)
    - Brightness / contrast jitter
    """
    if seed is not None:
        np.random.seed(seed)
        
    out = img.copy()
    
    # 50% chance horizontal flip
    if np.random.rand() > 0.5:
        out = ImageOps.mirror(out)
        
    # Subtle rotation
    angle = float(np.random.uniform(-15, 15))
    out = out.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
    
    # Brightness jitter (0.85 to 1.15)
    b_factor = float(np.random.uniform(0.85, 1.15))
    enhancer = ImageEnhance.Brightness(out)
    out = enhancer.enhance(b_factor)
    
    # Contrast jitter (0.85 to 1.15)
    c_factor = float(np.random.uniform(0.85, 1.15))
    enhancer = ImageEnhance.Contrast(out)
    out = enhancer.enhance(c_factor)
    
    return out

def get_train_transforms():
    """Returns PyTorch torchvision compose transforms if available, otherwise fallback."""
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
    except Exception:
        return lambda img: augment_pil_image(img)

def get_eval_transforms():
    """Returns PyTorch eval transforms."""
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
    except Exception:
        return lambda img: img.resize(IMAGE_SIZE)

