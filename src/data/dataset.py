"""
NagrikAI - PyTorch Dataset, Stratified Splitting & Class Imbalance Handler
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

from ..utils.config import RAW_DATA_DIR, CLASSES, BATCH_SIZE, RANDOM_SEED
from ..utils.helpers import logger
from .augmentation import get_train_transforms, get_eval_transforms
from .preprocessing import get_label_mappings

class GarbageDataset(Dataset):
    """Clean PyTorch Dataset for Garbage Classification."""
    
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)
        return img_tensor, label

def load_all_samples(dataset_dir: Path | str = RAW_DATA_DIR) -> Tuple[List[str], List[int]]:
    """Scan dataset directory and return valid image paths and integer labels."""
    dataset_dir = Path(dataset_dir)
    class_to_idx, _ = get_label_mappings()
    
    image_paths: List[str] = []
    labels: List[int] = []
    
    for cls_name in CLASSES:
        cls_dir = dataset_dir / cls_name
        if not cls_dir.is_dir():
            continue
        idx = class_to_idx[cls_name]
        for f in os.listdir(cls_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and not f.startswith("."):
                image_paths.append(str(cls_dir / f))
                labels.append(idx)
                
    return image_paths, labels

def create_stratified_splits(
    dataset_dir: Path | str = RAW_DATA_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Creates stratified 70% / 15% / 15% train, validation, and test splits
    guaranteeing no data leakage across sets.
    """
    image_paths, labels = load_all_samples(dataset_dir)
    
    # First split off train
    train_paths, rem_paths, train_lbls, rem_lbls = train_test_split(
        image_paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )
    
    # Then split val and test equally (50% each of the remaining 30%)
    val_paths, test_paths, val_lbls, test_lbls = train_test_split(
        rem_paths, rem_lbls,
        test_size=0.5,
        stratify=rem_lbls,
        random_state=seed
    )
    
    train_samples = list(zip(train_paths, train_lbls))
    val_samples = list(zip(val_paths, val_lbls))
    test_samples = list(zip(test_paths, test_lbls))
    
    logger.info(
        f"Stratified Splits Created: Train={len(train_samples):,} (70%), "
        f"Val={len(val_samples):,} (15%), Test={len(test_samples):,} (15%)"
    )
    
    return train_samples, val_samples, test_samples

def compute_class_weights(train_samples: List[Tuple[str, int]], num_classes: int = len(CLASSES)) -> torch.Tensor:
    """
    Computes inverse-frequency balanced class weights:
    weight[c] = total_samples / (num_classes * count[c])
    Effectively prevents dominant classes (clothes/glass) from drowning minority ones (battery/trash).
    """
    counts = Counter([s[1] for s in train_samples])
    total = len(train_samples)
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)

def get_dataloaders(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    test_samples: List[Tuple[str, int]],
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns PyTorch DataLoaders with appropriate transforms."""
    train_ds = GarbageDataset(train_samples, transform=get_train_transforms())
    val_ds = GarbageDataset(val_samples, transform=get_eval_transforms())
    test_ds = GarbageDataset(test_samples, transform=get_eval_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

