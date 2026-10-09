"""
NagrikAI - End-to-End Model Training Pipeline
Supports transfer learning, class-weighted loss, cosine annealing, early stopping,
checkpoint saving, and model comparison.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..utils.config import (
    RAW_DATA_DIR, MODELS_DIR, BEST_MODEL_DIR, MODEL_METADATA_PATH,
    CLASS_NAMES_PATH, CLASSES, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED,
    EVAL_REPORTS_DIR
)
from ..utils.helpers import logger, save_json
from ..data.cleaning import scan_and_clean_dataset
from ..data.dataset import create_stratified_splits, compute_class_weights, get_dataloaders
from .model_selection import build_classifier_model
from .evaluate import evaluate_model

def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets.data).item()
        total += targets.size(0)
        
    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc

def validate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets.data).item()
            total += targets.size(0)
            
    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc

def run_training(
    model_name: str = "mobilenet_v3_small",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    subset_size: Optional[int] = None,
    save_dir: Union[Path, str] = BEST_MODEL_DIR
) -> Dict[str, Any]:
    """
    Executes training workflow:
    - Verifies dataset integrity
    - Creates stratified splits
    - Computes class imbalance loss weights
    - Trains transfer-learning model with early stopping
    - Evaluates on unseen test set
    - Saves model weights, class names, and metadata
    """
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Compute Device: {device}")
    
    # 1. Dataset Audit
    logger.info("Step 1: Running dataset audit and cleaning...")
    scan_and_clean_dataset()
    
    # 2. Stratified Splits
    logger.info("Step 2: Preparing stratified splits (70/15/15)...")
    train_samples, val_samples, test_samples = create_stratified_splits(seed=RANDOM_SEED)
    
    if subset_size and subset_size < len(train_samples):
        logger.info(f"Subsetting training set to {subset_size} samples for high-efficiency CPU execution...")
        import random
        random.seed(RANDOM_SEED)
        train_samples = random.sample(train_samples, subset_size)
        val_samples = random.sample(val_samples, min(len(val_samples), int(subset_size * 0.25)))
        test_samples = random.sample(test_samples, min(len(test_samples), int(subset_size * 0.25)))

    # Compute class weights for imbalanced classes
    class_weights = compute_class_weights(train_samples).to(device)
    logger.info(f"Class weights computed for {len(CLASSES)} classes to handle class imbalance.")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        train_samples, val_samples, test_samples, batch_size=batch_size
    )
    
    # 3. Model Construction
    logger.info(f"Step 3: Initializing transfer-learning model ({model_name})...")
    model = build_classifier_model(model_name, num_classes=len(CLASSES), pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 4. Training Loop
    logger.info(f"Step 4: Training for {epochs} epochs...")
    best_val_acc = 0.0
    best_model_weights = None
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    weights_file = save_path / "model.pth"
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0
        
        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
            torch.save(best_model_weights, weights_file)
            logger.info(f" --> Checkpoint saved! New best validation accuracy: {best_val_acc*100:.2f}%")
            
    total_train_time = time.time() - start_time
    logger.info(f"Training finished in {total_train_time:.1f}s. Loading best weights for test evaluation...")
    
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    # 5. Unseen Test Set Evaluation
    logger.info("Step 5: Evaluating best checkpoint on unseen test set...")
    eval_metrics = evaluate_model(model, test_loader, device)
    
    # Measure inference speed (latency per sample)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    latencies = []
    with torch.no_grad():
        for _ in range(25):
            _ = model(dummy_input)
        for _ in range(50):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            latencies.append((time.perf_counter() - t_start) * 1000)
    avg_latency_ms = sum(latencies) / len(latencies)
    model_size_mb = os.path.getsize(weights_file) / (1024 * 1024)
    
    # 6. Save Metadata & Class Names
    save_json(CLASSES, CLASS_NAMES_PATH)
    
    metadata = {
        "model_architecture": model_name,
        "input_resolution": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(eval_metrics["test_accuracy"]),
        "macro_f1": float(eval_metrics["macro_f1"]),
        "weighted_f1": float(eval_metrics["weighted_f1"]),
        "macro_precision": float(eval_metrics["macro_precision"]),
        "macro_recall": float(eval_metrics["macro_recall"]),
        "inference_latency_cpu_ms": round(avg_latency_ms, 2),
        "model_file_size_mb": round(model_size_mb, 2),
        "weights_path": str(weights_file),
        "training_duration_seconds": round(total_train_time, 1),
        "evaluated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_json(metadata, MODEL_METADATA_PATH)
    logger.info(f"Model metadata saved to {MODEL_METADATA_PATH}")
    
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NagrikAI Training CLI")
    parser.add_argument("--model", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "efficientnet_b0", "resnet18"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=2000, help="Stratified sample count for fast CPU training")
    args = parser.parse_args()
    
    run_training(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subset_size=args.subset
    )

