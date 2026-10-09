"""
NagrikAI - Model Evaluation Engine
Generates comprehensive test metrics, confusion matrix heatmaps,
per-class performance breakdowns, and misclassification error audits.
"""

from pathlib import Path
from typing import Dict, Any, List, Union
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

from ..utils.config import EVAL_REPORTS_DIR, CLASSES, PALETTE
from ..utils.helpers import logger, save_json

plt.rcParams["figure.facecolor"] = PALETTE["cream_bg"]
plt.rcParams["axes.facecolor"] = PALETTE["card_warm"]
plt.rcParams["axes.edgecolor"] = PALETTE["border_muted"]
plt.rcParams["axes.labelcolor"] = PALETTE["deep_brown"]

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Union[Path, str] = EVAL_REPORTS_DIR
) -> Dict[str, Any]:
    """
    Evaluates model on untouched test dataset.
    Exports confusion matrix, per-class F1, and complete evaluation report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 1. Compute core metrics
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    
    class_names = [c.capitalize() for c in CLASSES]
    clf_report_dict = classification_report(
        all_targets, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    clf_report_text = classification_report(
        all_targets, all_preds,
        target_names=class_names,
        zero_division=0
    )
    
    per_class_f1 = {cls_name: float(clf_report_dict[cls_name]["f1-score"]) for cls_name in class_names if cls_name in clf_report_dict}
    sorted_classes = sorted(per_class_f1.items(), key=lambda x: x[1], reverse=True)
    strongest_classes = sorted_classes[:3]
    weakest_classes = sorted_classes[-3:]
    
    # Save text report
    report_text_path = output_dir / "classification_report.txt"
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("                 NAGRIKAI MODEL EVALUATION REPORT\n")
        f.write("======================================================================\n\n")
        f.write(f"Test Accuracy:    {acc * 100:.2f}%\n")
        f.write(f"Macro F1-Score:   {macro_f1 * 100:.2f}%\n")
        f.write(f"Weighted F1-Score:{weighted_f1 * 100:.2f}%\n")
        f.write(f"Macro Precision:  {macro_prec * 100:.2f}%\n")
        f.write(f"Macro Recall:     {macro_rec * 100:.2f}%\n\n")
        f.write("PER-CLASS CLASSIFICATION BREAKDOWN:\n")
        f.write("----------------------------------------------------------------------\n")
        f.write(clf_report_text)
        f.write("\n\nSTRONGEST CLASSES:\n")
        for c, s in strongest_classes:
            f.write(f"  + {c}: F1 = {s * 100:.1f}%\n")
        f.write("\nWEAKEST CLASSES (Common Misclassification Targets):\n")
        for c, s in weakest_classes:
            f.write(f"  - {c}: F1 = {s * 100:.1f}%\n")
            
    # 2. Confusion Matrix Plot (Pure Matplotlib)
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.YlOrBr)
    fig.colorbar(cax)
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", color=PALETTE["deep_brown"])
    ax.set_yticklabels(class_names, color=PALETTE["deep_brown"])
    
    # Text annotations inside confusion matrix cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, format(val, 'd'),
                    ha="center", va="center",
                    color="white" if val > thresh else PALETTE["deep_brown"],
                    fontweight="bold")
                    
    ax.set_title(f"NagrikAI Confusion Matrix (Test Accuracy: {acc*100:.1f}%)", fontsize=14, fontweight="bold", pad=12, color=PALETTE["deep_brown"])
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8, color=PALETTE["deep_brown"])
    ax.set_ylabel("Ground Truth Label", fontsize=12, labelpad=8, color=PALETTE["deep_brown"])
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    
    # 3. Per-Class F1 Chart
    plt.figure(figsize=(11, 5))
    cls_names_plot = [x[0] for x in sorted_classes]
    f1_values_plot = [x[1] * 100 for x in sorted_classes]
    colors = [PALETTE["green_eco"] if v >= 80 else (PALETTE["earth_accent"] if v >= 65 else PALETTE["red_risk"]) for v in f1_values_plot]
    
    bars = plt.barh(cls_names_plot[::-1], f1_values_plot[::-1], color=colors[::-1], edgecolor=PALETTE["deep_brown"], linewidth=1.0)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold", color=PALETTE["deep_brown"])
        
    plt.xlim(0, 105)
    plt.title("Per-Class F1 Performance on Unseen Test Split", fontsize=13, fontweight="bold", pad=12, color=PALETTE["deep_brown"])
    plt.xlabel("F1-Score (%)", fontsize=11, color=PALETTE["deep_brown"])
    plt.grid(axis="x", linestyle="--", alpha=0.4, color=PALETTE["border_muted"])
    plt.tight_layout()
    f1_chart_path = output_dir / "per_class_f1.png"
    plt.savefig(f1_chart_path, dpi=200)
    plt.close()
    
    metrics_summary = {
        "test_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class_f1": per_class_f1,
        "strongest_classes": dict(strongest_classes),
        "weakest_classes": dict(weakest_classes),
        "confusion_matrix_path": str(cm_path),
        "classification_report_path": str(report_text_path)
    }
    
    save_json(metrics_summary, output_dir / "eval_metrics.json")
    logger.info(f"Evaluation complete. Test Accuracy: {acc*100:.2f}%, Macro F1: {macro_f1*100:.2f}%")
    return metrics_summary

