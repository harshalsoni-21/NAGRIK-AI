"""
NagrikAI - Comprehensive Exploratory Data Analysis (EDA) Engine
Generates publication-quality charts and plots matching the cream-and-brown palette
under reports/eda/.
"""

import os
import random
from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ..utils.config import RAW_DATA_DIR, EDA_REPORTS_DIR, CLASSES, PALETTE
from ..utils.helpers import logger
from ..data.augmentation import augment_pil_image

# Set Matplotlib style to warm cream & deep brown
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = PALETTE["cream_bg"]
plt.rcParams["axes.facecolor"] = PALETTE["card_warm"]
plt.rcParams["axes.edgecolor"] = PALETTE["border_muted"]
plt.rcParams["axes.labelcolor"] = PALETTE["deep_brown"]
plt.rcParams["xtick.color"] = PALETTE["deep_brown"]
plt.rcParams["ytick.color"] = PALETTE["deep_brown"]
plt.rcParams["text.color"] = PALETTE["deep_brown"]

def run_eda(dataset_dir: Path | str = RAW_DATA_DIR, output_dir: Path | str = EDA_REPORTS_DIR):
    """Executes full exploratory data analysis and exports visualizations."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Executing Exploratory Data Analysis...")
    
    # 1. Collect statistics
    class_counts = {}
    sample_images = {}
    dimensions = []
    
    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and not f.startswith(".")]
        class_counts[cls] = len(files)
        if files:
            sample_images[cls] = cls_dir / random.choice(files)
            for sample_f in files[:50]:
                try:
                    with Image.open(cls_dir / sample_f) as img:
                        dimensions.append(img.size)
                except Exception:
                    pass

    # -------------------------------------------------------------
    # Chart 1: Class Distribution Bar Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    names = [k.capitalize() for k, _ in sorted_items]
    counts = [v for _, v in sorted_items]
    
    colors = [PALETTE["earth_accent"] if i < 3 else PALETTE["secondary_brown"] for i in range(len(names))]
    bars = plt.bar(names, counts, color=colors, edgecolor=PALETTE["deep_brown"], linewidth=1.2)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=PALETTE["deep_brown"])
        
    plt.title("NagrikAI Dataset Class Distribution (Total: 19,762 Images)", fontsize=14, fontweight="bold", pad=15, color=PALETTE["deep_brown"])
    plt.xlabel("Waste Category", fontsize=12, labelpad=10)
    plt.ylabel("Number of Annotated Samples", fontsize=12, labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5, color=PALETTE["border_muted"])
    plt.tight_layout()
    chart1_path = output_dir / "class_distribution.png"
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {chart1_path}")

    # -------------------------------------------------------------
    # Chart 2: Sample Gallery (10 Classes)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for idx, cls in enumerate(CLASSES):
        ax = axes[idx]
        if cls in sample_images:
            img = Image.open(sample_images[cls])
            ax.imshow(img)
        ax.set_title(f"{cls.capitalize()}\n({class_counts.get(cls, 0):,} imgs)", fontsize=11, fontweight="bold", color=PALETTE["deep_brown"])
        ax.axis("off")
    plt.suptitle("NagrikAI Representative Waste Class Samples", fontsize=15, fontweight="bold", y=1.02, color=PALETTE["deep_brown"])
    plt.tight_layout()
    chart2_path = output_dir / "sample_gallery.png"
    plt.savefig(chart2_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {chart2_path}")

    # -------------------------------------------------------------
    # Chart 3: Image Dimensions Analysis
    # -------------------------------------------------------------
    if dimensions:
        widths, heights = zip(*dimensions)
        plt.figure(figsize=(10, 5))
        plt.scatter(widths, heights, color=PALETTE["earth_accent"], alpha=0.6, edgecolors=PALETTE["deep_brown"], s=40)
        plt.axvline(224, color=PALETTE["red_risk"], linestyle="--", label="Model Input 224x224")
        plt.axhline(224, color=PALETTE["red_risk"], linestyle="--")
        plt.title("Resolution Distribution of Sampled Images", fontsize=13, fontweight="bold", color=PALETTE["deep_brown"])
        plt.xlabel("Width (pixels)", fontsize=11)
        plt.ylabel("Height (pixels)", fontsize=11)
        plt.legend(frameon=True, facecolor=PALETTE["card_warm"], edgecolor=PALETTE["border_muted"])
        plt.grid(True, linestyle="--", alpha=0.4, color=PALETTE["border_muted"])
        plt.tight_layout()
        chart3_path = output_dir / "image_dimensions.png"
        plt.savefig(chart3_path, dpi=200)
        plt.close()
        logger.info(f"Saved: {chart3_path}")

    # -------------------------------------------------------------
    # Chart 4: Data Augmentation Comparison (Original vs Augmented)
    # -------------------------------------------------------------
    sample_cls = "plastic" if "plastic" in sample_images else list(sample_images.keys())[0]
    raw_img = Image.open(sample_images[sample_cls]).convert("RGB").resize((224, 224))
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(raw_img)
    axes[0].set_title("Original Image", fontweight="bold", color=PALETTE["deep_brown"])
    axes[0].axis("off")
    
    for i in range(1, 4):
        aug_img = augment_pil_image(raw_img, seed=i*10)
        axes[i].imshow(aug_img)
        axes[i].set_title(f"Augmented #{i}", fontweight="bold", color=PALETTE["deep_brown"])
        axes[i].axis("off")
        
    plt.suptitle(f"Realistic Semantic-Preserving Augmentations ({sample_cls.capitalize()})", fontsize=14, fontweight="bold", y=1.02, color=PALETTE["deep_brown"])
    plt.tight_layout()
    chart4_path = output_dir / "augmentation_comparison.png"
    plt.savefig(chart4_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {chart4_path}")

    logger.info("EDA visual artifacts generated successfully.")

if __name__ == "__main__":
    run_eda()

