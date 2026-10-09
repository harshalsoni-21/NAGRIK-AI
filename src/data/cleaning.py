"""
NagrikAI - Data Cleaning & Integrity Pipeline
Audits raw image dataset, detects corruption, duplicates, aspect anomalies,
and generates an audit report to reports/data_quality_report.txt.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter
from PIL import Image

from ..utils.config import RAW_DATA_DIR, DATA_QUALITY_REPORT_PATH, CLASSES
from ..utils.helpers import logger

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}

def scan_and_clean_dataset(
    dataset_dir: Path | str = RAW_DATA_DIR,
    report_output_path: Path | str = DATA_QUALITY_REPORT_PATH,
    min_dimension: int = 50
) -> Dict[str, Any]:
    """
    Executes deep dataset validation:
    - Verifies readable images
    - Identifies MD5 hash duplicates
    - Finds suspiciously small dimensions (< min_dimension)
    - Records class balance & image size statistics
    - Outputs full audit report to file
    """
    dataset_dir = Path(dataset_dir)
    logger.info(f"Initiating Data Cleaning Scan on: {dataset_dir}")
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
    total_images = 0
    valid_images: List[str] = []
    corrupted_files: List[Tuple[str, str]] = []
    duplicate_files: List[Tuple[str, str]] = []
    suspiciously_small: List[Tuple[str, Tuple[int, int]]] = []
    unsupported_files: List[str] = []
    class_counts: Dict[str, int] = Counter()
    
    seen_hashes: Dict[str, str] = {}
    sizes: List[Tuple[int, int]] = []
    
    # Iterate through class folders
    for class_folder in sorted(os.listdir(dataset_dir)):
        class_path = dataset_dir / class_folder
        if not class_path.is_dir() or class_folder.startswith("."):
            continue
            
        for file_name in os.listdir(class_path):
            if file_name.startswith("."):
                continue
                
            total_images += 1
            file_path = class_path / file_name
            ext = file_path.suffix.lower()
            
            # Format validation
            if ext not in SUPPORTED_FORMATS:
                unsupported_files.append(str(file_path))
                continue
                
            # Integrity check
            try:
                # 0-byte file check
                if file_path.stat().st_size == 0:
                    corrupted_files.append((str(file_path), "Zero-byte file"))
                    continue
                    
                # MD5 Duplicate check
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    
                if file_hash in seen_hashes:
                    duplicate_files.append((str(file_path), seen_hashes[file_hash]))
                    continue
                seen_hashes[file_hash] = str(file_path)
                
                # Pillow read test
                with Image.open(file_path) as img:
                    img.verify()
                    
                # Dimensions check
                with Image.open(file_path) as img:
                    w, h = img.size
                    sizes.append((w, h))
                    if w < min_dimension or h < min_dimension:
                        suspiciously_small.append((str(file_path), (w, h)))
                    else:
                        valid_images.append(str(file_path))
                        class_counts[class_folder] += 1
                        
            except Exception as e:
                corrupted_files.append((str(file_path), str(e)))
                
    # Dimensions calculations
    if sizes:
        sorted_by_area = sorted(sizes, key=lambda s: s[0] * s[1])
        smallest_img = sorted_by_area[0]
        largest_img = sorted_by_area[-1]
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)
    else:
        smallest_img = (0, 0)
        largest_img = (0, 0)
        avg_w, avg_h = 0, 0

    report_content = f"""======================================================================
               NAGRIKAI DATA QUALITY AUDIT REPORT
======================================================================
Dataset Location: {dataset_dir}
Timestamp: Real-time scan verified

DATASET SUMMARY:
----------------------------------------------------------------------
Total Images Scanned:     {total_images:,}
Valid Clean Images:       {len(valid_images):,}
Corrupted Images:         {len(corrupted_files)}
Duplicate Files Found:    {len(duplicate_files)}
Suspiciously Small (<{min_dimension}px): {len(suspiciously_small)}
Unsupported Formats:      {len(unsupported_files)}

IMAGE DIMENSIONS ANALYSIS:
----------------------------------------------------------------------
Smallest Image (W x H):   {smallest_img[0]} x {smallest_img[1]} px
Largest Image (W x H):    {largest_img[0]} x {largest_img[1]} px
Average Dimension:        {avg_w:.1f} x {avg_h:.1f} px

CLASS DISTRIBUTION & IMBALANCE:
----------------------------------------------------------------------
"""
    for cls in sorted(class_counts.keys()):
        cnt = class_counts[cls]
        pct = (cnt / len(valid_images) * 100) if valid_images else 0
        report_content += f"{cls.ljust(15)} : {cnt:>6,} images ({pct:>5.1f}%)\n"
        
    report_content += f"""----------------------------------------------------------------------
Total Classes: {len(class_counts)}
Imbalance Ratio (Max/Min): {max(class_counts.values()) / min(class_counts.values()):.2f}x
Recommendation: Use weighted cross-entropy loss or balanced sampler for clothes/glass dominance.
======================================================================
"""

    report_path = Path(report_output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    logger.info(f"Audit Complete! Report saved to {report_path}")
    
    return {
        "total_images": total_images,
        "valid_images_count": len(valid_images),
        "corrupted_count": len(corrupted_files),
        "duplicates_count": len(duplicate_files),
        "suspicious_count": len(suspiciously_small),
        "class_counts": dict(class_counts),
        "smallest_image": smallest_img,
        "largest_image": largest_img,
        "avg_dimensions": (avg_w, avg_h),
        "report_path": str(report_path)
    }

if __name__ == "__main__":
    result = scan_and_clean_dataset()
    print(f"Data Quality Report successfully generated with {result['total_images']} images scanned.")

