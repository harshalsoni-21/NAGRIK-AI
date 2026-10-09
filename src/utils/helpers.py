"""
NagrikAI - Utility Helpers
Provides JSON serialization, image validation, logging, and color-coded CLI outputs.
"""

import os
import json
import logging
from pathlib import Path
from typing import Union
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NagrikAI")

def load_json(filepath: Union[Path, str]) -> Union[dict, list]:
    """Safely load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Union[dict, list], filepath: Union[Path, str], indent: int = 2) -> None:
    """Save data to JSON file with indentation."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)

def is_valid_image_file(filepath: Union[Path, str]) -> bool:
    """Check if file exists and can be opened as an image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def format_percentage(val: float) -> str:
    """Format decimal 0-1 to percentage string."""
    return f"{val * 100:.1f}%"

