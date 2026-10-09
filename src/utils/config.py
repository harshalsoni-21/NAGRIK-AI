"""
NagrikAI - Central Configuration Module
Defines system paths, color schemes, ML parameters, and civic intelligence mappings.
"""

from pathlib import Path
import os

# Root & workspace paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "garbage-dataset" if (BASE_DIR / "garbage-dataset").exists() else BASE_DIR / "data" / "raw" / "garbage_dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CIVIC_DATA_DIR = DATA_DIR / "civic"
DEMO_DATA_DIR = DATA_DIR / "demo"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_DIR = MODELS_DIR / "best_model"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

REPORTS_DIR = BASE_DIR / "reports"
EDA_REPORTS_DIR = REPORTS_DIR / "eda"
EVAL_REPORTS_DIR = REPORTS_DIR / "evaluation"
DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.txt"

ASSETS_DIR = BASE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"

# Standard classes verified directly from Kaggle garbage-dataset
CLASSES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]

# Image Preprocessing & Model Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Theme & Palette: Classic Cream + Warm White + Deep Brown + Muted Earth
PALETTE = {
    "cream_bg": "#FBF8F3",
    "card_bg": "#FFFFFF",
    "card_warm": "#F6F1EA",
    "deep_brown": "#2C1E16",
    "secondary_brown": "#594236",
    "earth_accent": "#A3704C",
    "earth_light": "#E9DFD4",
    "border_muted": "#E0D7CC",
    "green_eco": "#2E7D32",
    "green_light": "#E8F5E9",
    "amber_warn": "#E65100",
    "amber_light": "#FFF3E0",
    "red_risk": "#C62828",
    "red_light": "#FFEBEE",
}

# Hotspot Priority Weights (Transparent & Explainable)
HOTSPOT_WEIGHTS = {
    "recurrence": 0.30,
    "severity": 0.25,
    "frequency": 0.20,
    "drain_risk": 0.15,
    "exposure": 0.10
}

