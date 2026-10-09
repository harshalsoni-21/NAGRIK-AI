# 🇮🇳 NagrikAI — AI Waste Intelligence Network
> *"Turn Waste Into Intelligence."*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Hackathon-AI%20First%20Product%20Builder%202026-gold.svg)]()

---

## 📌 Executive Summary & Pitch

In Indian cities, municipal solid waste management suffers from a fundamental information asymmetry: **passive citizen reporting portals** merely store photos of overflowing garbage heaps without extracting actionable operational metadata. Municipal ward sanitation inspectors, informal waste pickers (*safai karamcharis*), and urban local bodies (ULBs) lack automated answers to critical questions:
- *Is this waste dry recyclable, biodegradable, textile, or toxic e-waste?*
- *Does this accumulation threaten to choke open roadside stormwater drains ahead of monsoon flooding?*
- *Is this an isolated litter event or a recurring illegal dumping hotspot driven by infrastructural deficit?*

**NagrikAI** transforms this workflow from a passive photo upload into a **closed-loop civic intelligence network**:
```text
Citizen Waste Photo
     ↓
Computer Vision Preprocessing (224×224 RGB Normalization)
     ↓
Transfer-Learning Neural Classifier (MobileNetV3 / EfficientNet-B0)
     ↓
Waste Category (Plastic, Battery, Organic, Textiles, Glass, etc.)
     ↓
Civic Risk Engine (Environmental Hazard & Drain Blockage Choke Index)
     ↓
Stakeholder Action Directives (Citizen Bin, Worker PPE, ULB Logistics SLA)
     ↓
Hotspot Recurrence Analytics & 5-Factor Priority Scoring
     ↓
Ward Municipal Resolution & Material Recovery Facility (MRF) Diversion
```

---

## 🏛️ System Architecture

```text
NagrikAI/
├── app.py                     # Premium Streamlit UI (Classic Cream + Brown + White)
├── requirements.txt           # Clean dependencies
├── README.md                  # Comprehensive pitch & documentation
├── .gitignore                 # Standard exclusions
├── .env.example               # Environment variables template
│
├── data/
│   ├── raw/                   # Raw Kaggle 19,762 Solid Waste Dataset
│   ├── processed/             # Cleaned stratified splits
│   ├── demo/                  # 10 verified test samples (1 per class)
│   └── civic/
│       ├── demo_reports.csv   # Synthetic municipal citizen reports
│       └── hotspots.csv       # Priority-scored hotspot registry
│
├── models/
│   ├── best_model/            # model.pth & model_traced.pt checkpoints
│   ├── class_names.json       # 10 output class labels
│   └── model_metadata.json    # Evaluated test metrics & latency
│
├── reports/
│   ├── data_quality_report.txt# Real-time scan of all 19,762 dataset images
│   ├── eda/                   # Distribution, gallery, and augmentation charts
│   └── evaluation/            # Confusion matrix, per-class F1, comparison table
│
├── src/
│   ├── data/
│   │   ├── cleaning.py        # 0-byte, MD5 duplicate, and aspect audit
│   │   ├── preprocessing.py   # Parity-guaranteed tensor transformation
│   │   ├── dataset.py         # Stratified 70/15/15 splits & class weights
│   │   └── augmentation.py    # Realistic daylight/rotation augmentations
│   ├── eda/
│   │   └── analysis.py        # Matplotlib/Seaborn EDA generation engine
│   ├── training/
│   │   ├── train.py           # Reproducible training pipeline
│   │   ├── evaluate.py        # Confusion matrix & classification report
│   │   ├── model_selection.py # Transfer learning model factory
│   │   └── compare.py         # Comparative benchmark across architectures
│   ├── inference/
│   │   └── predictor.py       # Production inference engine & Gemini vision layer
│   ├── civic/
│   │   ├── risk.py            # Solid Waste Management Rules 2016 hazard mapping
│   │   ├── recommendations.py # Stakeholder-specific action directives
│   │   └── hotspot.py         # Explainable 5-factor hotspot priority score
│   └── utils/
│       ├── config.py          # Central paths, palette & constants
│       └── helpers.py         # Safe I/O and formatters
│
└── tests/
    ├── test_preprocessing.py  # Tensor shape & mode conversion tests
    ├── test_civic.py          # Risk score & hotspot priority tests
    ├── test_model.py          # Forward pass & predictor inference tests
    └── test_app.py            # Startup integrity & dataframe tests
```

---

## 📊 Dataset & Quality Audit

The model is trained on the verified **Kaggle Solid Waste Dataset**:
- **Total Images Scanned:** 19,762 images across 10 classes.
- **Corrupted / Zero-byte Files:** 0
- **Duplicate MD5 Hashes:** 0
- **Smallest Image:** 100 × 71 px | **Largest Image:** 7,786 × 6,283 px
- **Average Dimensions:** 420.8 × 431.6 px
- **Imbalance Ratio:** 5.64x (Clothes: 5,327 images vs Battery: 944 images). Handled via **Class-Weighted Cross Entropy Loss**.

### Class Breakdown
| Waste Category | Samples | % Share | Civic Stream | Drain Choke Risk |
|---|---|---|---|---|
| **Clothes** | 5,327 | 27.0% | Textile Take-back / Shredding | 🔴 CRITICAL |
| **Glass** | 3,061 | 15.5% | Sharp Hazard / Dry Recyclable | 🟢 LOW |
| **Plastic** | 1,984 | 10.0% | Dry Recyclable / Polymer MRF | 🔴 CRITICAL |
| **Shoes** | 1,977 | 10.0% | Bulky Composite Footwear | 🟠 HIGH |
| **Cardboard** | 1,825 | 9.2% | Dry Recyclable Fibre | 🟡 MEDIUM |
| **Paper** | 1,680 | 8.5% | Dry Recyclable Pulp | 🟡 MEDIUM |
| **Metal** | 1,020 | 5.2% | Scrap Metal Dealer (Kabadiwala) | 🟡 MEDIUM |
| **Biological** | 997 | 5.0% | Wet Waste (Biomethanation / Compost) | 🟠 HIGH |
| **Trash** | 947 | 4.8% | Mixed Non-Recyclable Residue | 🟠 HIGH |
| **Battery** | 944 | 4.8% | Domestic Hazardous / E-Waste (TSDF) | 🔴 CRITICAL |

---

## 🏆 Model Architecture Benchmark & Selection

We benchmarked three transfer-learning candidate architectures on the garbage dataset:

| Architecture | Test Accuracy | Macro F1 | Model Size | CPU Latency | Training Time |
|---|---|---|---|---|---|
| **MobileNetV3-Small** 🏆 | **76.2%** | **0.74** | **10.1 MB** | **18.4 ms** | **Fastest** |
| **EfficientNet-B0** (Runner-up) | 81.4% | 0.80 | 20.4 MB | 46.2 ms | Moderate |
| **ResNet-18** | 78.8% | 0.77 | 44.6 MB | 58.1 ms | Slow |

### Winner Selection Rationale: MobileNetV3-Small
1. **Ultra-Low Latency on Edge / CPU:** At **18.4 ms** per image, MobileNetV3-Small delivers instantaneous UI responsiveness without requiring expensive cloud GPU instances.
2. **Compact Deployment Footprint:** At only **10.1 MB** (vs 44.6 MB for ResNet-18), it minimizes cold-start penalties and memory footprint on municipal servers.
3. **Inverted Residual Blocks:** Retains high feature discriminability for textured waste categories (e.g., distinguishing cardboard fibre from paper, or shoes from clothing).

---

## 🎯 Explainable Civic Hotspot Priority Scoring

Unlike opaque black-box models, NagrikAI calculates municipal action priority using a **100% explainable 5-factor civic formula**:

$$\text{Priority Score} = 0.30 \times R + 0.25 \times S + 0.20 \times F + 0.15 \times D + 0.10 \times E$$

Where:
- **$R$ (Recurrence Factor - 30%):** Habitual dumping history (repeated incidents at the same coordinate).
- **$S$ (Severity Factor - 25%):** Toxicity, chemical hazard, and sharp laceration index.
- **$F$ (Complaint Frequency - 20%):** Density of unique citizen reports within a 200m radius.
- **$D$ (Drain Blockage Vulnerability - 15%):** Proximity to open roadside stormwater drains (preventing urban waterlogging).
- **$E$ (Public Exposure - 10%):** Pedestrian footfall and marketplace proximity.

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation
```bash
git clone https://github.com/your-org/NagrikAI.git
cd NagrikAI

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows (or source .venv/bin/activate on Linux/Mac)

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Dataset Cleaning & EDA
```bash
python -m src.data.cleaning
python -m src.eda.analysis
```

### 4. Retrain or Benchmark Models
```bash
python -m src.training.compare
```

### 5. Execute Test Suite
```bash
python -m unittest discover tests
```

---

## 🎨 UI/UX Design Language

NagrikAI is crafted using an authentic **Classic Cream + Earthy Brown + Warm White** design system:
- **Cream Background:** `#FBF8F3`
- **Warm White Cards:** `#FFFFFF` / `#F6F1EA`
- **Deep Brown Typography:** `#2C1E16`
- **Muted Earth Accent:** `#A3704C`
- **Eco Green:** `#2E7D32` (reserved strictly for recyclables)
- **Risk Indicators:** `#C62828` (Critical) & `#E65100` (Warning)

---

## ⚖️ Ethical Disclosures & Technical Honesty

1. **Single-Item vs Mixed Clutter Imagery:** The training dataset consists of single-item images. Real garbage piles can be mixed. NagrikAI includes an automated entropy/uncertainty heuristic that flags cluttered images:
   > *"Mixed / cluttered waste pattern detected. Single-label classifier identifies the dominant feature, but multi-stream elements may coexist."*
2. **Confidence Calibration:** We report **Model Probability** from Softmax outputs directly and explicitly state that it is uncalibrated.
3. **Synthetic Demo Data:** Municipal complaint logs, GPS coordinates, and historical hotspot frequencies are synthetic demo fixtures to demonstrate ULB workflows.
4. **AI-Assisted Root Cause Hypotheses:** Every root-cause analysis clearly includes: *Field validation by Ward Sanitation Inspector required.*

---

## 👥 Hackathon Team & Vision
Built with ❤️ for **AI First Product Builder Hackathon 2026**.
*"Don't just report waste. Turn every waste report into intelligence."*

