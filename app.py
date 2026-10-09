"""
NagrikAI - AI Waste Intelligence Network
"Turn Waste Into Intelligence."
AI First Product Builder Hackathon 2026

Run with:   python -m streamlit run app.py

Design goals of this rewrite
  * Never shows a blank page: every import / data load is guarded, and problems are
    printed on the page itself (see the "Debug info" box in the sidebar).
  * No fragile CSS (no negative z-index layers, no hidden-until-animated content).
  * No nested-quote f-strings, so it runs on Python 3.9+.
  * Works even if the src/ modules or demo CSVs are missing (falls back to demo data).
"""

import os
import sys
import site
import time
import platform
import traceback
from pathlib import Path

# -----------------------------------------------------------------------------
# Path bootstrap (before importing project modules)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    _user_site = site.getusersitepackages()
    if _user_site and Path(_user_site).exists() and _user_site not in sys.path:
        sys.path.append(_user_site)
except Exception:
    pass

_venv_site = BASE_DIR / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.append(str(_venv_site))

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# -----------------------------------------------------------------------------
# If someone runs "python app.py", relaunch properly through Streamlit
# -----------------------------------------------------------------------------
if __name__ == "__main__" and not st.runtime.exists():
    import subprocess
    print("\n" + "=" * 60)
    print("  NagrikAI - launching Streamlit...")
    print("  URL: http://localhost:8501")
    print("=" * 60 + "\n")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])
    except KeyboardInterrupt:
        print("\nNagrikAI stopped.")
    sys.exit(0)

# -----------------------------------------------------------------------------
# Page config MUST be the first Streamlit call
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NagrikAI | Turn Waste Into Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Guarded imports (so a missing package shows a message instead of a blank page)
# -----------------------------------------------------------------------------
IMPORT_PROBLEMS = []

try:
    import plotly.express as px
except Exception as exc:  # pragma: no cover
    px = None
    IMPORT_PROBLEMS.append("plotly: " + str(exc))

try:
    import folium
    from streamlit_folium import st_folium
except Exception as exc:  # pragma: no cover
    folium = None
    st_folium = None
    IMPORT_PROBLEMS.append("folium / streamlit-folium: " + str(exc))

# Project config (fallbacks keep the UI alive if src/ is missing)
CLASSES = ["battery", "biological", "cardboard", "clothes", "glass",
           "metal", "paper", "plastic", "shoes", "trash"]
CIVIC_DATA_DIR = BASE_DIR / "data" / "civic"
DEMO_DATA_DIR = BASE_DIR / "data" / "demo"
EVAL_REPORTS_DIR = BASE_DIR / "reports"
MODEL_METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

try:
    from src.utils.config import (
        CLASSES, CIVIC_DATA_DIR, DEMO_DATA_DIR, EVAL_REPORTS_DIR, MODEL_METADATA_PATH,
    )
except Exception as exc:
    IMPORT_PROBLEMS.append("src.utils.config: " + str(exc))

try:
    from src.utils.helpers import load_json
except Exception as exc:
    IMPORT_PROBLEMS.append("src.utils.helpers: " + str(exc))
    import json

    def load_json(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

WastePredictor = None
analyze_with_gemini = None
PREDICTOR_ERROR = ""
try:
    from src.inference.predictor import WastePredictor, analyze_with_gemini
except Exception as exc:
    PREDICTOR_ERROR = traceback.format_exc()
    IMPORT_PROBLEMS.append("src.inference.predictor: " + str(exc))

try:
    from src.civic.hotspot import generate_root_cause_hypothesis
except Exception as exc:
    IMPORT_PROBLEMS.append("src.civic.hotspot: " + str(exc))

    def generate_root_cause_hypothesis(row):
        return {
            "contributing_factors": [
                "Repeated reports at the same location",
                "Nearby drain or open drain exposure",
                "Irregular collection schedule (assumed)",
            ],
            "hypothesis": "Habitual dumping where no fixed collection point exists, made worse by drain proximity.",
            "recommended_civic_intervention": "Add a fixed bin or collection slot and schedule a clean-up drive.",
            "verification_note": "Heuristic only. Validate with the Ward Sanitation Inspector.",
        }

# Newer Streamlit uses width="stretch"; older uses use_container_width=True
try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _v = (1, 0)
STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

# -----------------------------------------------------------------------------
# Design system (CSS). Kept simple on purpose: every element is visible by default.
# -----------------------------------------------------------------------------
BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

:root{
  --bg:#060A12; --txt:#E8F1FF; --mut:#8FA6C2; --line:rgba(120,255,220,.18);
  --mint:#2BFFC6; --cyan:#22D3EE; --vio:#8B5CF6; --warn:#FBBF24; --bad:#FB7185;
}

html, body{background:#060A12 !important;}
.stApp{
  color:var(--txt);
  font-family:'Space Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
  background-color:#060A12 !important;
  background-image:
    radial-gradient(40% 35% at 18% 14%, rgba(43,255,198,.12), transparent 70%),
    radial-gradient(35% 30% at 86% 24%, rgba(139,92,246,.14), transparent 70%),
    radial-gradient(40% 35% at 60% 92%, rgba(34,211,238,.10), transparent 70%),
    linear-gradient(rgba(120,255,220,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(120,255,220,.035) 1px, transparent 1px) !important;
  background-size:auto, auto, auto, 48px 48px, 48px 48px !important;
  background-attachment:fixed !important;
}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
.block-container{padding-top:2.2rem; padding-bottom:3rem; max-width:1280px;}

/* Make default Streamlit text readable on the dark background */
.stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"]{color:var(--txt);}
.stApp [data-testid="stCaptionContainer"]{color:var(--mut);}

@keyframes rise{from{transform:translateY(8px);} to{transform:none;}}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes ping{0%{box-shadow:0 0 0 0 rgba(43,255,198,.65);} 100%{box-shadow:0 0 0 12px rgba(43,255,198,0);}}
@keyframes floaty{50%{transform:translateY(-10px);}}
@keyframes grow{from{width:0;}}
@keyframes nudge{50%{transform:translateX(4px);}}

.glass{position:relative; background:rgba(14,23,38,.88); border:1px solid rgba(43,255,198,.25); border-radius:18px;
  padding:20px 22px; margin-bottom:16px; box-shadow:0 10px 40px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.06);
  transition:transform .25s ease, border-color .25s ease; animation:rise .4s ease-out;}
.glass:hover{transform:translateY(-3px); border-color:rgba(43,255,198,.5);}
.glass h4{margin:0 0 10px 0; font-size:1.05rem; font-weight:600; color:#fff;}
.glass p{margin:6px 0; color:var(--txt); line-height:1.55;}
.glass p b{color:var(--mint); font-weight:600;}
.lab{font-size:.78rem; color:var(--mut); font-weight:500; letter-spacing:.02em; margin-bottom:6px;}
.sub{font-size:.8rem; color:var(--mut); margin-top:8px;}

.page-hero{padding:4px 0 2px; animation:rise .4s ease-out;}
.eyebrow{font-family:'JetBrains Mono',monospace; font-size:.78rem; color:var(--mint);}
.ph-title{font-size:2.4rem; font-weight:700; letter-spacing:-.02em; line-height:1.15; margin:.25rem 0 .4rem; color:#fff;}
.page-hero p{color:var(--mut); max-width:760px; margin:0; line-height:1.6;}
.route{height:2px; margin:16px 0 24px; border-radius:2px; background:linear-gradient(90deg, var(--mint), var(--cyan), var(--vio), transparent);}

.hero{display:grid; grid-template-columns:1.5fr .7fr; gap:24px; align-items:center; padding:36px 34px; margin-bottom:22px;
  border-radius:24px; border:1px solid rgba(43,255,198,.28);
  background:linear-gradient(135deg, rgba(14,23,38,.94), rgba(18,32,54,.9) 60%, rgba(25,45,74,.7));
  box-shadow:0 12px 40px rgba(0,0,0,.4);}
.brand-title{font-size:3.6rem; font-weight:700; letter-spacing:-.03em; line-height:1; color:#fff;}
.brand-title em{font-style:normal; color:var(--mint); text-shadow:0 0 30px rgba(43,255,198,.55);}
.type{display:inline-block; margin:16px 0 14px; font-family:'JetBrains Mono',monospace; font-size:1.05rem; color:var(--cyan);
  border-right:2px solid var(--mint); padding-right:8px;}
.hero p{color:var(--mut); max-width:620px; line-height:1.65; margin:0 0 6px;}
.hero-logo{display:flex; justify-content:center; animation:floaty 5s ease-in-out infinite; filter:drop-shadow(0 0 28px rgba(43,255,198,.45));}
.hero-badge{display:inline-block; margin-bottom:14px; padding:5px 13px; border-radius:20px; font-size:.78rem; font-weight:600;
  color:var(--mint); border:1px solid rgba(43,255,198,.35); background:rgba(43,255,198,.07);}
@media (max-width:900px){.hero{grid-template-columns:1fr;} .brand-title{font-size:2.6rem;} .hero-logo{display:none;}}

.kpi-num{font-family:'JetBrains Mono',monospace; font-size:2.3rem; font-weight:700; line-height:1.1; color:var(--mint);}
.kpi-num .unit{font-size:1rem; margin-left:6px; color:var(--mut);}

.badge{display:inline-block; padding:4px 12px; border-radius:8px; font-size:.8rem; font-weight:600;}
.badge-critical{background:rgba(251,113,133,.14); color:var(--bad); border:1px solid rgba(251,113,133,.4);}
.badge-warn{background:rgba(251,191,36,.12); color:var(--warn); border:1px solid rgba(251,191,36,.4);}
.badge-eco{background:rgba(43,255,198,.10); color:var(--mint); border:1px solid rgba(43,255,198,.4);}
.meter{height:6px; margin-top:12px; border-radius:6px; background:rgba(255,255,255,.08); overflow:hidden;}
.meter i{display:block; height:100%; width:var(--w); border-radius:6px; background:linear-gradient(90deg,var(--c1),var(--c2));
  animation:grow 1.1s cubic-bezier(.16,1,.3,1) .2s backwards;}

.flow{display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:18px;}
.step{padding:10px 14px; border-radius:12px; border:1px solid var(--line); background:rgba(14,23,38,.88); font-size:.9rem; font-weight:600; color:var(--txt);}
.step small{display:block; color:var(--mut); font-size:.74rem; font-weight:400; margin-top:2px;}
.arrow{color:var(--mint); animation:nudge 1.4s ease-in-out infinite;}

.layer{display:grid; grid-template-columns:44px 1fr; gap:14px; align-items:center; padding:14px 18px; margin-bottom:8px;
  border-radius:14px; border:1px solid var(--line); background:rgba(14,23,38,.88);}
.layer .no{width:36px; height:36px; border-radius:10px; display:grid; place-items:center; font-family:'JetBrains Mono',monospace;
  font-weight:700; color:#021016; background:linear-gradient(135deg,var(--mint),var(--cyan));}
.layer b{display:block; font-size:1rem; color:var(--txt);}
.layer span{color:var(--mut); font-size:.86rem;}
.link{height:18px; margin:0 0 8px 29px; width:2px; background:linear-gradient(var(--mint), transparent);}

.core{display:flex; align-items:center; gap:16px; padding:16px 22px;}
.core i{width:38px; height:38px; border-radius:50%; border:3px solid transparent; border-top-color:var(--mint); border-right-color:var(--cyan); animation:spin .8s linear infinite;}
.dot{display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--mint); margin-right:8px; animation:ping 1.6s infinite;}
.stat-grid{display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:6px 0 14px;}
.stat-grid div{padding:9px 11px; border-radius:11px; background:rgba(14,23,38,.88); border:1px solid var(--line);}
.stat-grid small{display:block; color:var(--mut); font-size:.7rem;}
.stat-grid b{font-size:.84rem; font-family:'JetBrains Mono',monospace; word-break:break-word; color:var(--txt);}
.brand-row{display:flex; align-items:center; gap:12px; padding:6px 0 14px;}
.brand-row .nm{font-size:1.45rem; font-weight:700; letter-spacing:-.02em; line-height:1; color:#fff;}
.brand-row .nm em{font-style:normal; color:var(--mint);}
.brand-row small{display:block; font-size:.72rem; color:var(--mut); margin-top:3px;}

.stButton > button, .stFormSubmitButton > button{border-radius:14px; font-weight:600; padding:.6rem 1rem; color:var(--txt);
  background:rgba(14,23,38,.9); border:1px solid rgba(43,255,198,.3); transition:transform .18s ease, box-shadow .25s ease, border-color .25s ease;}
.stButton > button:hover, .stFormSubmitButton > button:hover{transform:translateY(-2px); border-color:var(--mint); color:#fff;
  box-shadow:0 8px 28px rgba(43,255,198,.18);}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"], .stFormSubmitButton > button{
  background:linear-gradient(100deg, var(--mint), var(--cyan)) !important; color:#021016 !important; border:0 !important;}

[data-testid="stSidebar"]{background:#090F1C !important; border-right:1px solid rgba(43,255,198,.2) !important;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:6px; display:flex; flex-direction:column;}
[data-testid="stSidebar"] [role="radiogroup"] label{width:100%; padding:10px 14px; border-radius:12px; border:1px solid rgba(255,255,255,.06);
  background:rgba(255,255,255,.03); cursor:pointer;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(43,255,198,.08); border-color:rgba(43,255,198,.3);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:rgba(43,255,198,.15) !important; border-color:#2BFFC6 !important;}
[data-testid="stSidebar"] [role="radiogroup"] label p{margin:0; font-weight:600; color:#E8F1FF !important;}

[data-testid="stFileUploaderDropzone"]{border:1.5px dashed rgba(43,255,198,.4); background:rgba(43,255,198,.04); border-radius:16px;}
.stTabs [data-baseweb="tab"]{border-radius:10px; padding:8px 16px; font-weight:600;}
.stTabs [aria-selected="true"]{color:var(--mint) !important;}
[data-testid="stImage"] img{border-radius:16px; border:1px solid var(--line);}

@media (prefers-reduced-motion: reduce){
  *, *::before, *::after{animation-duration:.01ms !important; transition-duration:.01ms !important;}
}

/* ===== OPENING SPLASH ===== */
.splash{position:fixed; top:0; left:0; right:0; bottom:0; z-index:999999; display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:16px; pointer-events:none;
  background:radial-gradient(circle at 50% 40%, #0E1B2E, #060A12 70%);
  animation:splashOut .7s ease 2.7s forwards;}
@keyframes splashOut{to{opacity:0; visibility:hidden;}}
@keyframes sDraw{to{stroke-dashoffset:0;}}
@keyframes sRise{from{transform:translateY(14px); opacity:0;} to{transform:none; opacity:1;}}
@keyframes sBar{from{width:0;} to{width:100%;}}
@keyframes sGlow{0%,100%{filter:drop-shadow(0 0 10px rgba(43,255,198,.35));} 50%{filter:drop-shadow(0 0 34px rgba(43,255,198,.8));}}
.splash-logo{animation:sGlow 1.8s ease-in-out infinite;}
.splash .sp{stroke-dasharray:1; stroke-dashoffset:1; animation:sDraw 1.3s ease forwards;}
.splash .sp2{animation-delay:.5s;}
.splash .sdot{animation:sRise .5s ease 1.2s backwards;}
.splash-name{font-size:2.8rem; font-weight:700; letter-spacing:-.02em; color:#fff; animation:sRise .8s ease .9s backwards;}
.splash-name em{font-style:normal; color:#2BFFC6; text-shadow:0 0 28px rgba(43,255,198,.6);}
.splash-tag{font-family:'JetBrains Mono',monospace; font-size:.95rem; color:#22D3EE; animation:sRise .8s ease 1.3s backwards;}
.splash-bar{width:220px; height:3px; border-radius:3px; background:rgba(255,255,255,.1); overflow:hidden; margin-top:8px;}
.splash-bar i{display:block; height:100%; width:100%; background:linear-gradient(90deg,#2BFFC6,#22D3EE); animation:sBar 2s ease .5s backwards;}
"""


def M(markup: str) -> str:
    """Minify HTML so Streamlit's markdown parser never sees blank or indented lines."""
    return " ".join(line.strip() for line in markup.strip().splitlines())


def H(markup: str):
    st.markdown(M(markup), unsafe_allow_html=True)


H("<style>" + BASE_CSS + "</style>")


def splash_html() -> str:
    return (
        '<div class="splash">'
        '<div class="splash-logo">'
        '<svg viewBox="0 0 100 100" width="140" height="140" xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="gsp" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#2BFFC6"/><stop offset="1" stop-color="#22D3EE"/></linearGradient></defs>'
        '<polygon class="sp" pathLength="1" points="50,4 90,27 90,73 50,96 10,73 10,27" fill="#2BFFC6" fill-opacity=".06" '
        'stroke="url(#gsp)" stroke-width="3" stroke-linejoin="round"/>'
        '<path class="sp sp2" pathLength="1" d="M50 74 C28 62 27 40 50 24 C73 40 72 62 50 74 Z" fill="url(#gsp)" fill-opacity=".16" '
        'stroke="url(#gsp)" stroke-width="2.5" stroke-linejoin="round"/>'
        '<path class="sp sp2" pathLength="1" d="M50 72 V42 M50 58 L40 49 M50 52 L60 44" stroke="url(#gsp)" '
        'stroke-width="2.5" stroke-linecap="round" fill="none"/>'
        '<circle class="sdot" cx="50" cy="24" r="3.2" fill="#2BFFC6"/>'
        '<circle class="sdot" cx="40" cy="49" r="2.4" fill="#22D3EE"/>'
        '<circle class="sdot" cx="60" cy="44" r="2.4" fill="#22D3EE"/>'
        '</svg></div>'
        '<div class="splash-name">Nagrik<em>AI</em></div>'
        '<div class="splash-tag">Turn Waste Into Intelligence.</div>'
        '<div class="splash-bar"><i></i></div>'
        '</div>'
    )


# Opening animation: shown only once per browser session
if not st.session_state.get("splash_done"):
    H(splash_html())
    st.session_state["splash_done"] = True


# -----------------------------------------------------------------------------
# Small UI helpers
# -----------------------------------------------------------------------------
def logo_svg(size: int = 64, uid: str = "a") -> str:
    return (
        f'<svg viewBox="0 0 100 100" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">'
        f'<defs><linearGradient id="g{uid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="#2BFFC6"/><stop offset="1" stop-color="#22D3EE"/></linearGradient></defs>'
        f'<polygon points="50,4 90,27 90,73 50,96 10,73 10,27" fill="#2BFFC6" fill-opacity=".06" '
        f'stroke="url(#g{uid})" stroke-width="3" stroke-linejoin="round"/>'
        f'<path d="M50 74 C28 62 27 40 50 24 C73 40 72 62 50 74 Z" fill="url(#g{uid})" fill-opacity=".16" '
        f'stroke="url(#g{uid})" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<path d="M50 72 V42 M50 58 L40 49 M50 52 L60 44" stroke="url(#g{uid})" stroke-width="2.5" '
        f'stroke-linecap="round" fill="none"/>'
        f'<circle cx="50" cy="24" r="3.2" fill="#2BFFC6"/><circle cx="40" cy="49" r="2.4" fill="#22D3EE"/>'
        f'<circle cx="60" cy="44" r="2.4" fill="#22D3EE"/></svg>'
    )


def badge(text, kind: str) -> str:
    return '<span class="badge badge-' + kind + '">' + str(text) + "</span>"


def risk_kind(level) -> str:
    level = str(level).upper()
    if level in ("HIGH", "CRITICAL"):
        return "critical"
    if level == "MEDIUM":
        return "warn"
    return "eco"


METER_COLORS = {"critical": ("#FB7185", "#F59E0B"), "warn": ("#FBBF24", "#F59E0B"), "eco": ("#2BFFC6", "#22D3EE")}


def kpi(label: str, value, sub: str = "", unit: str = ""):
    unit_html = '<span class="unit">' + unit + "</span>" if unit else ""
    sub_html = '<div class="sub">' + sub + "</div>" if sub else ""
    H('<div class="glass"><div class="kpi-num">' + str(value) + unit_html + "</div>"
      '<div class="lab">' + label + "</div>" + sub_html + "</div>")


def risk_card(title: str, level, score):
    kind = risk_kind(level)
    c1, c2 = METER_COLORS[kind]
    pct = pct_value(score)
    H('<div class="glass"><div class="lab">' + title + "</div>" + badge(level, kind) +
      '<div class="meter" style="--w:' + str(pct) + "%;--c1:" + c1 + ";--c2:" + c2 + '"><i></i></div>'
      '<div class="sub">Score ' + str(score) + "/100</div></div>")


def page_header(eyebrow: str, title: str, sub: str):
    H('<div class="page-hero"><div class="eyebrow">' + eyebrow + '</div><div class="ph-title">' + title + "</div>"
      "<p>" + sub + "</p></div><div class='route'></div>")


def style_fig(fig, height: int = 320):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", color="#E8F1FF"),
        height=height, margin=dict(l=8, r=8, t=8, b=8),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def show_chart(fig):
    st.plotly_chart(fig, config={"displayModeBar": False}, **STRETCH)


def pct_value(p) -> float:
    try:
        return max(0.0, min(100.0, float(str(p).replace("%", "").strip())))
    except Exception:
        return 0.0


NEON = ["#2BFFC6", "#22D3EE", "#8B5CF6", "#FBBF24", "#FB7185", "#34D399", "#60A5FA", "#F472B6", "#A3E635", "#F97316"]
BAR_SCALE = ["#123845", "#22D3EE", "#2BFFC6"]

# -----------------------------------------------------------------------------
# Data loading (with demo fallbacks so pages are never empty)
# -----------------------------------------------------------------------------
def _demo_hotspots() -> pd.DataFrame:
    rows = [
        ("Ward 23 - Koramangala", 12.9352, 77.6245, 91, "CRITICAL", "plastic", "HIGH", 14, "Action Required"),
        ("Ward 12 - Indiranagar", 12.9784, 77.6408, 84, "HIGH", "clothes", "HIGH", 11, "Action Required"),
        ("Ward 18 - Malleshwaram", 13.0035, 77.5710, 72, "HIGH", "biological", "MEDIUM", 9, "Monitoring"),
        ("Ward 45 - Shivajinagar", 12.9857, 77.6057, 66, "MEDIUM", "paper", "MEDIUM", 7, "Monitoring"),
        ("Ward 07 - Jayanagar", 12.9308, 77.5838, 58, "MEDIUM", "glass", "LOW", 5, "Monitoring"),
        ("Ward 31 - Whitefield", 12.9698, 77.7500, 49, "LOW", "cardboard", "LOW", 3, "Resolved"),
    ]
    cols = ["ward", "latitude", "longitude", "priority_score", "priority_level",
            "primary_waste", "drain_risk", "recurrence_count", "status"]
    return pd.DataFrame(rows, columns=cols)


def _demo_reports() -> pd.DataFrame:
    wards = ["Ward 23 - Koramangala", "Ward 12 - Indiranagar", "Ward 18 - Malleshwaram",
             "Ward 45 - Shivajinagar", "Ward 07 - Jayanagar", "Ward 31 - Whitefield"]
    cats = ["plastic", "clothes", "biological", "paper", "glass", "cardboard", "metal", "trash"]
    rows = []
    for i in range(40):
        rows.append({
            "report_id": "NGK-" + str(1001 + i),
            "timestamp": "2026-09-" + str(1 + i % 28).zfill(2) + " 10:00",
            "ward": wards[i % len(wards)],
            "waste_category": cats[(i * 3) % len(cats)],
            "severity": ["LOW", "MEDIUM", "HIGH"][i % 3],
            "drain_risk": ["LOW", "MEDIUM", "HIGH"][(i + 1) % 3],
            "status": ["Pending", "Resolved", "Action Required"][i % 3],
            "recurring": i % 2 == 0,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_civic_reports():
    try:
        path = Path(CIVIC_DATA_DIR) / "demo_reports.csv"
        if path.exists():
            return pd.read_csv(path), False
    except Exception:
        pass
    return _demo_reports(), True


@st.cache_data(show_spinner=False)
def load_civic_hotspots():
    try:
        path = Path(CIVIC_DATA_DIR) / "hotspots.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "priority_score" in df.columns:
                df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
            return df, False
    except Exception:
        pass
    return _demo_hotspots(), True


@st.cache_data(show_spinner=False)
def load_model_metadata():
    try:
        if Path(MODEL_METADATA_PATH).exists():
            return load_json(MODEL_METADATA_PATH)
    except Exception:
        pass
    return {}


@st.cache_resource(show_spinner=False)
def get_cached_predictor():
    return WastePredictor()


@st.cache_resource(show_spinner=False)
def build_hotspot_map(_df, sig):
    """Map with pulsing priority markers. Cached so it is built only once."""
    m = folium.Map(location=[12.9550, 77.6200], zoom_start=11, tiles="OpenStreetMap")
    m.get_root().header.add_child(folium.Element(
        "<style>@keyframes ngPulse{0%{transform:scale(.5);opacity:.9}100%{transform:scale(1.9);opacity:0}}"
        ".ng-div{background:transparent!important;border:0!important}"
        ".ng-pin{position:relative;width:100%;height:100%}"
        ".ng-pin .core{position:absolute;top:28%;left:28%;right:28%;bottom:28%;border-radius:50%;background:var(--c);box-shadow:0 0 14px var(--c)}"
        ".ng-pin .ring{position:absolute;top:0;left:0;right:0;bottom:0;border-radius:50%;border:2px solid var(--c);animation:ngPulse 2s ease-out infinite}</style>"
    ))
    for _, row in _df.iterrows():
        score = float(row["priority_score"])
        color = "#FB7185" if score >= 80 else ("#FBBF24" if score >= 65 else "#2BFFC6")
        size = int(22 + score / 4)
        popup_html = (
            "<div style='font-family:sans-serif;font-size:12px;width:200px;'>"
            "<b>" + str(row["ward"]) + "</b><br><b>Priority:</b> " + str(row["priority_score"]) + "/100 (" + str(row["priority_level"]) + ")<br>"
            "<b>Primary waste:</b> " + str(row["primary_waste"]) + "<br><b>Drain risk:</b> " + str(row["drain_risk"]) + "<br>"
            "<b>Recurrence:</b> " + str(row["recurrence_count"]) + " events<br><b>Status:</b> " + str(row["status"]) + "</div>"
        )
        icon = folium.DivIcon(
            html='<div class="ng-pin" style="--c:' + color + '"><span class="ring"></span><span class="core"></span></div>',
            icon_size=(size, size), icon_anchor=(size // 2, size // 2), class_name="ng-div",
        )
        folium.Marker(
            [row["latitude"], row["longitude"]], icon=icon,
            popup=folium.Popup(popup_html, max_width=250), tooltip=str(row["ward"]),
        ).add_to(m)
    return m


# -----------------------------------------------------------------------------
# Session state + navigation
# -----------------------------------------------------------------------------
PAGES = [
    ("🏠 Home", "Command center"),
    ("📸 Report Waste", "Citizen portal"),
    ("🔬 AI Vision & Analysis", "Model lab"),
    ("📊 Waste Intelligence", "Analytics"),
    ("📍 Hotspots & Map", "Geospatial"),
    ("🌱 Civic Impact", "Outcomes"),
    ("ℹ️ About AI & Architecture", "Transparency"),
]
NAV = [p[0] for p in PAGES]

st.session_state.setdefault("nav", NAV[0])
st.session_state.setdefault("user_reports", [])
st.session_state.setdefault("selected_sample", None)
st.session_state.setdefault("current_prediction", None)
st.session_state.setdefault("current_img", None)
st.session_state.setdefault("img_key", None)
st.session_state.setdefault("up_n", 0)


def go(page: str):
    """Button callback: switch page (runs before the rerun, so setting the radio key is safe)."""
    st.session_state["nav"] = page


def pick_sample(path):
    st.session_state["selected_sample"] = str(path)
    st.session_state["up_n"] += 1  # resets the uploader so the sample is used


def load_image(src) -> Image.Image:
    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((1280, 1280))
    return img


df_reports, reports_are_demo = load_civic_reports()
df_hotspots, hotspots_are_demo = load_civic_hotspots()
metadata = load_model_metadata()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    H('<div class="brand-row">' + logo_svg(44, "sb") +
      '<div><div class="nm">Nagrik<em>AI</em></div><small>Waste Intelligence Network</small></div></div>')
    nav_selection = st.radio("Navigation", NAV, key="nav", label_visibility="collapsed")

    arch = str(metadata.get("model_architecture", "MobileNetV3-Small"))
    latency = str(metadata.get("inference_latency_cpu_ms", 18.4))
    H('<div style="margin-top:18px"><span class="dot"></span><span class="lab">System online</span></div>'
      '<div class="stat-grid">'
      '<div><small>Model</small><b>' + arch + '</b></div>'
      '<div><small>Classes</small><b>' + str(len(CLASSES)) + '</b></div>'
      '<div><small>CPU latency</small><b>' + latency + ' ms</b></div>'
      '<div><small>Dataset</small><b>19,762 imgs</b></div></div>')

    with st.expander("🔑 Gemini Vision (optional)"):
        gemini_key_input = st.text_input("Gemini API key", type="password",
                                         help="Adds LLM scene reasoning on top of the local classifier.")
        if gemini_key_input:
            os.environ["GEMINI_API_KEY"] = gemini_key_input
            st.success("Gemini layer active")

    with st.expander("🛠 Debug info"):
        st.write("Python:", platform.python_version())
        st.write("Streamlit:", st.__version__)
        st.write("App folder:", str(BASE_DIR))
        st.write("Civic data folder exists:", Path(CIVIC_DATA_DIR).exists())
        st.write("Demo images folder exists:", Path(DEMO_DATA_DIR).exists())
        st.write("Hotspots from demo fallback:", hotspots_are_demo)
        st.write("Reports from demo fallback:", reports_are_demo)
        st.write("Predictor import OK:", WastePredictor is not None)
        if IMPORT_PROBLEMS:
            st.warning("Import problems:")
            for item in IMPORT_PROBLEMS:
                st.code(item)

# Show import problems loudly at the top so the page is never silently empty
if px is None:
    st.error("plotly is not installed. Run: pip install plotly")
if folium is None or st_folium is None:
    st.error("folium / streamlit-folium is not installed. Run: pip install folium streamlit-folium")

# =============================================================================
# PAGE 0: HOME
# =============================================================================
if nav_selection == NAV[0]:
    H('<div class="hero"><div>'
      '<span class="hero-badge">AI First Product Builder Hackathon 2026</span>'
      '<div class="brand-title">Nagrik<em>AI</em></div>'
      '<div class="type">Turn Waste Into Intelligence.</div>'
      '<p>A citizen uploads a waste photo. NagrikAI classifies the waste stream, scores environmental and drain-blockage risk, '
      'finds recurring dumping hotspots, and turns it all into prioritized action for the municipal ward.</p>'
      '</div><div class="hero-logo">' + logo_svg(190, "hero") + "</div></div>")

    cta1, cta2 = st.columns(2)
    with cta1:
        st.button("📸 Report waste now", type="primary", key="cta_report", on_click=go, args=(NAV[1],), **STRETCH)
    with cta2:
        st.button("📍 Explore hotspot map", key="cta_map", on_click=go, args=(NAV[4],), **STRETCH)

    st.write("")
    n_action = int((df_hotspots["status"] == "Action Required").sum()) if "status" in df_hotspots else 0
    n_drain = (int(df_hotspots["drain_risk"].astype(str).str.upper().isin(["HIGH", "CRITICAL"]).sum())
               if "drain_risk" in df_hotspots else 0)
    n_total = len(df_reports) + len(st.session_state["user_reports"])
    n_resolved = int((df_reports["status"] == "Resolved").sum()) if "status" in df_reports else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Active hotspots", n_action)
    with k2:
        kpi("Critical drain risks", n_drain)
    with k3:
        kpi("Citizen reports", n_total)
    with k4:
        kpi("Resolved and cleared", n_resolved)

    st.markdown("#### How a photo becomes action")
    steps = [
        ("Waste photo", "Citizen upload"), ("Vision model", "MobileNetV3, 224px"), ("Waste class", "10 categories"),
        ("Risk engine", "Hazard + drain index"), ("Action plan", "Citizen, worker, ward"),
        ("Hotspot score", "Recurrence analytics"), ("Resolution", "Ward and MRF routing"),
    ]
    chips = '<span class="arrow">&rarr;</span>'.join(
        '<div class="step">' + t + "<small>" + s + "</small></div>" for t, s in steps
    )
    H('<div class="flow">' + chips + "</div>")

    st.markdown("#### Open a module")
    tiles = [
        (NAV[2], "Probabilities, calibration and the optional Gemini layer."),
        (NAV[3], "Waste mix and ward-level incident density."),
        (NAV[5], "Diversion, response time and drains protected."),
        (NAV[6], "Architecture, benchmarks and honest limitations."),
    ]
    for col, (label, desc) in zip(st.columns(4), tiles):
        with col:
            st.button(label, key="tile_" + label, on_click=go, args=(label,), **STRETCH)
            H('<div class="sub" style="margin-top:-4px">' + desc + "</div>")

# =============================================================================
# PAGE 1: REPORT WASTE
# =============================================================================
elif nav_selection == NAV[1]:
    page_header(PAGES[1][1], "Report waste, get an action plan",
                "Upload a photo or tap a sample class. The result appears in under a second.")

    if WastePredictor is None:
        st.error("The AI model could not be loaded, so analysis is unavailable. Details below.")
        st.code(PREDICTOR_ERROR or "Unknown import error")

    st.markdown("##### Quick samples")
    class_list = list(CLASSES)
    for start in range(0, len(class_list), 5):
        for col, cls in zip(st.columns(5), class_list[start:start + 5]):
            with col:
                st.button(cls.capitalize(), key="sample_" + cls, on_click=pick_sample,
                          args=(Path(DEMO_DATA_DIR) / (cls + ".jpg"),), **STRETCH)

    up_col, prev_col = st.columns(2)
    with up_col:
        uploaded_file = st.file_uploader("Upload waste image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"],
                                         key="uploader_" + str(st.session_state["up_n"]))

    image_to_analyze, image_title, img_key = None, "", None
    if uploaded_file is not None:
        try:
            image_to_analyze = load_image(uploaded_file)
            image_title = uploaded_file.name
            img_key = "up:" + uploaded_file.name + ":" + str(uploaded_file.size)
        except Exception:
            st.error("That file could not be read. Upload a valid JPG, PNG or WEBP image.")
    elif st.session_state.get("selected_sample"):
        sample_path = Path(st.session_state["selected_sample"])
        if sample_path.exists():
            image_to_analyze = load_image(sample_path)
            image_title, img_key = sample_path.name, "sample:" + sample_path.name
        else:
            st.warning("Sample image not found: " + sample_path.name + ". Add it to the demo data folder.")

    if img_key and st.session_state["img_key"] != img_key:
        st.session_state["img_key"] = img_key
        st.session_state["current_prediction"] = None

    with prev_col:
        if image_to_analyze is not None:
            caption = image_title + " (" + str(image_to_analyze.size[0]) + " x " + str(image_to_analyze.size[1]) + " px)"
            st.image(image_to_analyze, caption=caption, **STRETCH)
        else:
            st.info("No image yet. Upload a file or tap a sample.")

    if image_to_analyze is not None and WastePredictor is not None:
        if st.button("🔍 Analyze waste", type="primary", key="analyze", **STRETCH):
            slot = st.empty()
            slot.markdown(M('<div class="glass core"><i></i><div><b>AI is reading the image</b>'
                            '<div class="sub" style="margin:2px 0 0">Classifying and scoring civic risk</div></div></div>'),
                          unsafe_allow_html=True)
            try:
                prediction = get_cached_predictor().predict(image_to_analyze)
                st.session_state["current_prediction"] = prediction
                st.session_state["current_img"] = image_to_analyze
            except Exception:
                st.error("Prediction failed. Details below.")
                st.code(traceback.format_exc())
            slot.empty()

    pred = st.session_state.get("current_prediction")
    if pred:
        prof, plan = pred["civic_profile"], pred["action_plan"]
        st.markdown("### Analysis result")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            p_val = pct_value(pred["model_probability_pct"])
            H('<div class="glass"><div class="lab">Primary class</div>'
              '<div style="font-size:1.55rem;font-weight:700;margin:2px 0;color:#fff">' + str(pred["primary_class"]) + "</div>"
              '<div class="meter" style="--w:' + str(p_val) + '%;--c1:#2BFFC6;--c2:#22D3EE"><i></i></div>'
              '<div class="sub">Model probability ' + str(pred["model_probability_pct"]) + "</div></div>")
        with r2:
            stream = str(prof["segregation_stream"])
            seg_kind = "eco" if "Dry" in stream else ("warn" if "Wet" in stream else "critical")
            H('<div class="glass"><div class="lab">Segregation stream</div>' + badge(stream, seg_kind) +
              '<div class="sub">' + str(prof["civic_category"]) + "</div></div>")
        with r3:
            risk_card("Environmental risk", prof["environmental_risk"], prof["environmental_risk_score"])
        with r4:
            risk_card("Drain blockage risk", prof["drain_risk"], prof["drain_risk_score"])

        note = plan.get("mixed_scene_note")
        if pred.get("is_mixed_clutter") or note:
            st.warning("⚠️ **Cluttered scene:** " + str(note or "This looks like mixed waste. Treat the label as the dominant stream only."))

        t_cit, t_wrk, t_ulb = st.tabs(["👤 Citizen", "👷 Sanitation worker", "🏛️ Municipal ward"])
        with t_cit:
            c = plan["citizen"]
            H('<div class="glass"><h4>What to do now</h4><p><b>Disposal bin:</b> ' + str(c["bin_color"]) + "</p>"
              "<p><b>Immediate step:</b> " + str(c["primary_action"]) + "</p><p><b>Safety:</b> " + str(c["safety_tip"]) + "</p></div>")
        with t_wrk:
            w = plan["sanitation_workers"]
            H('<div class="glass"><h4>Worker safety protocol</h4><p><b>Protective equipment:</b> ' + str(w["ppe_required"]) + "</p>"
              "<p><b>Collection urgency:</b> " + str(w["handling_priority"]) + "</p><p><b>Secondary route:</b> " + str(w["route_channel"]) + "</p></div>")
        with t_ulb:
            u = plan["municipal_authority"]
            H('<div class="glass"><h4>Ward logistics and facility route</h4><p><b>Response target:</b> within ' + str(u["sla_target_hours"]) + " hours</p>"
              "<p><b>Resource recovery:</b> " + str(u["recycling_potential"]) + "</p><p><b>Monsoon flooding risk:</b> " + str(u["monsoon_vulnerability"]) + "</p></div>")

        st.markdown("#### Send this report to the ward")
        with st.form("civic_report_submission"):
            f1, f2, f3 = st.columns(3)
            with f1:
                ward_choice = st.selectbox("Municipal ward", [
                    "Ward 23 - Koramangala", "Ward 12 - Indiranagar", "Ward 18 - Malleshwaram",
                    "Ward 45 - Shivajinagar", "Ward 07 - Jayanagar", "Ward 31 - Whitefield"])
            with f2:
                landmark_input = st.text_input("Landmark or location note", "Near roadside stormwater drain")
            with f3:
                recurrence_flag = st.checkbox("Habitual or recurring dumping spot", value=True)
            if st.form_submit_button("🚀 Submit report"):
                rep_id = "NGK-" + str(1000 + len(df_reports) + len(st.session_state["user_reports"]) + 1)
                st.session_state["user_reports"].append({
                    "report_id": rep_id, "latitude": 12.9352, "longitude": 77.6245, "ward": ward_choice,
                    "waste_category": pred["primary_class"], "severity": prof["environmental_risk"],
                    "drain_risk": prof["drain_risk"], "status": "Pending",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M"), "recurring": recurrence_flag,
                    "note": landmark_input,
                })
                st.success("Report " + rep_id + " registered and sent to the " + ward_choice + " sanitation roster.")
                st.balloons()

# =============================================================================
# PAGE 2: AI VISION & ANALYSIS
# =============================================================================
elif nav_selection == NAV[2]:
    page_header(PAGES[2][1], "Inside the model's decision",
                "Class probabilities, calibration notes and the optional generative reasoning layer.")

    pred = st.session_state.get("current_prediction")
    if pred:
        col_chart, col_info = st.columns([1.3, 1])
        with col_chart:
            st.markdown("##### Top candidates")
            if px is not None:
                df_c = pd.DataFrame(pred["top_candidates"])
                fig = px.bar(df_c, x="model_probability", y="class", orientation="h", text="percentage",
                             color="model_probability", color_continuous_scale=BAR_SCALE,
                             labels={"model_probability": "Softmax probability", "class": "Waste class"})
                fig.update_layout(xaxis=dict(range=[0, 1]), coloraxis_showscale=False, showlegend=False)
                fig.update_yaxes(autorange="reversed")
                show_chart(style_fig(fig, 310))
        with col_info:
            H('<div class="glass"><h4>Calibration check</h4><p><b>Top-1 class:</b> ' + str(pred["primary_class"]) + "</p>"
              "<p><b>Softmax probability:</b> " + str(pred["model_probability_pct"]) + "</p>"
              "<p><b>Status:</b> uncalibrated model probability</p>"
              '<div class="sub">We label this "model probability", not certainty. Uncalibrated softmax can be overconfident on images unlike the training data.</div></div>')

        st.markdown("### Generative reasoning layer (Google Gemini)")
        if os.getenv("GEMINI_API_KEY") and st.session_state.get("current_img") is not None and analyze_with_gemini:
            if st.button("Run Gemini scene analysis", type="primary", key="gemini_run"):
                with st.spinner("Asking Gemini..."):
                    result = analyze_with_gemini(st.session_state["current_img"], pred)
                if result:
                    st.json(result)
                else:
                    st.warning("Gemini returned nothing. Check the API key and quota.")
        else:
            st.info("Gemini is optional. Add a key in the sidebar to turn it on. Local classification and risk scoring work fully offline.")
    else:
        st.info("No image analyzed yet.")
        st.button("📸 Go to Report Waste", type="primary", key="vision_to_report", on_click=go, args=(NAV[1],))

# =============================================================================
# PAGE 3: WASTE INTELLIGENCE
# =============================================================================
elif nav_selection == NAV[3]:
    page_header(PAGES[3][1], "What the city is throwing away",
                "Live aggregates from every citizen report, including the ones you submit in this session.")

    combined = df_reports.copy()
    if st.session_state["user_reports"]:
        combined = pd.concat([combined, pd.DataFrame(st.session_state["user_reports"])], ignore_index=True)

    if combined.empty or "waste_category" not in combined.columns:
        st.info("No reports yet. Submit one from Report Waste.")
        st.button("📸 Report waste", type="primary", key="intel_to_report", on_click=go, args=(NAV[1],))
    elif px is None:
        st.error("plotly is missing, so charts cannot be drawn. Run: pip install plotly")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Waste composition")
            comp = combined["waste_category"].value_counts().reset_index()
            comp.columns = ["Category", "Reports"]
            fig = px.pie(comp, names="Category", values="Reports", hole=0.55, color_discrete_sequence=NEON)
            fig.update_traces(textinfo="percent", marker=dict(line=dict(color="#060A12", width=2)))
            show_chart(style_fig(fig, 330))
        with c2:
            st.markdown("##### Incidents by ward")
            wards = combined["ward"].value_counts().reset_index()
            wards.columns = ["Ward", "Incidents"]
            fig = px.bar(wards.head(7), x="Incidents", y="Ward", orientation="h",
                         color="Incidents", color_continuous_scale=BAR_SCALE)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_yaxes(autorange="reversed")
            show_chart(style_fig(fig, 330))

        st.markdown("##### Recent reports")
        cols = ["report_id", "timestamp", "ward", "waste_category", "severity", "drain_risk", "status", "recurring"]
        st.dataframe(combined[[c for c in cols if c in combined.columns]].tail(10), **STRETCH)

# =============================================================================
# PAGE 4: HOTSPOTS & MAP
# =============================================================================
elif nav_selection == NAV[4]:
    page_header(PAGES[4][1], "Where waste keeps coming back",
                "Recurring dump clusters ranked by priority, with vulnerable drains and a root-cause hypothesis.")

    if hotspots_are_demo:
        st.caption("Showing built-in demo hotspots because hotspots.csv was not found.")

    if df_hotspots.empty:
        st.info("No hotspot data found. Add hotspots.csv to the civic data folder.")
    else:
        col_map, col_list = st.columns([1.5, 1])
        with col_map:
            if folium is None or st_folium is None:
                st.error("Map libraries missing. Run: pip install folium streamlit-folium")
            else:
                try:
                    fmap = build_hotspot_map(df_hotspots, (len(df_hotspots), float(df_hotspots["priority_score"].sum())))
                    try:
                        st_folium(fmap, height=450, use_container_width=True, returned_objects=[])
                    except TypeError:
                        st_folium(fmap, height=450, width=700, returned_objects=[])
                except Exception:
                    st.error("The map could not be drawn. Details below.")
                    st.code(traceback.format_exc())
        with col_list:
            st.markdown("##### Top priority locations")
            for idx, row in df_hotspots.head(4).iterrows():
                kind = "critical" if row["priority_score"] >= 80 else "warn"
                score_badge = badge("Score " + str(row["priority_score"]), kind)
                H('<div class="glass"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
                  "<b>#" + str(idx + 1) + " " + str(row["ward"]) + "</b>" + score_badge + "</div>"
                  '<div class="sub">Primary waste: <b>' + str(row["primary_waste"]) + "</b> | Drain risk: <b>" + str(row["drain_risk"]) + "</b></div>"
                  '<div class="sub" style="margin-top:2px">' + str(row["recurrence_count"]) + " repeat reports | " + str(row["status"]) + "</div></div>")

        st.markdown("### Root-cause hypothesis")
        ward_pick = st.selectbox("Choose a hotspot", df_hotspots["ward"].tolist())
        target = df_hotspots[df_hotspots["ward"] == ward_pick].iloc[0].to_dict()
        try:
            rc = generate_root_cause_hypothesis(target)
            rc1, rc2 = st.columns(2)
            with rc1:
                items = "".join("<p>⚠️ " + str(f) + "</p>" for f in rc["contributing_factors"])
                H('<div class="glass"><h4>Contributing factors</h4>' + items + "</div>")
            with rc2:
                H('<div class="glass"><h4>AI hypothesis</h4><p>' + str(rc["hypothesis"]) + "</p>"
                  "<p><b>Recommended action:</b> " + str(rc["recommended_civic_intervention"]) + "</p>"
                  '<div class="sub">' + str(rc["verification_note"]) + "</div></div>")
        except Exception:
            st.error("Root-cause generation failed. Details below.")
            st.code(traceback.format_exc())

# =============================================================================
# PAGE 5: CIVIC IMPACT
# =============================================================================
elif nav_selection == NAV[5]:
    page_header(PAGES[5][1], "Measurable civic outcomes",
                "What changes when unstructured citizen photos become routed, prioritized action.")
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        kpi("Recyclables diverted", "84.2%", "Kept out of landfill")
    with i2:
        kpi("Monthly waste routed", "14.8 T", "Sent to MRF balers")
    with i3:
        kpi("Storm drains protected", 42, "Plastic and textile blockages cleared")
    with i4:
        kpi("Average ward response", 18, "Down from a 72 h baseline", unit="hrs")
    st.caption("Impact figures are illustrative demo values, not measured municipal data.")
    H('<div class="glass" style="text-align:center;padding:34px 22px"><h4 style="font-size:1.4rem">Do not just report waste. Turn every report into intelligence.</h4>'
      '<p style="color:var(--mut);max-width:700px;margin:0 auto">NagrikAI connects passive complaint portals to active municipal operations with risk indices, '
      "hazardous-material containment and data-driven resource recovery.</p></div>")

# =============================================================================
# PAGE 6: ABOUT AI & ARCHITECTURE
# =============================================================================
elif nav_selection == NAV[6]:
    page_header(PAGES[6][1], "How it works, and where it falls short",
                "Pipeline, measured benchmarks and the limitations we disclose up front.")

    tab_arch, tab_bench, tab_honest = st.tabs(["🏗️ Architecture", "📊 Benchmarks", "⚖️ Limitations"])
    with tab_arch:
        layers = [
            ("Citizen capture", "Photo upload from mobile or web (JPG, PNG, WEBP)"),
            ("Preprocessing", "RGB cleanup, resize to 224 x 224, ImageNet normalization"),
            ("Neural classifier", "MobileNetV3-Small transfer learning, 10 waste classes, class-weighted cross-entropy"),
            ("Civic risk engine", "Hazard matrix (SWM Rules 2016), drain-choke index, stakeholder directives"),
            ("Hotspot intelligence", "5-factor priority score, root-cause hypotheses, MRF channeling"),
        ]
        out = ""
        for k, (t, d) in enumerate(layers, 1):
            out += '<div class="layer"><div class="no">' + str(k) + "</div><div><b>" + t + "</b><span>" + d + "</span></div></div>"
            if k < len(layers):
                out += '<div class="link"></div>'
        H(out)
    with tab_bench:
        cmp_path = Path(EVAL_REPORTS_DIR) / "model_comparison.txt"
        if cmp_path.exists():
            st.code(cmp_path.read_text(encoding="utf-8"), language="text")
        else:
            st.info("Benchmark report appears here after running `python -m src.training.compare`.")
        b1, b2 = st.columns(2)
        with b1:
            p = Path(EVAL_REPORTS_DIR) / "confusion_matrix.png"
            if p.exists():
                st.image(str(p), caption="Confusion matrix (test split)", **STRETCH)
        with b2:
            p = Path(EVAL_REPORTS_DIR) / "per_class_f1.png"
            if p.exists():
                st.image(str(p), caption="Per-class F1", **STRETCH)
    with tab_honest:
        notes = [
            ("Classification, not detection",
             "The 19,762-image dataset mostly shows one centered item per photo. Real dumps are mixed, so NagrikAI flags cluttered scenes instead of pretending to localize every object."),
            ("Synthetic civic data",
             "Coordinates, complaint counts and hotspot recurrence are demo fixtures that show how a municipal body would use the output. They are not real municipal records."),
            ("Uncalibrated confidence",
             "We show raw softmax output as model probability and say plainly that it is not calibrated."),
            ("Hypotheses, not findings",
             "Root-cause statements come from heuristics and LLM inference. Each one requires field validation by the Ward Sanitation Inspector."),
        ]
        H("".join('<div class="glass"><h4>' + t + "</h4><p>" + d + "</p></div>" for t, d in notes))

# -----------------------------------------------------------------------------
# End of application (the model loads on demand on the Report Waste page)
# -----------------------------------------------------------------------------