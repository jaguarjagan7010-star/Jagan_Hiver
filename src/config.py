"""
config.py — Single source of truth for all project settings.

Every script imports from here. To change a setting, change it once here.
Never hardcode paths, seeds, or thresholds in individual scripts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load optional .env file (for API keys, etc.)
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
# ROOT is the hiver-support-agent/ directory, regardless of where you run from.
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_GOLDEN     = ROOT / "data" / "golden"
OUTPUTS         = ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS / "predictions"
EVALUATION_DIR  = OUTPUTS / "evaluation"
LOGS_DIR        = OUTPUTS / "logs"
REPORTS_DIR     = ROOT / "reports"
FIGURES_DIR     = REPORTS_DIR / "figures"
RESULTS_DIR     = REPORTS_DIR / "results"

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Dataset ────────────────────────────────────────────────────────────────────
# The Kaggle dataset may ship as one or more CSV files.
# loader.py will discover the actual filename at runtime.
RAW_DATA_FILENAME = "twcs/twcs.csv"     # actual path inside data/raw/
SAMPLE_SIZE       = 10_000              # default rows to sample (override via CLI)

# ── Brand ──────────────────────────────────────────────────────────────────────
SELECTED_BRAND = "AppleSupport"

# ── Intent classifier ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Data splits ────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # must sum to 1.0

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 5          # top-5 for evidence gate

# ── Escalation / Evidence Gate thresholds ──────────────────────────────────────
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.60
RETRIEVAL_SIMILARITY_THRESHOLD  = 0.30
# Risk tiers: intents that always escalate regardless of confidence
HIGH_RISK_INTENTS = {"account_access", "billing_payment"}  # Apple-specific

# ── LLM generation ─────────────────────────────────────────────────────────────
# Primary: ollama (real open-source LLM). Fallback: template.
LLM_BACKEND    = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_URL     = os.getenv("OLLAMA_URL",   "http://localhost:11434")
HF_MODEL_NAME  = os.getenv("HF_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ── Evaluation ─────────────────────────────────────────────────────────────────
GOLDEN_SET_SIZE  = 200
LLM_JUDGE_SAMPLE = 50
BOOTSTRAP_N      = 1000     # bootstrap iterations for 95% CI
DOUBLE_LABEL_N   = 50       # examples double-labelled for kappa

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
