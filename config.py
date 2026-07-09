"""
config.py — Central configuration for bubble prediction project.
"""
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FIRMS_DIR = DATA_DIR / "firms"
RAW_DIR = DATA_DIR / "raw"
SEC_DIR = DATA_DIR / "sec"
MEASURES_DIR = DATA_DIR / "measures"
METRICS_DIR = DATA_DIR / "metrics"
PANELS_DIR = DATA_DIR / "panels"
RESULTS_DIR = DATA_DIR / "results"
CIV_DIR = DATA_DIR / "civ"
FIGURES_DIR = PROJECT_ROOT / "figures"
MULTICHANNEL_DIR = DATA_DIR / "multichannel"

# Walk-forward split
WALK_FORWARD_CUTOFF = "2015-01"  # train on peak < 2015, test on peak >= 2015

# Survival model defaults
DEFAULT_SEED = 42
PRE_MONTHS = 36
NEAR_PEAK_THRESHOLD = -12  # tau >= -12 is "near peak"
COX_PENALIZER = 0.1
MERTON_DEBT_MATURITY = 5.0  # standard CDS/bond tenor for Merton inversion

# Feature column definitions (single source of truth)
VOL_COLS = ["metric_a", "metric_b", "metric_c", "metric_d", "metric_e", "metric_f"]
SEC_COLS = ["metric_g", "metric_h", "metric_i", "metric_j", "metric_k",
            "metric_l", "metric_m", "metric_n"]
BOND_CIV_COLS = ["metric_o", "metric_p", "metric_q"]
TEXT_CIV_COLS = ["metric_r", "metric_s"]
INTERACTION_COLS = ["metric_u", "metric_v", "metric_w", "metric_x"]
# U: leverage level, V: fragility vol (A*lev), W: mania vol (A*(1-lev)), X: fragility instability (D*lev)

ALL_FEATURE_COLS = VOL_COLS + SEC_COLS + BOND_CIV_COLS + TEXT_CIV_COLS + INTERACTION_COLS

FEATURE_SETS = {
    "vol_only": VOL_COLS,
    "vol_interactions": VOL_COLS + INTERACTION_COLS,
    "vol_sec": VOL_COLS + SEC_COLS,
    "vol_sec_interactions": VOL_COLS + SEC_COLS + INTERACTION_COLS,
    "all": VOL_COLS + SEC_COLS + BOND_CIV_COLS + TEXT_CIV_COLS + INTERACTION_COLS,
}

# SEC EDGAR
SEC_USER_AGENT = "Samuel Magill smagill@gradcenter.cuny.edu"
SEC_RATE_LIMIT = 0.11  # seconds between EDGAR requests

# Vol computation
VOL_WINDOW = 63  # trading days for realized vol
MIN_DAYS_PER_MONTH = 15
VOL_OF_VOL_WINDOW = 6  # months
ANNUALIZE = 252 ** 0.5
CORR_WINDOW = 63
MIN_FIRMS_CORR = 3
MIN_DAYS_CORR = 40
