"""
15_multichannel_figures.py — Side-by-side equity, CIV, and SEC case-study figures.

Reads sector-level derived metrics from data/multichannel/ and writes to figures/.

Produces:
  fig7_multichannel_cases.png — Case study panels (banks, homebuilders, shale, commodities)
  fig9_lead_lag.png           — Cross-correlation lead-lag analysis between channels
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data" / "multichannel"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ── Data loading ──────────────────────────────────────────────────────

def load_all():
    """Load vol metrics, SEC metrics, and CIV metrics."""
    vol = {}
    sec = {}
    civ = {}

    # Vol metrics (per bubble)
    metrics_dir = DATA_DIR / "metrics"
    for f in metrics_dir.glob("*.parquet"):
        vol[f.stem] = pd.read_parquet(f)

    # SEC metrics
    sec_path = DATA_DIR / "sec_metrics" / "_all_sec_metrics.parquet"
    if sec_path.exists():
        sec_all = pd.read_parquet(sec_path)
        for bid in sec_all.bubble_id.unique():
            sec[bid] = sec_all[sec_all.bubble_id == bid].copy()

    # CIV metrics
    civ_path = DATA_DIR / "civ_metrics" / "_all_civ_metrics.parquet"
    if civ_path.exists():
        civ_all = pd.read_parquet(civ_path)
        for bid in civ_all.bubble_id.unique():
            civ[bid] = civ_all[civ_all.bubble_id == bid].copy()

    return vol, sec, civ


def _align_on_peak(df, peak_date):
    """Add tau column (months relative to peak)."""
    if peak_date is None:
        return df
    peak_ts = pd.Timestamp(peak_date + "-01")
    df = df.copy()
    df["tau"] = (
        (df["date"].dt.year - peak_ts.year) * 12
        + df["date"].dt.month - peak_ts.month
    )
    return df


# ── Figure 7: Case study multi-channel panels ────────────────────────

CASE_STUDIES = [
    ("banks_gfc", "Banks / GFC", "2007-06"),
    ("homebuilders", "Homebuilders", "2005-07"),
    ("shale", "Shale Oil", "2014-06"),
    ("commodity_super", "Commodity Supercycle", "2008-06"),
]


def fig7_multichannel_cases(vol, sec, civ):
    """
    For each case study: 3-row panel showing
    Row 1: Equity vol (sector vol + market vol) and correlation
    Row 2: CIV and CIV-equity wedge
    Row 3: SEC sentiment (negativity ratio) and risk escalation language
    """
    cases = [c for c in CASE_STUDIES if c[0] in vol and c[0] in civ]
    n_cases = len(cases)
    if n_cases == 0:
        print("  No cases with all three channels")
        return

    fig, axes = plt.subplots(3, n_cases, figsize=(4.5 * n_cases, 9), squeeze=False)

    for col_i, (bid, label, peak) in enumerate(cases):
        v = _align_on_peak(vol[bid], peak)
        c = _align_on_peak(civ[bid], peak)

        # Trim to event window
        v = v[(v["tau"] >= -36) & (v["tau"] <= 24)]
        c = c[(c["tau"] >= -36) & (c["tau"] <= 24)]

        # Row 1: Equity vol + correlation
        ax1 = axes[0, col_i]
        if "sector_vol" in v.columns:
            ax1.plot(v["tau"], v["sector_vol"] * 100, "b-", lw=1.5,
                     label="Sector vol", alpha=0.8)
        if "market_vol" in v.columns:
            ax1.plot(v["tau"], v["market_vol"] * 100, "k--", lw=1, alpha=0.5,
                     label="Market vol")
        ax1.axvline(0, color="red", lw=0.8, ls="--", alpha=0.7)
        ax1.set_ylabel("Annualized vol (%)")
        ax1.set_title(label, fontweight="bold")
        if "metric_c" in v.columns:
            ax1b = ax1.twinx()
            ax1b.plot(v["tau"], v["metric_c"], "g-", lw=1, alpha=0.6,
                      label="Correlation")
            ax1b.set_ylabel("Pairwise corr", color="g", fontsize=8)
            ax1b.tick_params(axis="y", labelcolor="g", labelsize=7)
        if col_i == 0:
            ax1.legend(loc="upper left", fontsize=7)

        # Row 2: CIV and wedge
        ax2 = axes[1, col_i]
        ax2.plot(c["tau"], c["metric_o"] * 100, "r-", lw=1.5,
                 label="CDS-implied vol (CIV)", alpha=0.8)
        if "equity_asset_vol" in c.columns:
            ev = c.dropna(subset=["equity_asset_vol"])
            ax2.plot(ev["tau"], ev["equity_asset_vol"] * 100, "b--", lw=1,
                     alpha=0.6, label="Equity asset vol")
        ax2.axvline(0, color="red", lw=0.8, ls="--", alpha=0.7)
        ax2.set_ylabel("Asset vol (%)")
        if "metric_p" in c.columns:
            cp = c.dropna(subset=["metric_p"])
            if not cp.empty:
                ax2b = ax2.twinx()
                ax2b.fill_between(cp["tau"], cp["metric_p"] * 100, 0,
                                  alpha=0.15, color="purple")
                ax2b.set_ylabel("Wedge (pp)", color="purple", fontsize=8)
                ax2b.tick_params(axis="y", labelcolor="purple", labelsize=7)
        if col_i == 0:
            ax2.legend(loc="upper left", fontsize=7)

        # Row 3: SEC metrics
        ax3 = axes[2, col_i]
        if bid in sec:
            s = _align_on_peak(sec[bid], peak)
            s = s[(s["tau"] >= -36) & (s["tau"] <= 24)]
            if "metric_n" in s.columns:
                sn = s.dropna(subset=["metric_n"])
                ax3.plot(sn["tau"], sn["metric_n"], "darkorange", lw=1.5,
                         label="Negativity ratio", alpha=0.8)
            if "metric_k" in s.columns:
                sk = s.dropna(subset=["metric_k"])
                if not sk.empty:
                    ax3b = ax3.twinx()
                    ax3b.plot(sk["tau"], sk["metric_k"], "brown", lw=1,
                              alpha=0.6, label="Risk escalation")
                    ax3b.set_ylabel("Risk per 10k", color="brown", fontsize=8)
                    ax3b.tick_params(axis="y", labelcolor="brown", labelsize=7)
        else:
            ax3.text(0.5, 0.5, "No SEC data", transform=ax3.transAxes,
                     ha="center", va="center", fontsize=10, color="gray")
        ax3.axvline(0, color="red", lw=0.8, ls="--", alpha=0.7)
        ax3.set_xlabel("Months relative to peak")
        ax3.set_ylabel("Negativity ratio")
        if col_i == 0 and bid in sec:
            ax3.legend(loc="upper left", fontsize=7)

    # Row labels on left
    for row_i, label in enumerate(["Equity Volatility", "CDS-Implied Volatility",
                                    "SEC Filing Language"]):
        axes[row_i, 0].annotate(label, xy=(-0.35, 0.5),
                                xycoords="axes fraction", fontsize=10,
                                fontweight="bold", rotation=90, va="center")

    fig.suptitle("Multi-Channel Bubble Anatomy: Equity, Credit, and Text Signals",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_multichannel_cases.png")
    plt.close()
    print(f"  Saved fig7_multichannel_cases.png ({n_cases} cases)")


# ── Figure 9: Lead-lag cross-correlation ──────────────────────────────

def fig9_lead_lag(vol, sec, civ):
    """
    Cross-correlation between equity vol, CIV, and SEC sentiment at
    different lags. Which signal moves first?
    """
    from src.catalog import get_bubbles as get_catalog

    # Collect paired monthly series for bubbles with all three channels
    pairs = []
    for bubble in get_catalog():
        bid = bubble["id"]
        if bid not in vol or bid not in civ or bid not in sec:
            continue
        peak = bubble["peak_date"]
        if peak is None:
            continue

        v = _align_on_peak(vol[bid], peak)
        c = _align_on_peak(civ[bid], peak)
        s = _align_on_peak(sec[bid], peak)

        # Merge on tau
        merged = v[["tau", "metric_a", "sector_vol"]].merge(
            c[["tau", "metric_o"]], on="tau", how="inner")
        merged = merged.merge(s[["tau", "metric_n", "metric_k"]], on="tau", how="inner")
        if len(merged) >= 12:
            pairs.append(merged)

    if not pairs:
        print("  No bubbles with all three channels for lead-lag")
        return

    all_pairs = pd.concat(pairs, ignore_index=True)

    # Compute cross-correlation at different lags
    max_lag = 12
    lags = range(-max_lag, max_lag + 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    channel_pairs = [
        ("metric_a", "metric_o", "Equity Vol Ratio vs CIV", "b"),
        ("metric_a", "metric_n", "Equity Vol Ratio vs SEC Negativity", "darkorange"),
        ("metric_o", "metric_n", "CIV vs SEC Negativity", "purple"),
    ]

    for ax, (col_x, col_y, title, color) in zip(axes, channel_pairs):
        corrs = []
        for lag in lags:
            shifted = all_pairs.copy()
            shifted[col_y] = shifted[col_y].shift(lag)
            valid = shifted.dropna(subset=[col_x, col_y])
            if len(valid) >= 10:
                corrs.append(valid[col_x].corr(valid[col_y]))
            else:
                corrs.append(np.nan)

        ax.bar(list(lags), corrs, color=color, alpha=0.6, width=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="red", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Lag (months, positive = Y leads)")
        ax.set_ylabel("Correlation")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-max_lag - 0.5, max_lag + 0.5)

    fig.suptitle("Lead-Lag Cross-Correlations Between Information Channels",
                 fontsize=11, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_lead_lag.png")
    plt.close()
    print(f"  Saved fig9_lead_lag.png ({len(pairs)} bubbles)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    vol, sec, civ = load_all()
    print(f"  Vol: {len(vol)} bubbles, SEC: {len(sec)}, CIV: {len(civ)}")

    print("\nGenerating figures...")
    fig7_multichannel_cases(vol, sec, civ)
    fig9_lead_lag(vol, sec, civ)

    print("\nDone.")


if __name__ == "__main__":
    main()
