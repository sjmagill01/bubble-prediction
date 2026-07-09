"""
06_align_panel.py — Build aligned panels with PRE-PEAK ONLY training.

Critical fix from v1: no post-peak data in training labels.

Outputs:
  data/panels/train.parquet  — WF-train episodes, tau < 0 for bubbles
  data/panels/test.parquet   — WF-test episodes, tau < 0 for bubbles
  data/panels/target.parquet — ongoing targets (AI, quantum, nuclear2)
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import DATA_DIR, METRICS_DIR, PANELS_DIR, WALK_FORWARD_CUTOFF, PRE_MONTHS, NEAR_PEAK_THRESHOLD
from src.catalog import get_bubbles, get_near_bubbles, get_targets

PANELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = DATA_DIR / "metrics"


def load_and_align():
    """Load metrics, align on event time, enforce pre-peak-only training."""
    bubbles = get_bubbles()
    nears = get_near_bubbles()
    targets = get_targets()

    train_frames = []
    test_frames = []
    target_frames = []

    # ── Bubbles: PRE-PEAK ONLY (tau < 0) ──────────────────────────────
    for b in bubbles:
        bid = b["id"]
        path = METRICS_DIR / f"{bid}.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        peak_ts = pd.Timestamp(b["peak_date"] + "-01")

        # Compute tau (months relative to peak)
        df["tau"] = (
            (df["date"].dt.year - peak_ts.year) * 12
            + df["date"].dt.month - peak_ts.month
        )

        # PRE-PEAK ONLY: tau < 0
        df = df[df["tau"] < 0].copy()

        # Trim to window
        df = df[df["tau"] >= -PRE_MONTHS]

        # Survival labels
        df["event_12m"] = (df["tau"] >= NEAR_PEAK_THRESHOLD).astype(int)
        df["time_to_peak"] = (-df["tau"]).clip(lower=1)

        # Metadata
        df["source"] = "bubble"
        df["category"] = b["category"]
        df["severity"] = b["severity"]
        df["peak_date"] = b["peak_date"]

        # Walk-forward split
        if b["peak_date"] < WALK_FORWARD_CUTOFF:
            train_frames.append(df)
        else:
            test_frames.append(df)

    # ── Near-bubbles: all rows (right-censored) ───────────────────────
    for nb in nears:
        bid = nb["id"]
        path = METRICS_DIR / f"{bid}.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        peak_ts = pd.Timestamp(nb["peak_date"] + "-01")

        df["tau"] = (
            (df["date"].dt.year - peak_ts.year) * 12
            + df["date"].dt.month - peak_ts.month
        )

        # Include pre-peak window (same as bubbles for fair comparison)
        df = df[(df["tau"] >= -PRE_MONTHS) & (df["tau"] < 0)].copy()

        # Right-censored: event never occurs
        df["event_12m"] = 0
        df["time_to_peak"] = 36  # censored at max

        df["source"] = "near_bubble"
        df["category"] = nb["category"]
        df["severity"] = nb.get("max_drawdown", None)
        df["peak_date"] = nb["peak_date"]

        if nb["peak_date"] < WALK_FORWARD_CUTOFF:
            train_frames.append(df)
        else:
            test_frames.append(df)

    # ── Targets: calendar time only ───────────────────────────────────
    for t in targets:
        tid = t["id"]
        path = METRICS_DIR / f"{tid}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["source"] = "target"
        df["category"] = t["category"]
        target_frames.append(df)

    # Stack
    train = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    test = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    target = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()

    return train, test, target


def summary(panel, label):
    if panel.empty:
        print(f"  {label}: EMPTY")
        return

    n_episodes = panel["episode_id"].nunique()
    n_bubbles = panel[panel["source"] == "bubble"]["episode_id"].nunique()
    n_near = panel[panel["source"] == "near_bubble"]["episode_id"].nunique()

    print(f"  {label}:")
    print(f"    Episodes: {n_episodes} ({n_bubbles} bubbles + {n_near} near-bubbles)")
    print(f"    Rows: {len(panel):,}")
    if "tau" in panel.columns:
        print(f"    Tau range: {panel['tau'].min()} to {panel['tau'].max()}")
        n_post = (panel["tau"] >= 0).sum()
        print(f"    Post-peak rows: {n_post} (MUST BE ZERO)")
    if "event_12m" in panel.columns:
        event_rate = panel["event_12m"].mean()
        print(f"    Event rate: {event_rate:.1%}")

    # Metric coverage
    metric_cols = [c for c in panel.columns if c.startswith("metric_")]
    if metric_cols:
        print(f"    Metrics: {len(metric_cols)}")
        for c in sorted(metric_cols):
            pct = panel[c].notna().mean()
            if pct > 0:
                print(f"      {c}: {pct:.0%}")


def main():
    print("Building aligned panels (pre-peak only)...")
    print("=" * 60)

    train, test, target = load_and_align()

    # Save
    train.to_parquet(PANELS_DIR / "train.parquet", index=False)
    test.to_parquet(PANELS_DIR / "test.parquet", index=False)
    if not target.empty:
        target.to_parquet(PANELS_DIR / "target.parquet", index=False)

    print()
    summary(train, "WF-TRAIN (peak < 2015)")
    print()
    summary(test, "WF-TEST (peak >= 2015)")
    print()
    summary(target, "TARGETS (ongoing)")

    # Verification
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    for label, panel in [("train", train), ("test", test)]:
        if "tau" in panel.columns:
            n_post = (panel["tau"] >= 0).sum()
            status = "PASS" if n_post == 0 else f"FAIL ({n_post} post-peak rows!)"
            print(f"  {label} post-peak check: {status}")


if __name__ == "__main__":
    main()
