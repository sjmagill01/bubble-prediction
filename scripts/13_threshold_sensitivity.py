"""
13_threshold_sensitivity.py — Drawdown threshold sensitivity analysis.

Tests whether the headline AUC (0.844) is robust to the choice of 40%
drawdown threshold that separates bubbles from near-bubbles.

Sweeps thresholds [30, 35, 38, 40, 42, 45, 50], relabels episodes,
and re-runs the LR episode classifier at each threshold.

Output:
  data/results/robustness/threshold_sensitivity.csv
  figures/fig_robustness_threshold.png
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from config import (PANELS_DIR, RESULTS_DIR, FIGURES_DIR, DEFAULT_SEED,
                    ALL_FEATURE_COLS, VOL_COLS, SEC_COLS, BOND_CIV_COLS,
                    TEXT_CIV_COLS, INTERACTION_COLS)
from src.catalog import get_bubbles, get_near_bubbles

ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [30, 35, 38, 40, 42, 45, 50]
N_SPLITS = 200
FEATURE_COLS = VOL_COLS + SEC_COLS + BOND_CIV_COLS + TEXT_CIV_COLS + INTERACTION_COLS


def relabel_episodes(threshold):
    """
    Relabel episodes based on drawdown threshold.
    Bubbles with severity < threshold become near-bubbles.
    Near-bubbles with max_drawdown >= threshold become bubbles.
    Returns (bubble_ids, near_ids, n_flipped_to_near, n_flipped_to_bubble).
    """
    bubbles = get_bubbles()
    nears = get_near_bubbles()

    bubble_ids = []
    near_ids = []
    flipped_to_near = 0
    flipped_to_bubble = 0

    for b in bubbles:
        if b["severity"] >= threshold:
            bubble_ids.append(b["id"])
        else:
            near_ids.append(b["id"])
            flipped_to_near += 1

    for n in nears:
        if n["max_drawdown"] >= threshold:
            bubble_ids.append(n["id"])
            flipped_to_bubble += 1
        else:
            near_ids.append(n["id"])

    return bubble_ids, near_ids, flipped_to_near, flipped_to_bubble


def run_at_threshold(ep_features, threshold, n_splits=N_SPLITS):
    """Run LR classifier with relabeled episodes at given threshold."""
    bubble_ids, near_ids, fl_near, fl_bub = relabel_episodes(threshold)

    # Update labels in ep_features
    df = ep_features.copy()
    df["is_bubble"] = df["episode_id"].isin(bubble_ids).astype(int)

    # Only keep episodes that are in bubble_ids or near_ids
    valid_ids = set(bubble_ids) | set(near_ids)
    df = df[df["episode_id"].isin(valid_ids)].reset_index(drop=True)

    bub_in_data = df[df["is_bubble"] == 1]["episode_id"].unique()
    near_in_data = df[df["is_bubble"] == 0]["episode_id"].unique()

    if len(bub_in_data) < 3 or len(near_in_data) < 3:
        return None

    n_ho_bub = max(1, round(len(bub_in_data) * 0.2))
    n_ho_near = max(1, round(len(near_in_data) * 0.2))

    rng = np.random.RandomState(DEFAULT_SEED)
    results = []
    seen = set()

    while len(results) < n_splits:
        ho_b = tuple(sorted(rng.choice(bub_in_data, size=n_ho_bub, replace=False)))
        ho_n = tuple(sorted(rng.choice(near_in_data, size=n_ho_near, replace=False)))
        key = (ho_b, ho_n)
        if key in seen:
            continue
        seen.add(key)

        ho_all = set(ho_b) | set(ho_n)
        tr = df[~df["episode_id"].isin(ho_all)]
        te = df[df["episode_id"].isin(ho_all)]

        # Use only features present in data
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        X_tr = tr[feat_cols].fillna(0).values
        y_tr = tr["is_bubble"].values
        X_te = te[feat_cols].fillna(0).values
        y_te = te["is_bubble"].values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        try:
            lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
            lr.fit(X_tr_s, y_tr)
            prob = lr.predict_proba(X_te_s)[:, 1]
            auc = roc_auc_score(y_te, prob)
            results.append(auc)
        except Exception:
            continue

    return results


def main():
    print("Threshold Sensitivity Analysis")
    print("=" * 60)

    # Load panel and collapse to episode-level
    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
    ep_features = (
        panel.groupby(["episode_id", "source"])[feat_cols]
        .mean()
        .reset_index()
    )

    rows = []
    for threshold in THRESHOLDS:
        bubble_ids, near_ids, fl_near, fl_bub = relabel_episodes(threshold)
        n_bub = sum(1 for eid in bubble_ids if eid in ep_features["episode_id"].values)
        n_near = sum(1 for eid in near_ids if eid in ep_features["episode_id"].values)

        print(f"\n  Threshold {threshold}%: {n_bub} bubbles, {n_near} near-bubbles "
              f"(+{fl_bub} promoted, -{fl_near} demoted)")

        aucs = run_at_threshold(ep_features, threshold, N_SPLITS)
        if aucs is None:
            print(f"    SKIPPED (too few episodes)")
            continue

        aucs = np.array(aucs)
        med = np.median(aucs)
        ci_lo = np.percentile(aucs, 2.5)
        ci_hi = np.percentile(aucs, 97.5)

        print(f"    LR AUC: {med:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

        rows.append({
            "threshold": threshold,
            "n_bubbles": n_bub,
            "n_near": n_near,
            "flipped_to_near": fl_near,
            "flipped_to_bubble": fl_bub,
            "median_auc": med,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "mean_auc": aucs.mean(),
            "std_auc": aucs.std(),
            "n_splits": len(aucs),
        })

    results = pd.DataFrame(rows)
    results.to_csv(ROBUSTNESS_DIR / "threshold_sensitivity.csv", index=False)
    print(f"\nSaved: {ROBUSTNESS_DIR / 'threshold_sensitivity.csv'}")

    # ── Figure ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(results["threshold"], results["median_auc"], "o-", color="steelblue",
                 linewidth=2, markersize=8, label="Median AUC")
        ax1.fill_between(results["threshold"], results["ci_low"], results["ci_high"],
                         alpha=0.2, color="steelblue")
        ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
        ax1.axvline(40, color="red", linestyle=":", alpha=0.6, label="Baseline (40%)")
        ax1.set_xlabel("Drawdown Threshold (%)", fontsize=12)
        ax1.set_ylabel("Episode-Level AUC", fontsize=12)
        ax1.set_ylim(0.4, 1.0)
        ax1.legend(loc="lower left", fontsize=10)

        # Secondary axis: episode counts
        ax2 = ax1.twinx()
        ax2.bar(results["threshold"] - 0.5, results["n_bubbles"], width=1.0,
                alpha=0.15, color="red", label="Bubbles")
        ax2.bar(results["threshold"] + 0.5, results["n_near"], width=1.0,
                alpha=0.15, color="green", label="Near-bubbles")
        ax2.set_ylabel("Episode Count", fontsize=12, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.legend(loc="upper right", fontsize=9)

        ax1.set_title("Sensitivity to Drawdown Threshold", fontsize=14)
        plt.tight_layout()

        fig_path = FIGURES_DIR / "fig_robustness_threshold.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")
    except ImportError:
        print("matplotlib not available — skipped figure")


if __name__ == "__main__":
    main()
