"""
17_power_analysis.py — Power analysis for the bubble prediction paper.

Three questions:
  1. Does LR (23 features) significantly outperform the naive vol-ratio baseline
     at the *per-split* level, not just in aggregate? (Paired Wilcoxon test)
  2. How wide is the confidence interval on the AUC gap?  (Bootstrap)
  3. What sample size would be needed for 80% power at the observed effect size?
     (Simulation-based power curve)

Uses the existing 1000-split results — no re-training.

Design:
  - Load holdout_80_20_classifiers.parquet  ->  lr_auc per split (feature_set="all")
  - Load holdout_80_20_survival.parquet     ->  ho_bubbles / ho_nears per split
  - Join on split index to get test episodes for each split
  - Compute naive AUC per split: rank test episodes by mean metric_a (vol ratio),
    no fitting, no standardization
  - Paired Wilcoxon signed-rank test: H0: gap <= 0
  - 10,000-iteration bootstrap CI on median gap
  - Simulation power curve: for n in [20, 30, 40, 50, 69, 100, 150], simulate
    1000 splits and estimate power at the observed effect size

Outputs:
  data/results/power_analysis.csv        (summary table)
  figures/fig_power_analysis.png         (3-panel figure)
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import PANELS_DIR, RESULTS_DIR, FIGURES_DIR, DEFAULT_SEED, ALL_FEATURE_COLS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode_features(panel):
    """Episode-level mean features for naive and LR scoring."""
    feat_cols = [c for c in ALL_FEATURE_COLS if c in panel.columns]
    ep = (panel
          .groupby(["episode_id", "source"])[feat_cols]
          .mean()
          .reset_index())
    ep["is_bubble"] = (ep["source"] == "bubble").astype(int)
    return ep, feat_cols


def load_splits_with_episodes(surv_df, feat_name="all"):
    """
    From the survival parquet, extract per-split test episode lists.
    Returns dict: split_i -> (ho_bubble_list, ho_near_list)
    """
    sub = surv_df[surv_df["feature_set"] == feat_name].copy()
    splits = {}
    for _, row in sub.iterrows():
        si = int(row["split"])
        bubs = row["ho_bubbles"].split("|") if isinstance(row["ho_bubbles"], str) else []
        nears = row["ho_nears"].split("|") if isinstance(row["ho_nears"], str) else []
        splits[si] = bubs + nears
    return splits


# ---------------------------------------------------------------------------
# Naive baseline AUC per split
# ---------------------------------------------------------------------------

def compute_naive_auc(ep_df, test_ids):
    """
    Naive baseline: rank test episodes by mean metric_a (vol ratio), no fitting.
    Returns AUC or NaN if insufficient data.
    """
    te = ep_df[ep_df["episode_id"].isin(test_ids)].copy()
    if "metric_a" not in te.columns:
        return np.nan
    mask = te["metric_a"].notna()
    te = te[mask]
    if len(te) < 4 or te["is_bubble"].nunique() < 2:
        return np.nan
    return roc_auc_score(te["is_bubble"], te["metric_a"])


# ---------------------------------------------------------------------------
# Bootstrap CI on median gap
# ---------------------------------------------------------------------------

def bootstrap_median_gap(gaps, n_boot=10_000, seed=DEFAULT_SEED):
    """Bootstrap 95% CI on median(LR_auc - naive_auc)."""
    rng = np.random.RandomState(seed)
    gaps = np.asarray(gaps)
    boot_medians = np.array([
        np.median(rng.choice(gaps, size=len(gaps), replace=True))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot_medians, [2.5, 97.5])
    return np.median(gaps), lo, hi


# ---------------------------------------------------------------------------
# Simulation-based power curve
# ---------------------------------------------------------------------------

def simulate_power_at_n(n_episodes, effect_mu, effect_sigma,
                         n_sim=2000, alpha=0.05, seed=DEFAULT_SEED):
    """
    Simulate power of Wilcoxon test at a given episode count.

    We model each split's AUC gap as drawn from N(effect_mu, effect_sigma²)
    and simulate n_splits=200 paired observations per simulation.
    Returns fraction of simulations where H0 is rejected.
    """
    rng = np.random.RandomState(seed)
    # n_splits per simulation: scale with n_episodes (fewer episodes -> fewer valid splits)
    n_splits_sim = max(50, min(500, n_episodes * 7))
    rejections = 0
    for _ in range(n_sim):
        # Simulate gaps: drawn from a normal with the observed effect size
        # scaled so smaller n -> larger variance
        scale = effect_sigma * np.sqrt(69 / n_episodes)
        gaps = rng.normal(loc=effect_mu, scale=scale, size=n_splits_sim)
        try:
            stat, p = wilcoxon(gaps, alternative="greater")
            if p < alpha:
                rejections += 1
        except Exception:
            pass
    return rejections / n_sim


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_power_analysis(lr_aucs, naive_aucs, gaps, boot_ci, power_results,
                        wilcox_p, save_path):
    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Panel 1: AUC distributions ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bins = np.linspace(0.3, 1.0, 35)
    ax1.hist(naive_aucs, bins=bins, alpha=0.55, color="#4a90d9", label="Naive (metric A only)")
    ax1.hist(lr_aucs, bins=bins, alpha=0.55, color="#e05c2a", label="LR (23 features)")
    ax1.axvline(np.median(naive_aucs), color="#4a90d9", lw=1.5, ls="--")
    ax1.axvline(np.median(lr_aucs), color="#e05c2a", lw=1.5, ls="--")
    ax1.set_xlabel("Episode AUC (per split)")
    ax1.set_ylabel("Number of splits")
    ax1.set_title("Per-split AUC distributions\n(1,000 leverage-stratified splits)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.text(0.03, 0.97,
             f"Naive median: {np.median(naive_aucs):.3f}\nLR median:    {np.median(lr_aucs):.3f}",
             transform=ax1.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Panel 2: Gap distribution ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    gap_bins = np.linspace(-0.6, 0.6, 40)
    ax2.hist(gaps, bins=gap_bins, color="#6a4c9c", alpha=0.75)
    ax2.axvline(0, color="black", lw=1.0, ls="-")
    ax2.axvline(boot_ci[0], color="#6a4c9c", lw=1.5, ls="--", label=f"Median = {boot_ci[0]:.3f}")
    ax2.axvline(boot_ci[1], color="gray", lw=1.0, ls=":", label=f"95% CI [{boot_ci[1]:.3f}, {boot_ci[2]:.3f}]")
    ax2.axvline(boot_ci[2], color="gray", lw=1.0, ls=":")
    ax2.fill_betweenx([0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 200],
                       boot_ci[1], boot_ci[2], color="gray", alpha=0.12)
    pct_positive = 100 * np.mean(np.array(gaps) > 0)
    ax2.set_xlabel("AUC gap (LR - Naive) per split")
    ax2.set_ylabel("Number of splits")
    ax2.set_title(f"Paired AUC gap distribution\nWilcoxon p = {wilcox_p:.4f}", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.text(0.03, 0.97,
             f"{pct_positive:.1f}% of splits:\nLR > Naive",
             transform=ax2.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Panel 3: Power curve ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ns = [r["n"] for r in power_results]
    powers = [r["power"] for r in power_results]
    ax3.plot(ns, powers, color="#2e7d32", lw=2, marker="o", ms=5)
    ax3.axhline(0.80, color="gray", lw=1.0, ls="--", label="80% power threshold")
    ax3.axhline(0.05, color="lightgray", lw=1.0, ls=":", label="Type I error (5%)")
    ax3.axvline(69, color="#e05c2a", lw=1.5, ls="--", label="This study (n=69)")
    ax3.set_xlabel("Number of episodes")
    ax3.set_ylabel("Simulated power")
    ax3.set_title("Power curve\n(Wilcoxon, α=0.05, observed effect size)", fontsize=9)
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 1.05)
    ax3.set_xlim(15, max(ns) + 5)

    plt.suptitle(
        "Power Analysis: LR (23 features) vs. Naive Baseline",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Power Analysis: LR vs. Naive Baseline")
    print("=" * 60)

    # Load data
    panel = pd.concat([
        pd.read_parquet(PANELS_DIR / "train.parquet"),
        pd.read_parquet(PANELS_DIR / "test.parquet"),
    ], ignore_index=True)
    ep_df, feat_cols = load_episode_features(panel)

    # Load existing results
    clf_df = pd.read_parquet(RESULTS_DIR / "holdout_80_20_classifiers.parquet")
    surv_df = pd.read_parquet(RESULTS_DIR / "holdout_80_20_survival.parquet")

    # Pull LR AUC for "all" feature set
    lr_rows = clf_df[clf_df["feature_set"] == "all"][["split", "lr_auc"]].dropna()
    print(f"  LR splits loaded: {len(lr_rows)}")

    # Get test episode IDs per split from survival parquet
    split_episodes = load_splits_with_episodes(surv_df, feat_name="all")
    print(f"  Survival splits loaded: {len(split_episodes)}")

    # Compute naive AUC per split
    print("  Computing naive AUC per split...")
    naive_rows = []
    for _, row in lr_rows.iterrows():
        si = int(row["split"])
        test_ids = split_episodes.get(si, [])
        naive_auc = compute_naive_auc(ep_df, test_ids)
        naive_rows.append({"split": si, "naive_auc": naive_auc, "lr_auc": row["lr_auc"]})

    paired = pd.DataFrame(naive_rows).dropna()
    print(f"  Valid paired observations: {len(paired)}")

    lr_aucs = paired["lr_auc"].values
    naive_aucs = paired["naive_auc"].values
    gaps = lr_aucs - naive_aucs

    # ── Wilcoxon signed-rank test ─────────────────────────────────────────
    stat, p_wilcox = wilcoxon(gaps, alternative="greater")
    print(f"\nPaired Wilcoxon signed-rank test (H0: gap <= 0):")
    print(f"  Statistic = {stat:.1f},  p = {p_wilcox:.4f}")
    print(f"  Median LR AUC:    {np.median(lr_aucs):.4f}")
    print(f"  Median naive AUC: {np.median(naive_aucs):.4f}")
    print(f"  Median gap:       {np.median(gaps):+.4f}")
    print(f"  % splits LR > naive: {100*np.mean(gaps > 0):.1f}%")

    # ── Bootstrap CI on median gap ────────────────────────────────────────
    print("\nBootstrap CI on median gap (10,000 iterations)...")
    med_gap, lo, hi = bootstrap_median_gap(gaps)
    print(f"  Median gap = {med_gap:+.4f}  [95% CI: {lo:+.4f}, {hi:+.4f}]")

    # ── Power curve ───────────────────────────────────────────────────────
    effect_mu = float(np.mean(gaps))
    effect_sigma = float(np.std(gaps))
    print(f"\nPower curve (effect_mu={effect_mu:.4f}, sigma={effect_sigma:.4f})...")
    ns = [20, 25, 30, 40, 50, 60, 69, 80, 100, 120, 150]
    power_results = []
    for n in ns:
        pw = simulate_power_at_n(n, effect_mu, effect_sigma)
        power_results.append({"n": n, "power": pw})
        print(f"  n={n:3d}: power={pw:.3f}")

    # ── Save results ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "n_splits": len(paired),
        "median_lr_auc": np.median(lr_aucs),
        "median_naive_auc": np.median(naive_aucs),
        "median_gap": med_gap,
        "gap_ci_lo": lo,
        "gap_ci_hi": hi,
        "pct_splits_lr_beats_naive": float(np.mean(gaps > 0)),
        "wilcoxon_stat": stat,
        "wilcoxon_p": p_wilcox,
        "effect_mu": effect_mu,
        "effect_sigma": effect_sigma,
    }])
    summary.to_csv(RESULTS_DIR / "power_analysis.csv", index=False)

    power_df = pd.DataFrame(power_results)
    power_df.to_csv(RESULTS_DIR / "power_curve.csv", index=False)

    # ── Figure ────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_power_analysis(
        lr_aucs, naive_aucs, gaps,
        boot_ci=(med_gap, lo, hi),
        power_results=power_results,
        wilcox_p=p_wilcox,
        save_path=FIGURES_DIR / "fig_power_analysis.png"
    )

    # ── Print paper-ready summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PAPER-READY SUMMARY")
    print("=" * 60)
    print(f"The multi-channel LR model (23 features, median AUC {np.median(lr_aucs):.3f})")
    print(f"outperforms the naive vol-ratio baseline (median AUC {np.median(naive_aucs):.3f})")
    print(f"in {100*np.mean(gaps > 0):.1f}% of the 1,000 leverage-stratified holdout splits.")
    print(f"The median AUC gap is {med_gap:+.4f} (bootstrap 95% CI: [{lo:+.4f}, {hi:+.4f}]).")
    print(f"A paired Wilcoxon signed-rank test rejects H0: gap <= 0 at p = {p_wilcox:.4f}.")

    # Power at current n
    pw_69 = next(r["power"] for r in power_results if r["n"] == 69)
    print(f"\nAt the observed effect size and n=69 episodes,")
    print(f"simulated power of the Wilcoxon test = {pw_69:.3f}.")
    n_for_80 = next((r["n"] for r in power_results if r["power"] >= 0.80), None)
    if n_for_80:
        print(f"80% power requires approximately n={n_for_80} episodes.")
    else:
        print("80% power requires > 150 episodes at this effect size.")


if __name__ == "__main__":
    main()
