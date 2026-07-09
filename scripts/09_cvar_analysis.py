"""
09_cvar_analysis.py — CVaR analysis of model performance.

Runs 1000 non-repeating 80/20 holdout splits, tracks which episodes
are held out, then analyzes the tail:

  - Loss distribution of episode AUC across splits
  - CVaR at 1%, 5%, 10% (expected AUC in worst splits)
  - "Poison episodes" — which episodes appear most in the worst splits
  - Per-episode impact: how does holding out each episode affect AUC

This tells us WHERE the model fails and WHY.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from config import (PANELS_DIR, RESULTS_DIR, FIGURES_DIR, DEFAULT_SEED,
                    VOL_COLS, FEATURE_SETS)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import runners
spec = importlib.util.spec_from_file_location(
    "tt", Path(__file__).resolve().parent / "07b_timing_test.py")
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)

N_SPLITS = 1000
FEATURE_NAME = "vol_sec"  # best-performing feature set


def run_splits():
    """Run 1000 splits with episode tracking, return DataFrame."""
    cache = RESULTS_DIR / "cvar_splits.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"Loaded cached splits: {len(df)} rows")
        return df

    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    bubble_ids = sorted(panel[panel["source"] == "bubble"]["episode_id"].unique())
    near_ids = sorted(panel[panel["source"] == "near_bubble"]["episode_id"].unique())

    features = FEATURE_SETS[FEATURE_NAME]
    splits = tt.generate_splits(bubble_ids, near_ids, N_SPLITS, DEFAULT_SEED)

    print(f"Running {N_SPLITS} splits with {FEATURE_NAME} ({len(features)} features)...")

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(tt.run_one_split_survival)(panel, features, s, i)
        for i, s in enumerate(splits)
    )
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.to_parquet(cache, index=False)
    print(f"  {len(df)} valid splits saved")
    return df


def cvar_analysis(df):
    """Compute CVaR at multiple thresholds."""
    print(f"\n{'='*60}")
    print("CVaR Analysis of Model Performance")
    print(f"{'='*60}")

    metric = "rsf_episode_auc"
    vals = df[metric].dropna().sort_values()
    n = len(vals)

    print(f"\nDistribution ({metric}, {n} splits):")
    print(f"  Mean:   {vals.mean():.3f}")
    print(f"  Median: {vals.median():.3f}")
    print(f"  Std:    {vals.std():.3f}")
    print(f"  Min:    {vals.min():.3f}")
    print(f"  Max:    {vals.max():.3f}")

    print(f"\nCVaR (expected AUC in worst X% of splits):")
    cvar_results = {}
    for pct in [1, 2, 5, 10, 25]:
        cutoff_idx = max(1, int(n * pct / 100))
        tail = vals.iloc[:cutoff_idx]
        cvar = tail.mean()
        threshold = vals.iloc[cutoff_idx - 1]
        cvar_results[pct] = {"cvar": cvar, "threshold": threshold, "n_splits": cutoff_idx}
        print(f"  CVaR {pct:2d}%: {cvar:.3f} "
              f"(worst {cutoff_idx} splits, threshold={threshold:.3f})")

    return cvar_results


def poison_episode_analysis(df):
    """Find which episodes appear most in the worst splits."""
    print(f"\n{'='*60}")
    print("Poison Episode Analysis")
    print(f"{'='*60}")

    metric = "rsf_episode_auc"
    df = df.dropna(subset=[metric]).copy()
    df = df.sort_values(metric)

    # Parse holdout episode lists
    df["ho_bubble_list"] = df["ho_bubbles"].str.split("|")
    df["ho_near_list"] = df["ho_nears"].str.split("|")

    results = {}
    for pct_label, pct in [("Bottom 5%", 5), ("Bottom 10%", 10), ("Bottom 25%", 25)]:
        cutoff_idx = max(1, int(len(df) * pct / 100))
        worst = df.iloc[:cutoff_idx]
        rest = df.iloc[cutoff_idx:]

        print(f"\n{pct_label} ({cutoff_idx} splits, AUC <= {worst[metric].max():.3f}):")

        # Count bubble appearances in worst splits
        bubble_counts_worst = {}
        bubble_counts_all = {}
        for _, row in df.iterrows():
            for b in row["ho_bubble_list"]:
                if b:
                    bubble_counts_all[b] = bubble_counts_all.get(b, 0) + 1
        for _, row in worst.iterrows():
            for b in row["ho_bubble_list"]:
                if b:
                    bubble_counts_worst[b] = bubble_counts_worst.get(b, 0) + 1

        # Near-bubble appearances
        near_counts_worst = {}
        near_counts_all = {}
        for _, row in df.iterrows():
            for n in row["ho_near_list"]:
                if n:
                    near_counts_all[n] = near_counts_all.get(n, 0) + 1
        for _, row in worst.iterrows():
            for n in row["ho_near_list"]:
                if n:
                    near_counts_worst[n] = near_counts_worst.get(n, 0) + 1

        # Enrichment: how often does episode appear in worst vs expected
        print(f"  Bubble episodes over-represented in worst splits:")
        bubble_enrichment = {}
        for b in bubble_counts_all:
            expected_rate = bubble_counts_all[b] / len(df)
            worst_rate = bubble_counts_worst.get(b, 0) / cutoff_idx
            enrichment = worst_rate / max(expected_rate, 1e-6)
            bubble_enrichment[b] = {
                "worst_count": bubble_counts_worst.get(b, 0),
                "total_count": bubble_counts_all[b],
                "enrichment": enrichment,
            }

        for b, info in sorted(bubble_enrichment.items(),
                               key=lambda x: x[1]["enrichment"], reverse=True)[:8]:
            e = info["enrichment"]
            flag = " <<<" if e > 1.5 else ""
            print(f"    {b:25s}: {info['worst_count']:3d}/{info['total_count']:3d} "
                  f"in worst ({e:.2f}x enrichment){flag}")

        print(f"  Near-bubble episodes over-represented:")
        near_enrichment = {}
        for n in near_counts_all:
            expected_rate = near_counts_all[n] / len(df)
            worst_rate = near_counts_worst.get(n, 0) / cutoff_idx
            enrichment = worst_rate / max(expected_rate, 1e-6)
            near_enrichment[n] = {
                "worst_count": near_counts_worst.get(n, 0),
                "total_count": near_counts_all[n],
                "enrichment": enrichment,
            }

        for n, info in sorted(near_enrichment.items(),
                               key=lambda x: x[1]["enrichment"], reverse=True)[:8]:
            e = info["enrichment"]
            flag = " <<<" if e > 1.5 else ""
            print(f"    {n:25s}: {info['worst_count']:3d}/{info['total_count']:3d} "
                  f"in worst ({e:.2f}x enrichment){flag}")

        results[pct] = {"bubbles": bubble_enrichment, "nears": near_enrichment}

    return results


def per_episode_impact(df):
    """For each episode, compute mean AUC when it's held out vs when it's not."""
    print(f"\n{'='*60}")
    print("Per-Episode Impact on AUC")
    print(f"{'='*60}")

    metric = "rsf_episode_auc"
    df = df.dropna(subset=[metric]).copy()
    df["ho_bubble_list"] = df["ho_bubbles"].str.split("|")
    df["ho_near_list"] = df["ho_nears"].str.split("|")

    all_episodes = set()
    for _, row in df.iterrows():
        all_episodes.update(b for b in row["ho_bubble_list"] if b)
        all_episodes.update(n for n in row["ho_near_list"] if n)

    impacts = []
    for ep in sorted(all_episodes):
        # Splits where this episode is held out
        mask = df.apply(
            lambda r: ep in r["ho_bubble_list"] or ep in r["ho_near_list"], axis=1)
        auc_when_out = df.loc[mask, metric].mean()
        auc_when_in = df.loc[~mask, metric].mean()
        n_out = mask.sum()
        impact = auc_when_out - auc_when_in
        impacts.append({
            "episode": ep,
            "auc_when_held_out": auc_when_out,
            "auc_when_in_training": auc_when_in,
            "impact": impact,  # negative = holding out HURTS (model needs it)
            "n_splits_out": n_out,
        })

    impact_df = pd.DataFrame(impacts).sort_values("impact")

    print(f"\n  Episodes that HURT model when held out (model depends on them):")
    for _, row in impact_df.head(10).iterrows():
        print(f"    {row['episode']:25s}: AUC {row['auc_when_held_out']:.3f} "
              f"(vs {row['auc_when_in_training']:.3f} when in train, "
              f"impact={row['impact']:+.3f})")

    print(f"\n  Episodes that HELP model when held out (model struggles with them):")
    for _, row in impact_df.tail(10).iterrows():
        print(f"    {row['episode']:25s}: AUC {row['auc_when_held_out']:.3f} "
              f"(vs {row['auc_when_in_training']:.3f} when in train, "
              f"impact={row['impact']:+.3f})")

    impact_df.to_parquet(RESULTS_DIR / "episode_impact.parquet", index=False)
    return impact_df


def plot_cvar(df, cvar_results):
    """Plot AUC distribution with CVaR markers."""
    metric = "rsf_episode_auc"
    vals = df[metric].dropna().sort_values().values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram with CVaR markers
    ax = axes[0]
    ax.hist(vals, bins=25, alpha=0.6, color="steelblue", edgecolor="white")
    for pct, info in sorted(cvar_results.items()):
        ax.axvline(info["threshold"], color="red", ls="--", lw=0.8, alpha=0.7)
        ax.text(info["threshold"] - 0.01, ax.get_ylim()[1] * 0.9,
                f"{pct}%", fontsize=7, color="red", ha="right")
    ax.axvline(np.median(vals), color="black", ls="-", lw=1.5,
               label=f"Median={np.median(vals):.3f}")
    ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Episode AUC")
    ax.set_ylabel("Count")
    ax.set_title("AUC Distribution with CVaR Thresholds")
    ax.legend()

    # Right: CVaR curve
    ax = axes[1]
    pcts = sorted(cvar_results.keys())
    cvars = [cvar_results[p]["cvar"] for p in pcts]
    ax.plot(pcts, cvars, "ro-", lw=2, markersize=8)
    ax.set_xlabel("Tail Percentile (%)")
    ax.set_ylabel("CVaR (Expected AUC)")
    ax.set_title("Model CVaR: Expected AUC in Worst Splits")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="Chance")
    ax.legend()
    ax.set_ylim(0.3, 0.9)
    for p, c in zip(pcts, cvars):
        ax.annotate(f"{c:.3f}", (p, c), textcoords="offset points",
                    xytext=(5, 10), fontsize=9)

    fig.suptitle(f"CVaR Analysis ({FEATURE_NAME}, {N_SPLITS} splits)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig10_cvar_analysis.png")
    plt.close()
    print(f"\n  fig10_cvar_analysis.png")


def plot_episode_impact(impact_df):
    """Bar chart of per-episode impact."""
    fig, ax = plt.subplots(figsize=(12, 6))

    impact_df = impact_df.sort_values("impact")
    colors = ["coral" if v < 0 else "steelblue" for v in impact_df["impact"]]

    ax.barh(range(len(impact_df)), impact_df["impact"].values,
            color=colors, alpha=0.7)
    ax.set_yticks(range(len(impact_df)))
    ax.set_yticklabels(impact_df["episode"].values, fontsize=6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Impact on AUC (negative = model needs this episode)")
    ax.set_title("Per-Episode Impact: AUC When Held Out Minus When In Training",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig11_episode_impact.png")
    plt.close()
    print(f"  fig11_episode_impact.png")


def main():
    df = run_splits()
    cvar_results = cvar_analysis(df)
    poison = poison_episode_analysis(df)
    impact_df = per_episode_impact(df)

    plot_cvar(df, cvar_results)
    plot_episode_impact(impact_df)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
