"""
14_paper_figures.py — Generate the paper's core result figures.

Reads from data/results/, data/civ/ and writes PNGs to figures/.

Outputs:
  fig_episode_auc_dist.png   — episode AUC distributions (RSF/LR, holdout splits)
  fig_cox_coefficients.png   — Cox PH coefficients with 95% CIs
  fig_cvar_and_impact.png    — CVaR tail analysis + per-episode AUC impact
  fig_text_vs_bond_civ.png   — text-CIV vs bond-CIV validation scatter
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_ROOT = Path(__file__).resolve().parent.parent
V2_RESULTS = V2_ROOT / "data" / "results"
V2_METRICS = V2_ROOT / "data" / "metrics"
V2_CIV = V2_ROOT / "data" / "civ"
V2_PANELS = V2_ROOT / "data" / "panels"
FIGURES_DIR = V2_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})


def fig_episode_auc_dist():
    """Distribution of episode-level AUC across holdout splits."""
    surv = pd.read_parquet(V2_RESULTS / "holdout_80_20_survival.parquet")
    cls = pd.read_parquet(V2_RESULTS / "holdout_80_20_classifiers.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # RSF vol_only
    ax = axes[0]
    vals = surv[(surv.feature_set == "vol_only")]["rsf_episode_auc"].dropna()
    ax.hist(vals, bins=20, alpha=0.6, color="steelblue", edgecolor="white")
    ax.axvline(vals.median(), color="black", ls="-", lw=1.5,
               label=f"Median={vals.median():.3f}")
    ax.axvline(0.780, color="red", ls="--", lw=1.2, label="Baseline (A)=0.780")
    ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Episode AUC")
    ax.set_ylabel("Count")
    ax.set_title("RSF vol-only (A–F)")
    ax.legend(fontsize=7)
    ax.set_xlim(0.3, 1.05)

    # RSF all
    ax = axes[1]
    vals = surv[(surv.feature_set == "all")]["rsf_episode_auc"].dropna()
    ax.hist(vals, bins=20, alpha=0.6, color="purple", edgecolor="white")
    ax.axvline(vals.median(), color="black", ls="-", lw=1.5,
               label=f"Median={vals.median():.3f}")
    ax.axvline(0.780, color="red", ls="--", lw=1.2, label="Baseline (A)=0.780")
    ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Episode AUC")
    ax.set_title("RSF all features (A–S, U–X)")
    ax.legend(fontsize=7)
    ax.set_xlim(0.3, 1.05)

    # LR all
    ax = axes[2]
    vals = cls[(cls.feature_set == "all")]["lr_auc"].dropna()
    ax.hist(vals, bins=20, alpha=0.6, color="darkorange", edgecolor="white")
    ax.axvline(vals.median(), color="black", ls="-", lw=1.5,
               label=f"Median={vals.median():.3f}")
    ax.axvline(0.780, color="red", ls="--", lw=1.2, label="Baseline (A)=0.780")
    ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Episode AUC")
    ax.set_title("LR all features (best model)")
    ax.legend(fontsize=7)
    ax.set_xlim(0.3, 1.05)

    fig.suptitle("Episode-Level AUC Distribution (1,000 Stratified Holdout Splits)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_episode_auc_dist.png")
    plt.close()
    print("  fig_episode_auc_dist.png")


def fig_cox_coefficients():
    """Cox coefficient plot from holdout splits."""
    surv = pd.read_parquet(V2_RESULTS / "holdout_80_20_survival.parquet")

    # Use vol_sec feature set
    vs = surv[surv.feature_set == "vol_sec"].copy()

    # Find beta columns — only A-N for vol_sec feature set
    vol_sec_metrics = ["a", "b", "c", "d", "e", "f",
                       "g", "h", "i", "j", "k", "l", "m", "n"]
    beta_cols = [f"cox_beta_metric_{m}" for m in vol_sec_metrics
                 if f"cox_beta_metric_{m}" in vs.columns]
    if not beta_cols:
        print("  fig_cox_coefficients: no beta columns found")
        return

    # Compute median and CI
    records = []
    for col in beta_cols:
        name = col.replace("cox_beta_", "").replace("metric_", "").upper()
        vals = vs[col].dropna()
        if len(vals) < 10:
            continue
        records.append({
            "metric": name,
            "median": vals.median(),
            "lo": vals.quantile(0.025),
            "hi": vals.quantile(0.975),
        })

    df = pd.DataFrame(records).sort_values("median")

    fig, ax = plt.subplots(figsize=(7, 6))
    y = range(len(df))

    colors = []
    for _, row in df.iterrows():
        if row["lo"] > 0 or row["hi"] < 0:
            colors.append("firebrick")
        else:
            colors.append("gray")

    ax.barh(list(y), df["median"].values, color=colors, alpha=0.7, height=0.6)
    ax.errorbar(df["median"].values, list(y),
                xerr=[df["median"].values - df["lo"].values,
                      df["hi"].values - df["median"].values],
                fmt="none", color="black", capsize=3, lw=0.8)

    ax.set_yticks(list(y))

    # Map metric letters to descriptions
    desc = {
        "A": "A: Vol ratio", "B": "B: Vol gap", "C": "C: Correlation",
        "D": "D: Vol-of-vol", "E": "E: Investment", "F": "F: Leverage traj",
        "G": "G: Sentiment", "H": "H: Uncertainty", "I": "I: Readability",
        "J": "J: Off-bal-sheet", "K": "K: Risk escalation", "L": "L: Growth narr",
        "M": "M: Filing length", "N": "N: Negativity",
    }
    labels = [desc.get(m, m) for m in df["metric"].values]
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Cox Coefficient (per 1 SD)")
    ax.set_title("Cox PH Coefficients — vol+SEC (1,000 splits)", fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_cox_coefficients.png")
    plt.close()
    print("  fig_cox_coefficients.png")


def fig_cvar_and_impact():
    """CVaR distribution + per-episode impact bar chart."""
    cvar_df = pd.read_parquet(V2_RESULTS / "cvar_splits.parquet")
    impact_path = V2_RESULTS / "episode_impact.parquet"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AUC histogram with CVaR markers
    ax = axes[0]
    metric = "rsf_episode_auc"
    vals = cvar_df[metric].dropna().sort_values().values

    ax.hist(vals, bins=25, alpha=0.6, color="steelblue", edgecolor="white")
    ax.axvline(np.median(vals), color="black", ls="-", lw=1.5,
               label=f"Median={np.median(vals):.3f}")
    ax.axvline(0.5, color="gray", ls=":", lw=0.8)

    # CVaR thresholds
    n = len(vals)
    for pct, color_marker in [(5, "red"), (10, "darkorange"), (25, "orange")]:
        cutoff_idx = max(1, int(n * pct / 100))
        threshold = vals[cutoff_idx - 1]
        tail_mean = vals[:cutoff_idx].mean()
        ax.axvline(threshold, color=color_marker, ls="--", lw=0.8, alpha=0.8)
        ax.text(threshold - 0.02, ax.get_ylim()[1] * 0.85,
                f"CVaR {pct}%\n={tail_mean:.2f}", fontsize=7,
                color=color_marker, ha="right")

    ax.set_xlabel("Episode AUC")
    ax.set_ylabel("Count")
    ax.set_title("AUC Distribution with CVaR Thresholds")
    ax.legend(fontsize=8)

    # Right: episode impact
    ax = axes[1]
    if impact_path.exists():
        impact = pd.read_parquet(impact_path).sort_values("impact")
        # Show top/bottom 10
        show = pd.concat([impact.head(8), impact.tail(8)])
        show = show.drop_duplicates(subset="episode").sort_values("impact")

        colors = ["coral" if v < 0 else "steelblue" for v in show["impact"]]
        ax.barh(range(len(show)), show["impact"].values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels(show["episode"].values, fontsize=7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Impact on AUC\n(negative = model needs this episode)")
        ax.set_title("Per-Episode AUC Impact")
    else:
        ax.text(0.5, 0.5, "episode_impact.parquet\nnot found",
                ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("CVaR Analysis (vol+SEC, 1,000 splits)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_cvar_and_impact.png")
    plt.close()
    print("  fig_cvar_and_impact.png")


def fig_text_vs_bond_civ():
    """Scatter: text-CIV vs bond-CIV validation."""
    pairs = []
    for bond_path in V2_CIV.glob("*_bond_civ.parquet"):
        eid = bond_path.stem.replace("_bond_civ", "")
        text_path = V2_CIV / f"{eid}_text_civ.parquet"
        if not text_path.exists():
            continue
        bond = pd.read_parquet(bond_path)
        text = pd.read_parquet(text_path)
        bond["date"] = pd.to_datetime(bond["date"])
        text["date"] = pd.to_datetime(text["date"])
        merged = bond[["date", "civ_median"]].merge(
            text[["date", "text_civ"]], on="date", how="inner").dropna()
        if not merged.empty:
            merged["episode_id"] = eid
            pairs.append(merged)

    if not pairs:
        print("  fig_text_vs_bond_civ: no overlapping data")
        return

    all_pairs = pd.concat(pairs, ignore_index=True)
    corr = all_pairs["civ_median"].corr(all_pairs["text_civ"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_pairs.civ_median * 100, all_pairs.text_civ * 100,
               alpha=0.1, s=5, c="steelblue")
    ax.set_xlabel("Bond-CIV (%)")
    ax.set_ylabel("Text-CIV (%)")
    ax.set_title(f"Text-CIV vs Bond-CIV (r={corr:.3f}, n={len(all_pairs):,})",
                 fontweight="bold")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "r--", lw=0.8, alpha=0.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_text_vs_bond_civ.png")
    plt.close()
    print(f"  fig_text_vs_bond_civ.png (r={corr:.3f})")


def main():
    print("Generating paper figures...")
    print(f"  Reading from: {V2_RESULTS}")
    print(f"  Writing to:   {FIGURES_DIR}\n")

    fig_episode_auc_dist()
    fig_cox_coefficients()
    fig_cvar_and_impact()
    fig_text_vs_bond_civ()

    print("\nDone. Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
