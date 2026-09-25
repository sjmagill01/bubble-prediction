"""
18_bw_horse_race.py — Baker-Wurgler sentiment horse-race.

Question: do the 23 project features add information beyond readily available
macro sentiment measures, or does our model merely capture the business cycle?

Two sentiment benchmarks:
  1. UMCSENT (University of Michigan Consumer Sentiment, FRED) — monthly,
     1978-present, full episode coverage.
  2. Baker-Wurgler SENT (orthogonalized, monthly) — 1958-2010, covers only
     the pre-2010 subsample of episodes (used as a subsidiary check).

Four model specs compared (1,000 leverage-stratified 80/20 splits each):
  A. LR-23       : our 23 features (baseline)
  B. LR-SENT     : sentiment feature only (UMCSENT or BW)
  C. LR-23+SENT  : 23 + sentiment (does sentiment add to ours?)
  D. Naive        : vol ratio (metric_a) only, no fitting

Key test: does LR-23+SENT improve on LR-23?
  If NO  -> our 23 features already subsume sentiment information.
  If YES -> sentiment adds incremental value; note the gap.

Outputs:
  data/results/bw_horse_race.csv
  figures/fig_bw_horse_race.png
"""
import warnings
warnings.filterwarnings("ignore")

import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (PANELS_DIR, RESULTS_DIR, FIGURES_DIR,
                    DEFAULT_SEED, ALL_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Fetch sentiment data
# ---------------------------------------------------------------------------

def fetch_umcsent():
    """University of Michigan Consumer Sentiment from FRED. Monthly 1978-present."""
    r = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UMCSENT",
        timeout=30
    )
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["observation_date"])
    df = (df.dropna()
            .rename(columns={"observation_date": "date", "UMCSENT": "umcsent"})
            .set_index("date")["umcsent"])
    df.index = df.index.to_period("M")
    # Standardize to zero mean unit variance over full history
    df = (df - df.mean()) / df.std()
    print(f"  UMCSENT: {df.index.min()} to {df.index.max()} ({len(df)} months)")
    return df


def fetch_bw_sent():
    """Baker-Wurgler orthogonalized sentiment (SENT). Monthly 1958-2010."""
    url = "https://pages.stern.nyu.edu/~jwurgler/data/Investor_Sentiment_V16_Post.xls"
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    df = pd.read_excel(io.BytesIO(raw), sheet_name="POST Monthly", header=9)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["yearmo"].apply(lambda x: str(x).strip().isdigit())].copy()
    df["yearmo"] = df["yearmo"].astype(int)
    df = df[df["SENT"].notna()].copy()
    # Parse yearmo -> period
    df["period"] = pd.PeriodIndex(
        df["yearmo"].astype(str).str.zfill(6), freq="M"
    )
    s = df.set_index("period")["SENT"]
    s = (s - s.mean()) / s.std()
    print(f"  BW SENT: {s.index.min()} to {s.index.max()} ({len(s)} months)")
    return s


# ---------------------------------------------------------------------------
# Build episode-level sentiment features
# ---------------------------------------------------------------------------

def episode_sentiment(panel, sent_series, col_name):
    """
    For each episode, compute the mean sentiment over its pre-peak window
    (tau from tau_min to -1). Returns a Series indexed by episode_id.
    """
    panel = panel.copy()
    panel["period"] = pd.PeriodIndex(
        pd.to_datetime(panel["date"]).dt.to_period("M")
    )
    panel[col_name] = panel["period"].map(sent_series)
    ep_sent = (panel
               .groupby("episode_id")[col_name]
               .mean())
    n_missing = ep_sent.isna().sum()
    n_total = len(ep_sent)
    print(f"  {col_name}: {n_total - n_missing}/{n_total} episodes have coverage")
    return ep_sent


# ---------------------------------------------------------------------------
# Split reconstruction (same seed -> same splits as original run)
# ---------------------------------------------------------------------------

def load_split_episodes(surv_df, feat_name="all"):
    """Dict: split_i -> list of held-out episode_ids."""
    sub = surv_df[surv_df["feature_set"] == feat_name]
    out = {}
    for _, row in sub.iterrows():
        si = int(row["split"])
        bubs = row["ho_bubbles"].split("|") if isinstance(row["ho_bubbles"], str) else []
        nears = row["ho_nears"].split("|") if isinstance(row["ho_nears"], str) else []
        out[si] = bubs + nears
    return out


# ---------------------------------------------------------------------------
# One-split evaluation
# ---------------------------------------------------------------------------

def run_one_split(ep_df, feature_cols, sent_col, test_ids, seed=42):
    """
    Runs four models on one split. Returns dict with AUCs.
    feature_cols: the base 23 feature columns
    sent_col: name of sentiment column in ep_df (may have NaNs)
    """
    tr = ep_df[~ep_df["episode_id"].isin(test_ids)].copy()
    te = ep_df[ep_df["episode_id"].isin(test_ids)].copy()

    if len(np.unique(tr["is_bubble"])) < 2 or len(np.unique(te["is_bubble"])) < 2:
        return None

    y_tr, y_te = tr["is_bubble"].values, te["is_bubble"].values

    def fit_score(X_tr, X_te):
        sc = StandardScaler()
        Xts = sc.fit_transform(X_tr)
        Xes = sc.transform(X_te)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        try:
            lr.fit(Xts, y_tr)
            return roc_auc_score(y_te, lr.predict_proba(Xes)[:, 1])
        except Exception:
            return np.nan

    # Base features (impute NaN with 0 as in original pipeline)
    def get_X(df, cols, extra_cols=None):
        out = df[cols].fillna(0).values
        if extra_cols:
            for c in extra_cols:
                col_vals = df[c].values.reshape(-1, 1)
                # impute sentiment NaN with 0
                col_vals = np.where(np.isnan(col_vals.astype(float)), 0.0, col_vals.astype(float))
                out = np.hstack([out, col_vals])
        return out

    result = {}

    # A: LR-23
    result["lr23"] = fit_score(get_X(tr, feature_cols), get_X(te, feature_cols))

    # B: LR-SENT only
    sent_available_tr = tr[sent_col].notna().sum()
    sent_available_te = te[sent_col].notna().sum()
    if sent_available_tr >= 2 and sent_available_te >= 1:
        result["lr_sent"] = fit_score(
            get_X(tr, [], [sent_col]),
            get_X(te, [], [sent_col])
        )
    else:
        result["lr_sent"] = np.nan

    # C: LR-23+SENT
    if sent_available_tr >= 2 and sent_available_te >= 1:
        result["lr23_sent"] = fit_score(
            get_X(tr, feature_cols, [sent_col]),
            get_X(te, feature_cols, [sent_col])
        )
    else:
        result["lr23_sent"] = np.nan

    # D: Naive (metric_a, no fitting)
    if "metric_a" in te.columns:
        mask = te["metric_a"].notna() & tr["metric_a"].notna()
        try:
            result["naive"] = roc_auc_score(y_te, te["metric_a"].fillna(0).values)
        except Exception:
            result["naive"] = np.nan
    else:
        result["naive"] = np.nan

    return result


# ---------------------------------------------------------------------------
# Run full horse-race
# ---------------------------------------------------------------------------

def run_horse_race(ep_df, feat_cols, sent_col, split_episodes, label):
    """Run all splits for one sentiment measure. Returns DataFrame of AUCs."""
    print(f"\n  Running {label} horse-race ({len(split_episodes)} splits)...")
    rows = []
    for si, test_ids in split_episodes.items():
        r = run_one_split(ep_df, feat_cols, sent_col, test_ids)
        if r is not None:
            r["split"] = si
            rows.append(r)
    df = pd.DataFrame(rows)
    for col in ["lr23", "lr_sent", "lr23_sent", "naive"]:
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"    {col:12s}: median={vals.median():.4f}  "
                  f"[{vals.quantile(0.025):.3f}, {vals.quantile(0.975):.3f}]  "
                  f"n={len(vals)}")
    return df


# ---------------------------------------------------------------------------
# Wilcoxon tests
# ---------------------------------------------------------------------------

def paired_wilcoxon(a, b, label_a, label_b):
    """Paired Wilcoxon: H0 a <= b (one-sided, a > b)."""
    paired = pd.DataFrame({"a": a, "b": b}).dropna()
    gaps = paired["a"].values - paired["b"].values
    try:
        stat, p = wilcoxon(gaps, alternative="greater")
    except Exception:
        stat, p = np.nan, np.nan
    med_gap = np.median(gaps)
    pct_better = np.mean(gaps > 0)
    print(f"  {label_a} vs {label_b}: median gap={med_gap:+.4f}, "
          f"{100*pct_better:.1f}% splits better, p={p:.4f}")
    return {"label_a": label_a, "label_b": label_b,
            "median_gap": med_gap, "pct_better": pct_better,
            "wilcoxon_p": p, "n_pairs": len(paired)}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_horse_race(umcsent_df, bw_df, save_path):
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    colors = {
        "lr23":      "#e05c2a",
        "lr_sent":   "#4a90d9",
        "lr23_sent": "#6a4c9c",
        "naive":     "#888888",
    }
    labels = {
        "lr23":      "LR-23 (base model)",
        "lr_sent":   "LR-SENT only",
        "lr23_sent": "LR-23 + SENT",
        "naive":     "Naive (vol ratio)",
    }
    bins = np.linspace(0.3, 1.0, 35)

    # ── Panel 1: UMCSENT AUC distributions ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    for col in ["naive", "lr_sent", "lr23", "lr23_sent"]:
        vals = umcsent_df[col].dropna()
        ax1.hist(vals, bins=bins, alpha=0.55, color=colors[col],
                 label=f"{labels[col]}\n(med={vals.median():.3f})")
    ax1.set_xlabel("Episode AUC")
    ax1.set_ylabel("Splits")
    ax1.set_title("UMCSENT horse-race\n(full sample, n=69 episodes)", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left")

    # ── Panel 2: BW AUC distributions (subsample) ───────────────────────────
    ax2 = fig.add_subplot(gs[1])
    if bw_df is not None:
        for col in ["naive", "lr_sent", "lr23", "lr23_sent"]:
            vals = bw_df[col].dropna()
            ax2.hist(vals, bins=bins, alpha=0.55, color=colors[col],
                     label=f"{labels[col]}\n(med={vals.median():.3f})")
        ax2.set_title("BW SENT horse-race\n(pre-2010 subsample)", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "BW data\nnot available", ha="center", va="center",
                 transform=ax2.transAxes)
    ax2.set_xlabel("Episode AUC")
    ax2.set_ylabel("Splits")
    ax2.legend(fontsize=7, loc="upper left")

    # ── Panel 3: Incremental value of adding SENT to LR-23 ──────────────────
    ax3 = fig.add_subplot(gs[2])
    gap_umcsent = (umcsent_df["lr23_sent"] - umcsent_df["lr23"]).dropna()
    ax3.hist(gap_umcsent, bins=np.linspace(-0.4, 0.4, 35),
             color="#4a90d9", alpha=0.7, label=f"UMCSENT (med={gap_umcsent.median():+.3f})")
    if bw_df is not None:
        gap_bw = (bw_df["lr23_sent"] - bw_df["lr23"]).dropna()
        ax3.hist(gap_bw, bins=np.linspace(-0.4, 0.4, 35),
                 color="#e05c2a", alpha=0.55, label=f"BW (med={gap_bw.median():+.3f})")
    ax3.axvline(0, color="black", lw=1.2)
    ax3.set_xlabel("AUC gap: (LR-23+SENT) - LR-23")
    ax3.set_ylabel("Splits")
    ax3.set_title("Incremental value of adding sentiment\nto the 23-feature model", fontsize=9)
    ax3.legend(fontsize=8)

    plt.suptitle("Baker-Wurgler Sentiment Horse-Race", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Baker-Wurgler / UMCSENT Sentiment Horse-Race")
    print("=" * 60)

    # Load panel
    panel = pd.concat([
        pd.read_parquet(PANELS_DIR / "train.parquet"),
        pd.read_parquet(PANELS_DIR / "test.parquet"),
    ], ignore_index=True)

    feat_cols = [c for c in ALL_FEATURE_COLS if c in panel.columns]
    ep_base = (panel
               .groupby(["episode_id", "source"])[feat_cols]
               .mean()
               .reset_index())
    ep_base["is_bubble"] = (ep_base["source"] == "bubble").astype(int)

    # Load splits (same as original 1000-split run)
    surv_df = pd.read_parquet(RESULTS_DIR / "holdout_80_20_survival.parquet")
    split_episodes = load_split_episodes(surv_df, feat_name="all")
    print(f"  Splits loaded: {len(split_episodes)}")

    # ── UMCSENT ──────────────────────────────────────────────────────────────
    print("\n[1] Fetching UMCSENT (FRED)...")
    umcsent = fetch_umcsent()
    ep_umcsent = ep_base.copy()
    ep_umcsent["sent"] = ep_umcsent["episode_id"].map(
        episode_sentiment(panel, umcsent, "umcsent")
    )
    umcsent_df = run_horse_race(ep_umcsent, feat_cols, "sent", split_episodes, "UMCSENT")

    # ── BW SENT (subsample) ───────────────────────────────────────────────────
    print("\n[2] Fetching Baker-Wurgler SENT...")
    bw_df = None
    try:
        bw_sent = fetch_bw_sent()
        ep_bw = ep_base.copy()
        ep_bw["sent"] = ep_bw["episode_id"].map(
            episode_sentiment(panel, bw_sent, "bw_sent")
        )
        # Only keep episodes where BW has >=50% coverage of their pre-peak window
        panel["period"] = pd.PeriodIndex(
            pd.to_datetime(panel["date"]).dt.to_period("M")
        )
        panel["bw_avail"] = panel["period"].map(bw_sent).notna()
        bw_coverage = panel.groupby("episode_id")["bw_avail"].mean()
        covered = bw_coverage[bw_coverage >= 0.5].index
        print(f"  Episodes with >=50% BW coverage: {len(covered)}/69")

        # Filter split_episodes to only those where both train+test have coverage
        bw_splits = {}
        ep_ids = set(ep_bw["episode_id"])
        covered_set = set(covered)
        for si, test_ids in split_episodes.items():
            if all(tid in covered_set for tid in test_ids):
                bw_splits[si] = test_ids
        print(f"  Splits with full BW test-set coverage: {len(bw_splits)}")

        if len(bw_splits) >= 50:
            bw_df = run_horse_race(ep_bw, feat_cols, "sent", bw_splits, "BW SENT")
        else:
            print("  Too few BW-covered splits; skipping BW analysis.")
    except Exception as e:
        print(f"  BW fetch failed: {e}")

    # ── Statistical tests ─────────────────────────────────────────────────────
    print("\nPaired Wilcoxon tests (H0: row <= column):")
    wilcox_rows = []

    # UMCSENT tests
    for pair in [("lr23", "naive"), ("lr23", "lr_sent"), ("lr23_sent", "lr23")]:
        r = paired_wilcoxon(umcsent_df[pair[0]], umcsent_df[pair[1]],
                            f"UMCSENT:{pair[0]}", f"UMCSENT:{pair[1]}")
        r["benchmark"] = "UMCSENT"
        wilcox_rows.append(r)

    # BW tests
    if bw_df is not None:
        for pair in [("lr23", "naive"), ("lr23", "lr_sent"), ("lr23_sent", "lr23")]:
            r = paired_wilcoxon(bw_df[pair[0]], bw_df[pair[1]],
                                f"BW:{pair[0]}", f"BW:{pair[1]}")
            r["benchmark"] = "BW"
            wilcox_rows.append(r)

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Combine all AUC distributions
    umcsent_df["benchmark"] = "UMCSENT"
    if bw_df is not None:
        bw_df["benchmark"] = "BW"
        combined = pd.concat([umcsent_df, bw_df], ignore_index=True)
    else:
        combined = umcsent_df.copy()
    combined.to_csv(RESULTS_DIR / "bw_horse_race_aucs.csv", index=False)

    wilcox_df = pd.DataFrame(wilcox_rows)
    wilcox_df.to_csv(RESULTS_DIR / "bw_horse_race.csv", index=False)
    print(f"\nResults saved.")

    # ── Figure ────────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_horse_race(umcsent_df, bw_df,
                   save_path=FIGURES_DIR / "fig_bw_horse_race.png")

    # ── Paper-ready summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PAPER-READY SUMMARY")
    print("=" * 60)
    lr23_med = umcsent_df["lr23"].dropna().median()
    sent_med = umcsent_df["lr_sent"].dropna().median()
    combo_med = umcsent_df["lr23_sent"].dropna().median()
    naive_med = umcsent_df["naive"].dropna().median()
    incr_gap = (umcsent_df["lr23_sent"] - umcsent_df["lr23"]).dropna().median()

    print(f"UMCSENT full-sample results (1,000 splits):")
    print(f"  Naive baseline:     {naive_med:.3f}")
    print(f"  LR-SENT only:       {sent_med:.3f}")
    print(f"  LR-23 (base):       {lr23_med:.3f}")
    print(f"  LR-23 + UMCSENT:    {combo_med:.3f}  (incremental gap: {incr_gap:+.4f})")

    incr_p = next((r["wilcoxon_p"] for r in wilcox_rows
                   if r["label_a"] == "UMCSENT:lr23_sent"), np.nan)
    print(f"  Wilcoxon p (23+SENT vs 23): {incr_p:.4f}")

    if abs(incr_gap) < 0.005 or incr_p > 0.10:
        print(f"\n  >> UMCSENT does NOT significantly improve on LR-23.")
        print(f"     The 23-feature model subsumes macro sentiment information.")
    else:
        print(f"\n  >> UMCSENT adds {incr_gap:+.4f} AUC to LR-23 (p={incr_p:.4f}).")
        print(f"     Consider adding UMCSENT as a 24th feature.")


if __name__ == "__main__":
    main()
