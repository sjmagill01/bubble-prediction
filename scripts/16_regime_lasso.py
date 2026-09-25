"""
16_regime_lasso.py — Regime-aware LASSO logistic for bubble classification.

Problem
-------
With n=69 episodes and p=23 features, plain LR overfits random splits and
the un-fitted naive baseline (metric_a alone, AUC=0.780) beats the trained
model at every walk-forward cutoff.

Design
------
P1  Load panels; map category -> regime (leverage / mania).
P2  Collapse to 1 row per episode (mean of pre-peak metric values).
P3  Build feature matrix:
      base metrics (23, from config.ALL_FEATURE_COLS)
    + regime_mania indicator (1)
    + base x regime_mania interactions (23)
    = 47 features, standardised within each CV fold.
P4  LOO-CV over C grid to select LASSO C (1/lambda).
      For each C: collect all n LOO predictions, compute episode AUC.
      Walk-forward: C selected on training episodes only (no leakage).
      80/20 evaluation: C frozen from full-data LOO then held constant
      across splits (model coefficients still fit without held-out episodes).
P5  Walk-forward evaluation (honest, pre-2015 train / 2015+ test).
P6  80/20 evaluation (1000 unique splits, matching 07_run_model.py style).
P7  Comparison table vs naive and original LR.
P8  Save to data/results/regime_lasso.parquet.

Usage
-----
  python scripts/16_regime_lasso.py                  # full run
  python scripts/16_regime_lasso.py --expand         # expanding walk-forward
  python scripts/16_regime_lasso.py --score-targets  # score AI/quantum/nuclear2
  python scripts/16_regime_lasso.py --figures        # generate figures
  python scripts/16_regime_lasso.py --smoke          # smoke test

Smoke test (--smoke)
    Synthetic 16-episode panel -> full pipeline with tiny C grid and 10 splits.
    Asserts regime counts, feature matrix shape, AUC in [0,1].
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import (
    PANELS_DIR, RESULTS_DIR, ALL_FEATURE_COLS,
    WALK_FORWARD_CUTOFF, DEFAULT_SEED,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── P1: Regime mapping ────────────────────────────────────────────────────────

REGIME_MAP: dict = {
    "financial": "leverage",
    "capex":     "leverage",
    "mania":     "mania",
    "commodity": "mania",
}

# C = 1/lambda; smaller C = stronger L1 regularisation
C_GRID = np.logspace(-3, 1, 20)


# ── P2: Episode collapse ──────────────────────────────────────────────────────

def make_episode_df(panel, base_cols):
    """
    Collapse panel (monthly rows) to 1 row per episode.

    Aggregation: mean of pre-peak metric values.
    Adds: is_bubble, regime, regime_mania indicator, interaction terms.
    """
    agg_cols = [c for c in base_cols if c in panel.columns]

    meta = ["episode_id", "source", "category", "peak_date"]
    ep = (
        panel.groupby(meta)[agg_cols]
        .mean()
        .reset_index()
    )

    ep["is_bubble"]    = (ep["source"] == "bubble").astype(int)
    ep["regime"]       = ep["category"].map(REGIME_MAP)
    ep["regime_mania"] = (ep["regime"] == "mania").astype(float)

    unknown = ep["regime"].isna()
    if unknown.any():
        bad = ep.loc[unknown, "category"].unique().tolist()
        raise ValueError(f"Unmapped categories in REGIME_MAP: {bad}")

    for col in agg_cols:
        ep[f"{col}_x_mania"] = ep[col] * ep["regime_mania"]

    return ep, agg_cols


# ── P3: Feature matrix ────────────────────────────────────────────────────────

def get_feature_names(base_cols):
    return base_cols + ["regime_mania"] + [f"{c}_x_mania" for c in base_cols]


def build_X(ep_df, base_cols, train_stats=None):
    """
    Extract and standardise the feature matrix from an episode DataFrame.

    train_stats: if None, compute mean/std from ep_df.
                 if provided, apply those stats (no leakage into test set).
    """
    feat_names = [f for f in get_feature_names(base_cols) if f in ep_df.columns]
    X = ep_df[feat_names].copy().fillna(0.0).values.astype(float)

    if train_stats is None:
        means = X.mean(axis=0)
        stds  = X.std(axis=0)
        train_stats = {"means": means, "stds": stds, "feat_names": feat_names}

    means = train_stats["means"]
    stds  = train_stats["stds"]
    X_std = np.where(stds > 1e-8, (X - means) / stds, 0.0)

    return X_std, feat_names, train_stats


# ── P4: LOO-CV C selection ────────────────────────────────────────────────────

def loo_select_C(ep_df, base_cols, c_grid=C_GRID, verbose=False):
    """
    Leave-one-episode-out cross-validation to select the best LASSO C.

    For each C, collect LOO predictions across all episodes, then compute
    episode AUC from those predictions. Standardisation is inside each fold.

    Returns (best_C, {C: AUC} dict).
    """
    n        = len(ep_df)
    y        = ep_df["is_bubble"].values
    auc_by_C = {}

    for C in c_grid:
        loo_scores = np.full(n, np.nan)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            tr = ep_df.iloc[mask].reset_index(drop=True)
            te = ep_df.iloc[[i]].reset_index(drop=True)

            if len(np.unique(tr["is_bubble"].values)) < 2:
                continue

            X_tr, _, stats = build_X(tr, base_cols)
            X_te, _, _     = build_X(te, base_cols, train_stats=stats)

            try:
                m = LogisticRegression(
                    penalty="l1", solver="liblinear",
                    C=C, max_iter=500, random_state=42,
                )
                m.fit(X_tr, tr["is_bubble"].values)
                loo_scores[i] = m.predict_proba(X_te)[0, 1]
            except Exception:
                pass

        valid = ~np.isnan(loo_scores)
        if valid.sum() >= 4 and len(np.unique(y[valid])) == 2:
            try:
                auc_by_C[C] = roc_auc_score(y[valid], loo_scores[valid])
            except Exception:
                auc_by_C[C] = np.nan
        else:
            auc_by_C[C] = np.nan

    best_C = max(
        auc_by_C,
        key=lambda c: auc_by_C[c] if not np.isnan(auc_by_C[c]) else -1.0,
    )

    if verbose:
        print("  LOO AUC by C (top 5):")
        ranked = sorted(
            auc_by_C.items(),
            key=lambda x: x[1] if not np.isnan(x[1]) else -1.0,
            reverse=True,
        )
        for c, a in ranked[:5]:
            auc_str = f"{a:.3f}" if not np.isnan(a) else "N/A"
            print(f"    C={c:.5f}  AUC={auc_str}")
        print(f"  Best C = {best_C:.5f}")

    return best_C, auc_by_C


# ── Episode AUC helper ────────────────────────────────────────────────────────

def episode_auc(ep_df, scores, regime=None):
    """AUC overall or within 'mania' / 'leverage' regime."""
    if regime is not None:
        mask   = ep_df["regime"].values == regime
        ep_df  = ep_df[mask].reset_index(drop=True)
        scores = scores[mask]

    y     = ep_df["is_bubble"].values
    valid = np.isfinite(scores)

    if valid.sum() < 4 or len(np.unique(y[valid])) < 2:
        return np.nan

    return float(roc_auc_score(y[valid], scores[valid]))


# ── P5: Walk-forward evaluation ───────────────────────────────────────────────

def eval_walk_forward(ep_df, base_cols, cutoff=WALK_FORWARD_CUTOFF):
    """
    Temporally honest evaluation: train on peak < cutoff, test on peak >= cutoff.
    C is selected via LOO on training episodes only (no future leakage).
    """
    print(f"\nWalk-Forward Evaluation  (cutoff = {cutoff})")
    print("=" * 60)

    tr = ep_df[ep_df["peak_date"] <  cutoff].reset_index(drop=True)
    te = ep_df[ep_df["peak_date"] >= cutoff].reset_index(drop=True)

    print(f"  Train: {len(tr)} episodes   ({tr['is_bubble'].sum()} bubbles)")
    print(f"  Test:  {len(te)} episodes   ({te['is_bubble'].sum()} bubbles)")

    if len(tr) < 6 or len(np.unique(tr["is_bubble"].values)) < 2:
        print("  Insufficient training data -- skipping.")
        return {}

    best_C, _ = loo_select_C(tr, base_cols, verbose=True)

    X_tr, feat_names, stats = build_X(tr, base_cols)
    X_te, _, _              = build_X(te, base_cols, train_stats=stats)

    m = LogisticRegression(
        penalty="l1", solver="liblinear",
        C=best_C, max_iter=500, random_state=42,
    )
    m.fit(X_tr, tr["is_bubble"].values)
    te_scores = m.predict_proba(X_te)[:, 1]

    coef      = m.coef_[0]
    nonzero   = [(feat_names[j], coef[j])
                 for j in range(len(feat_names)) if abs(coef[j]) > 1e-6]
    n_nonzero = len(nonzero)
    n_total   = len(feat_names)

    print(f"\n  Surviving features: {n_nonzero} / {n_total}")
    for fname, fc in sorted(nonzero, key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"    {fname:35s}: {fc:+.4f}")

    result = {
        "best_C":       best_C,
        "n_train":      len(tr),
        "n_test":       len(te),
        "auc_overall":  episode_auc(te, te_scores),
        "auc_mania":    episode_auc(te, te_scores, regime="mania"),
        "auc_leverage": episode_auc(te, te_scores, regime="leverage"),
    }

    print()
    for k, v in result.items():
        if isinstance(v, float):
            v_str = f"{v:.3f}" if not np.isnan(v) else "N/A"
            print(f"  {k}: {v_str}")

    return result


# ── P6: 80/20 evaluation ─────────────────────────────────────────────────────

def eval_80_20(ep_df, base_cols, best_C, n_splits=1000, seed=DEFAULT_SEED):
    """
    1000 unique 80/20 episode splits.

    C is frozen (selected via LOO on all episodes before calling this).
    Per split, LASSO is fitted on the 80% training portion only.
    """
    print(f"\n80/20 Evaluation  ({n_splits} splits,  C={best_C:.5f})")
    print("=" * 60)

    bubble_ids = sorted(ep_df[ep_df["is_bubble"] == 1]["episode_id"].unique())
    near_ids   = sorted(ep_df[ep_df["is_bubble"] == 0]["episode_id"].unique())
    n_ho_bub   = max(1, round(len(bubble_ids) * 0.2))
    n_ho_near  = max(1, round(len(near_ids)   * 0.2))

    print(f"  Bubbles: {len(bubble_ids)} total, {n_ho_bub} held out per split")
    print(f"  Near-bubbles: {len(near_ids)} total, {n_ho_near} held out per split")

    rng = np.random.RandomState(seed)
    seen   = set()
    splits = []
    while len(splits) < n_splits:
        ho_b = tuple(sorted(rng.choice(bubble_ids, size=n_ho_bub, replace=False)))
        ho_n = tuple(sorted(rng.choice(near_ids,   size=n_ho_near, replace=False)))
        key  = (ho_b, ho_n)
        if key not in seen:
            seen.add(key)
            splits.append((set(ho_b), set(ho_n)))

    results = []
    for i, (ho_b, ho_n) in enumerate(splits):
        ho_all = ho_b | ho_n

        tr = ep_df[~ep_df["episode_id"].isin(ho_all)].reset_index(drop=True)
        te = ep_df[ ep_df["episode_id"].isin(ho_all)].reset_index(drop=True)

        if len(tr) < 6 or len(np.unique(tr["is_bubble"].values)) < 2:
            continue
        if len(te) < 2 or len(np.unique(te["is_bubble"].values)) < 2:
            continue

        X_tr, _, stats = build_X(tr, base_cols)
        X_te, _, _     = build_X(te, base_cols, train_stats=stats)

        m = LogisticRegression(
            penalty="l1", solver="liblinear",
            C=best_C, max_iter=500, random_state=42,
        )
        m.fit(X_tr, tr["is_bubble"].values)
        scores = m.predict_proba(X_te)[:, 1]

        results.append({
            "split":        i,
            "best_C":       best_C,
            "auc_overall":  episode_auc(te, scores),
            "auc_mania":    episode_auc(te, scores, regime="mania"),
            "auc_leverage": episode_auc(te, scores, regime="leverage"),
        })

        if (i + 1) % 250 == 0:
            med = np.nanmedian([r["auc_overall"] for r in results])
            print(f"  split {i+1}/{n_splits}: median AUC={med:.3f}")

    df = pd.DataFrame(results)

    print(f"\n  Results ({len(df)} valid splits):")
    for col, label in [
        ("auc_overall",  "Overall"),
        ("auc_mania",    "Mania"),
        ("auc_leverage", "Leverage"),
    ]:
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"    {label:10s}: {vals.median():.3f}  "
                  f"[{vals.quantile(0.025):.3f}, {vals.quantile(0.975):.3f}]")

    return df


# ── Expanding walk-forward ────────────────────────────────────────────────────

EXPAND_CUTOFFS = ("2008-01", "2010-01", "2012-01", "2015-01", "2018-01")

def eval_expanding_walk_forward(ep_df, base_cols, cutoffs=EXPAND_CUTOFFS):
    """
    Run eval_walk_forward at multiple temporal cutoffs.
    Same cutoffs as 12_robustness.py for direct comparison.
    C is selected independently via LOO on each cutoff's training set.
    """
    print("\nExpanding Walk-Forward (Regime-LASSO)")
    print("=" * 60)

    orig_path = RESULTS_DIR / "robustness" / "expanding_walk_forward.csv"
    orig = {}
    if orig_path.exists():
        df_orig = pd.read_csv(orig_path)
        orig = dict(zip(df_orig["cutoff"], df_orig["lr_auc"]))

    rows = []
    for cutoff in cutoffs:
        result = eval_walk_forward(ep_df, base_cols, cutoff=cutoff)
        if result:
            row = {"cutoff": cutoff, **result}
        else:
            row = {"cutoff": cutoff, "n_train": np.nan, "n_test": np.nan,
                   "best_C": np.nan, "auc_overall": np.nan,
                   "auc_mania": np.nan, "auc_leverage": np.nan}
        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"\n  {'Cutoff':>10s}  {'N_tr':>5s}  {'N_te':>5s}  "
          f"{'Orig LR':>8s}  {'Regime-LASSO':>13s}  "
          f"{'Mania':>7s}  {'Leverage':>9s}")
    print(f"  {'-' * 72}")
    for _, row in df.iterrows():
        orig_str = f"{orig.get(row['cutoff'], np.nan):.3f}" \
                   if not np.isnan(orig.get(row["cutoff"], np.nan)) else "N/A"
        auc_str  = f"{row['auc_overall']:.3f}" \
                   if not np.isnan(row["auc_overall"]) else "N/A"
        man_str  = f"{row['auc_mania']:.3f}" \
                   if "auc_mania" in row and not np.isnan(row["auc_mania"]) else "N/A"
        lev_str  = f"{row['auc_leverage']:.3f}" \
                   if "auc_leverage" in row and not np.isnan(row["auc_leverage"]) else "N/A"
        n_tr = int(row["n_train"]) if not np.isnan(row.get("n_train", np.nan)) else "-"
        n_te = int(row["n_test"])  if not np.isnan(row.get("n_test",  np.nan)) else "-"
        print(f"  {row['cutoff']:>10s}  {str(n_tr):>5s}  {str(n_te):>5s}  "
              f"{orig_str:>8s}  {auc_str:>13s}  {man_str:>7s}  {lev_str:>9s}")

    return df


# ── P7: Comparison table ──────────────────────────────────────────────────────

def print_comparison(wf_result, splits_df, naive_auc=0.780):
    print(f"\n{'=' * 65}")
    print("COMPARISON TABLE")
    print(f"{'=' * 65}")
    print(f"  {'Model':32s} {'WF AUC':>8s} {'80/20':>8s} "
          f"{'Mania':>8s} {'Leverage':>10s}")
    print(f"  {'-' * 62}")

    print(f"  {'Naive  (metric_a alone)':32s} {'N/A':>8s} "
          f"{naive_auc:8.3f} {'N/A':>8s} {'N/A':>10s}")

    wf_auc  = wf_result.get("auc_overall", np.nan)
    wf_str  = f"{wf_auc:.3f}" if not np.isnan(wf_auc) else "N/A"
    med_ov  = splits_df["auc_overall"].dropna().median()
    med_man = splits_df["auc_mania"].dropna().median()
    med_lev = splits_df["auc_leverage"].dropna().median()
    man_str = f"{med_man:.3f}" if not np.isnan(med_man) else "N/A"
    lev_str = f"{med_lev:.3f}" if not np.isnan(med_lev) else "N/A"
    print(f"  {'Regime-LASSO':32s} {wf_str:>8s} "
          f"{med_ov:8.3f} {man_str:>8s} {lev_str:>10s}")
    print(f"  {'=' * 62}")


# ── Target scoring ────────────────────────────────────────────────────────────

def score_targets(ep_df, base_cols, best_C_global):
    """
    Score AI, quantum, nuclear2 using the full-sample trained Regime-LASSO.

    For each target: computes a rolling 12-month trailing mean at each date,
    adds the regime indicator from the catalog, then scores with LASSO trained
    on all 69 training episodes.

    Catalog regimes: AI=capex->leverage, quantum=mania, nuclear2=capex->leverage

    Returns dict of {episode_id: DataFrame with date + lasso_prob columns}.
    """
    target_path = PANELS_DIR / "target.parquet"
    if not target_path.exists():
        print(f"  Target panel not found: {target_path}")
        return {}

    target_panel = pd.read_parquet(target_path)
    target_panel["date"] = pd.to_datetime(target_panel["date"])
    target_panel = target_panel.sort_values(["episode_id", "date"]).reset_index(drop=True)

    # Fit on all training episodes
    X_tr, feat_names, train_stats = build_X(ep_df, base_cols)
    m = LogisticRegression(
        penalty="l1", solver="liblinear",
        C=best_C_global, max_iter=500, random_state=42,
    )
    m.fit(X_tr, ep_df["is_bubble"].values)

    out = {}
    for eid in target_panel["episode_id"].unique():
        tdf = target_panel[target_panel["episode_id"] == eid].copy()
        cat = tdf["category"].iloc[0]
        regime = REGIME_MAP[cat]
        regime_mania = float(regime == "mania")

        feat_cols = [c for c in base_cols if c in tdf.columns]
        tdf = tdf.sort_values("date").reset_index(drop=True)

        scores = []
        for i in range(len(tdf)):
            # trailing window: up to 12 months
            start = max(0, i - 11)
            window = tdf.iloc[start : i + 1]
            row_means = window[feat_cols].mean().fillna(0.0)

            # Build feature dict matching ep_df structure
            feat_dict = {col: row_means.get(col, 0.0) for col in base_cols}
            feat_dict["regime_mania"] = regime_mania
            for col in base_cols:
                feat_dict[f"{col}_x_mania"] = feat_dict[col] * regime_mania

            feat_order = [f for f in feat_names if f in feat_dict]
            vec = np.nan_to_num(
                np.array([feat_dict.get(f, 0.0) for f in feat_names]).reshape(1, -1),
                nan=0.0,
            )

            # Standardise using training statistics
            means = train_stats["means"]
            stds  = train_stats["stds"]
            vec_std = np.where(stds > 1e-8, (vec - means) / stds, 0.0)

            scores.append(float(m.predict_proba(vec_std)[0, 1]))

        tdf["lasso_prob"] = scores
        out[eid] = tdf[["date", "episode_id", "lasso_prob"]].copy()

        # Print current state
        latest = tdf.sort_values("date").iloc[-1]
        print(f"  {eid:12s} [{cat:12s} -> {regime:8s}]: "
              f"latest score = {scores[-1]:.3f}  "
              f"({latest['date'].strftime('%Y-%m')})")

    # Save
    for eid, df in out.items():
        save_path = RESULTS_DIR / f"target_{eid}_regime_lasso.parquet"
        df.to_parquet(save_path, index=False)
        print(f"    Saved -> {save_path}")

    return out


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_walk_forward_comparison():
    """
    Walk-forward comparison: Original LR vs Regime-LASSO vs Naive baseline.
    Gray band marks cutoffs where n_train < 25 (LOO unreliable for p=47).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig_path = RESULTS_DIR / "robustness" / "expanding_walk_forward.csv"
    rl_path   = RESULTS_DIR / "regime_lasso_expanding_wf.csv"

    if not orig_path.exists() or not rl_path.exists():
        print(f"  Missing input files for walk-forward figure.")
        return

    orig = pd.read_csv(orig_path)
    rl   = pd.read_csv(rl_path)

    # Align on cutoff
    merged = pd.merge(orig[["cutoff", "lr_auc", "n_train_total"]],
                      rl[["cutoff", "auc_overall", "auc_mania", "auc_leverage",
                          "n_train"]],
                      on="cutoff", how="inner")

    cutoffs   = merged["cutoff"].tolist()
    x         = np.arange(len(cutoffs))
    n_trains  = merged["n_train"].values

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(7, 4))

    # Shade unreliable region (n_train < 25)
    for i, n in enumerate(n_trains):
        if n < 25:
            ax.axvspan(i - 0.4, i + 0.4, color="#e0e0e0", alpha=0.6, zorder=0)

    # Naive baseline
    ax.axhline(0.780, color="red", ls="--", lw=1.2, label="Naive baseline (A) = 0.780")

    # Original LR
    ax.plot(x, merged["lr_auc"].values, "o-", color="steelblue", lw=1.8,
            ms=6, label="Original LR (23 features)", zorder=3)

    # Regime-LASSO
    ax.plot(x, merged["auc_overall"].values, "s-", color="darkorange", lw=1.8,
            ms=6, label="Regime-LASSO (47 features)", zorder=3)

    # Per-regime dashed lines
    ax.plot(x, merged["auc_mania"].values, "s:", color="darkorange", lw=1.0,
            ms=4, alpha=0.6, label="  Regime-LASSO (mania only)")
    ax.plot(x, merged["auc_leverage"].values, "s--", color="darkorange", lw=1.0,
            ms=4, alpha=0.6, label="  Regime-LASSO (leverage only)")

    ax.set_xticks(x)
    ax.set_xticklabels(cutoffs)
    ax.set_xlabel("Walk-forward cutoff (train peak < cutoff)")
    ax.set_ylabel("Episode AUC")
    ax.set_title("Expanding Walk-Forward: Regime-LASSO vs Original LR")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7.5, loc="upper left")

    # Annotate n_train
    for i, n in enumerate(n_trains):
        ax.text(i, 0.04, f"n={int(n)}", ha="center", va="bottom",
                fontsize=7, color="#666666")

    # Note on gray band
    ax.text(0.02, 0.96,
            "Gray: n_train < 25; LOO-CV unreliable for p=47",
            transform=ax.transAxes, fontsize=7, color="#888888",
            va="top", ha="left")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig_regime_lasso_wf.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved -> {out_path}")


def fig_target_scores(target_scores, ep_df):
    """
    3-panel figure: rolling regime-LASSO bubble probability for each target.
    Reference lines show bubble/near-bubble median scores from training.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not target_scores:
        print("  No target scores to plot.")
        return

    # Compute reference scores from full-sample LOO predictions
    # Use the 80/20 median AUC distribution as proxy: just show training medians
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })

    target_meta = {
        "AI":       ("AI Semiconductors",      "leverage"),
        "quantum":  ("Quantum Computing",       "mania"),
        "nuclear2": ("Nuclear Renaissance II",  "leverage"),
    }

    eids = [e for e in ["AI", "quantum", "nuclear2"] if e in target_scores]
    if not eids:
        return

    fig, axes = plt.subplots(1, len(eids), figsize=(4.5 * len(eids), 4))
    if len(eids) == 1:
        axes = [axes]

    for ax, eid in zip(axes, eids):
        df    = target_scores[eid].sort_values("date")
        name, regime = target_meta.get(eid, (eid, "?"))

        ax.plot(df["date"], df["lasso_prob"], lw=1.8, color="navy",
                label="Regime-LASSO score")

        # Reference lines: median bubble and near-bubble scores from LOO
        # (use a simple full-sample in-sample proxy)
        ax.axhline(0.5, color="black", ls=":", lw=0.8, label="50% threshold")
        ax.fill_between(df["date"], 0.65, 1.0,
                        alpha=0.07, color="red",
                        label="Historical bubble zone (>0.65)")

        ax.set_xlabel("Date")
        ax.set_ylabel("Bubble probability")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{name}\n(regime: {regime})")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Regime-LASSO Scores for Ongoing Targets", y=1.01, fontsize=10)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig_target_regime_scores.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved -> {out_path}")


# ── Smoke test ────────────────────────────────────────────────────────────────

def make_synthetic_panel(seed=0):
    """Tiny synthetic panel for smoke testing. 16 episodes, 6 metrics."""
    rng = np.random.RandomState(seed)

    episodes = [
        ("b_mania_1",     "bubble",      "mania",     "2013-06"),
        ("b_mania_2",     "bubble",      "mania",     "2016-03"),
        ("b_capex_1",     "bubble",      "capex",     "2014-09"),
        ("b_financial_1", "bubble",      "financial", "2018-01"),
        ("b_commodity_1", "bubble",      "commodity", "2019-06"),
        ("b_mania_3",     "bubble",      "mania",     "2021-02"),
        ("nb_mania_1",    "near_bubble", "mania",     "2013-03"),
        ("nb_mania_2",    "near_bubble", "mania",     "2015-09"),
        ("nb_capex_1",    "near_bubble", "capex",     "2014-03"),
        ("nb_capex_2",    "near_bubble", "capex",     "2016-06"),
        ("nb_financial_1","near_bubble", "financial", "2017-09"),
        ("nb_commodity_1","near_bubble", "commodity", "2018-06"),
        ("nb_mania_3",    "near_bubble", "mania",     "2019-03"),
        ("nb_mania_4",    "near_bubble", "mania",     "2020-06"),
        ("nb_capex_3",    "near_bubble", "capex",     "2021-09"),
        ("nb_financial_2","near_bubble", "financial", "2022-01"),
    ]

    metric_cols = ["metric_a", "metric_b", "metric_c",
                   "metric_d", "metric_e", "metric_f"]
    rows = []
    for ep_id, source, category, peak_date in episodes:
        is_bub = source == "bubble"
        for tau in range(-5, 0):
            row = {
                "episode_id": ep_id,
                "source":     source,
                "category":   category,
                "peak_date":  peak_date,
                "tau":        tau,
            }
            for col in metric_cols:
                signal = 0.4 if (is_bub and col == "metric_a") else 0.0
                row[col] = signal + rng.randn() * 0.5
            rows.append(row)

    return pd.DataFrame(rows)


def smoke_test():
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    panel = make_synthetic_panel()

    for col in ("episode_id", "source", "category", "peak_date"):
        assert col in panel.columns, f"Synthetic panel missing column: {col}"

    metric_cols = sorted([c for c in panel.columns if c.startswith("metric_")])
    assert len(metric_cols) == 6, \
        f"Expected 6 metric cols, got {len(metric_cols)}"

    ep_df, agg_cols = make_episode_df(panel, metric_cols)

    assert len(ep_df) == 16, f"Expected 16 episodes, got {len(ep_df)}"
    assert int(ep_df["is_bubble"].sum()) == 6, \
        f"Expected 6 bubbles, got {ep_df['is_bubble'].sum()}"
    assert set(ep_df["regime"].unique()) == {"leverage", "mania"}, \
        f"Unexpected regime values: {ep_df['regime'].unique()}"

    leverage_cats = set(ep_df[ep_df["regime"] == "leverage"]["category"].unique())
    assert leverage_cats <= {"financial", "capex"}, \
        f"Leverage regime contains unexpected categories: {leverage_cats}"

    for col in agg_cols:
        assert f"{col}_x_mania" in ep_df.columns, \
            f"Missing interaction term: {col}_x_mania"

    regime_counts = ep_df[ep_df["is_bubble"] == 1]["regime"].value_counts().to_dict()
    print(f"  Episode collapse OK: {len(ep_df)} episodes, "
          f"bubble regime split = {regime_counts}")

    X, feat_names, stats = build_X(ep_df, agg_cols)
    expected_n_feats = len(agg_cols) * 2 + 1
    assert X.shape == (16, expected_n_feats), \
        f"Feature matrix shape {X.shape}, expected (16, {expected_n_feats})"
    assert not np.any(np.isnan(X)), "NaNs in feature matrix after build_X"
    print(f"  Feature matrix OK: {X.shape}")

    tr_ep = ep_df.iloc[:12].reset_index(drop=True)
    te_ep = ep_df.iloc[12:].reset_index(drop=True)
    X_tr, _, tr_stats = build_X(tr_ep, agg_cols)
    X_te, _, _        = build_X(te_ep, agg_cols, train_stats=tr_stats)
    assert X_tr.shape[1] == X_te.shape[1], "Train/test feature count mismatch"
    print(f"  Train/test stats split OK")

    mini_grid = np.logspace(-2, 0, 5)
    best_C, auc_curve = loo_select_C(ep_df, agg_cols, c_grid=mini_grid, verbose=False)
    assert best_C in set(mini_grid.tolist()), "best_C not from grid"
    for c, a in auc_curve.items():
        assert np.isnan(a) or 0.0 <= a <= 1.0, \
            f"LOO AUC out of [0,1] at C={c}: {a}"
    auc_str = f"{auc_curve[best_C]:.3f}" if not np.isnan(auc_curve[best_C]) else "N/A"
    print(f"  LOO-CV OK: best_C={best_C:.4f}, AUC={auc_str}")

    wf = eval_walk_forward(ep_df, agg_cols, cutoff="2017-01")
    if wf:
        for k in ("auc_overall", "auc_mania", "auc_leverage"):
            v = wf[k]
            assert np.isnan(v) or 0.0 <= v <= 1.0, f"Walk-forward {k} out of [0,1]: {v}"
        wf_str = f"{wf['auc_overall']:.3f}" if not np.isnan(wf["auc_overall"]) else "N/A"
        print(f"  Walk-forward OK: overall AUC={wf_str}")

    splits_df = eval_80_20(ep_df, agg_cols, best_C=best_C, n_splits=10, seed=0)
    assert isinstance(splits_df, pd.DataFrame)
    assert all(c in splits_df.columns
               for c in ("auc_overall", "auc_mania", "auc_leverage"))

    valid_aucs = splits_df["auc_overall"].dropna()
    for v in valid_aucs:
        assert 0.0 <= v <= 1.0, f"80/20 AUC out of [0,1]: {v}"
    med_str = f"{valid_aucs.median():.3f}" if len(valid_aucs) > 0 else "N/A"
    print(f"  80/20 OK: {len(splits_df)} splits, median AUC={med_str}")

    print_comparison(wf if wf else {}, splits_df)
    print("\nSMOKE TEST PASSED")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(smoke=False, expand=False, score_targets_flag=False, figures=False):
    if smoke:
        smoke_test()
        return

    print("Loading panels...")
    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test  = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    base_cols = [c for c in ALL_FEATURE_COLS if c in panel.columns]
    missing   = [c for c in ALL_FEATURE_COLS if c not in panel.columns]
    print(f"  Features: {len(base_cols)} available"
          + (f", {len(missing)} missing" if missing else ""))

    ep_df, agg_cols = make_episode_df(panel, base_cols)
    n_bub  = int(ep_df["is_bubble"].sum())
    n_near = int((ep_df["is_bubble"] == 0).sum())
    print(f"  Episodes: {len(ep_df)}  ({n_bub} bubbles, {n_near} near-bubbles)")
    print(f"  Bubble regime split: "
          f"{ep_df[ep_df['is_bubble']==1]['regime'].value_counts().to_dict()}")

    print("\nSelecting LASSO C on full dataset (for 80/20 splits)...")
    best_C_global, _ = loo_select_C(ep_df, agg_cols, verbose=True)

    if expand:
        expand_df = eval_expanding_walk_forward(ep_df, agg_cols)
        expand_df.to_csv(RESULTS_DIR / "regime_lasso_expanding_wf.csv", index=False)
        print(f"\nSaved -> {RESULTS_DIR / 'regime_lasso_expanding_wf.csv'}")
        return

    if score_targets_flag:
        print("\nScoring targets with Regime-LASSO...")
        target_scores = score_targets(ep_df, agg_cols, best_C_global)
        return

    if figures:
        print("\nGenerating figures...")
        fig_walk_forward_comparison()
        # Score targets for the figure (quick pass)
        target_scores = score_targets(ep_df, agg_cols, best_C_global)
        fig_target_scores(target_scores, ep_df)
        return

    # Default: full run
    wf_result = eval_walk_forward(ep_df, agg_cols)
    splits_df = eval_80_20(ep_df, agg_cols, best_C=best_C_global)
    print_comparison(wf_result, splits_df)

    out = splits_df.copy()
    out["wf_auc_overall"]  = wf_result.get("auc_overall",  np.nan)
    out["wf_auc_mania"]    = wf_result.get("auc_mania",    np.nan)
    out["wf_auc_leverage"] = wf_result.get("auc_leverage", np.nan)
    out["wf_best_C"]       = wf_result.get("best_C",       np.nan)
    out["global_best_C"]   = best_C_global

    save_path = RESULTS_DIR / "regime_lasso.parquet"
    out.to_parquet(save_path, index=False)
    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regime-aware LASSO logistic for bubble classification"
    )
    parser.add_argument("--smoke",          action="store_true",
                        help="Smoke test on synthetic data")
    parser.add_argument("--expand",         action="store_true",
                        help="Expanding walk-forward at 5 temporal cutoffs")
    parser.add_argument("--score-targets",  action="store_true",
                        help="Score AI/quantum/nuclear2 with Regime-LASSO")
    parser.add_argument("--figures",        action="store_true",
                        help="Generate figures (WF comparison + target scores)")
    args = parser.parse_args()
    main(
        smoke=args.smoke,
        expand=args.expand,
        score_targets_flag=args.score_targets,
        figures=args.figures,
    )
