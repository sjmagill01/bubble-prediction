"""
10_alternative_models.py — Alternative models for bubble prediction.

1. Bayesian Logistic Regression (PyMC)    — episode identification
2. Hidden Markov Model (hmmlearn)         — crash timing
3. CUSUM monitoring                       — crash timing

All use the same leverage-stratified holdout splits as 07b for comparability.
"""
import warnings
warnings.filterwarnings("ignore")

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from config import (PANELS_DIR, RESULTS_DIR, FIGURES_DIR, DEFAULT_SEED,
                    VOL_COLS, SEC_COLS, TEXT_CIV_COLS, INTERACTION_COLS,
                    FEATURE_SETS, ALL_FEATURE_COLS)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EVENT_COL = "event_12m"
DURATION_COL = "time_to_peak"
N_SPLITS = 1000

# Feature set for new models: vol + SEC (best from prior analysis)
VOL_SEC_COLS = VOL_COLS + SEC_COLS


# ── Data loading ─────────────────────────────────────────────────────

def load_data():
    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)
    bubble_ids = sorted(panel[panel["source"] == "bubble"]["episode_id"].unique())
    near_ids = sorted(panel[panel["source"] == "near_bubble"]["episode_id"].unique())
    return panel, bubble_ids, near_ids


def make_episode_features(panel, feature_cols):
    """Collapse monthly panel to one row per episode (mean of pre-peak metrics)."""
    ep = panel.groupby(["episode_id", "source"])[feature_cols].mean().reset_index()
    ep["is_bubble"] = (ep["source"] == "bubble").astype(int)
    # Add median leverage for stratification
    lev = panel.groupby("episode_id")["med_leverage"].median()
    ep["med_leverage"] = ep["episode_id"].map(lev)
    return ep


def generate_splits(bubble_ids, near_ids, n_splits, seed, panel=None):
    """Leverage-stratified 80/20 holdout splits (same as 07b)."""
    rng = np.random.RandomState(seed)

    if panel is not None and "med_leverage" in panel.columns:
        ep_lev = panel.groupby("episode_id")["med_leverage"].median()
        lev_median = ep_lev.median()

        hi_bub = [b for b in bubble_ids if ep_lev.get(b, 0) >= lev_median]
        lo_bub = [b for b in bubble_ids if ep_lev.get(b, 0) < lev_median]
        hi_near = [n for n in near_ids if ep_lev.get(n, 0) >= lev_median]
        lo_near = [n for n in near_ids if ep_lev.get(n, 0) < lev_median]

        n_ho = {
            "hi_bub": max(1, round(len(hi_bub) * 0.2)),
            "lo_bub": max(1, round(len(lo_bub) * 0.2)),
            "hi_near": max(1, round(len(hi_near) * 0.2)),
            "lo_near": max(1, round(len(lo_near) * 0.2)),
        }

        seen = set()
        splits = []
        while len(splits) < n_splits:
            ho = []
            for group, n in [(hi_bub, n_ho["hi_bub"]), (lo_bub, n_ho["lo_bub"]),
                             (hi_near, n_ho["hi_near"]), (lo_near, n_ho["lo_near"])]:
                if group:
                    ho.extend(rng.choice(group, size=min(n, len(group)), replace=False))
            key = tuple(sorted(ho))
            if key not in seen:
                seen.add(key)
                splits.append(set(ho))
        return splits
    else:
        n_ho_bub = max(1, round(len(bubble_ids) * 0.2))
        n_ho_near = max(1, round(len(near_ids) * 0.2))
        seen = set()
        splits = []
        while len(splits) < n_splits:
            ho_b = tuple(sorted(rng.choice(bubble_ids, size=n_ho_bub, replace=False)))
            ho_n = tuple(sorted(rng.choice(near_ids, size=n_ho_near, replace=False)))
            key = (ho_b, ho_n)
            if key not in seen:
                seen.add(key)
                splits.append(set(ho_b) | set(ho_n))
        return splits


# ═══════════════════════════════════════════════════════════════════════
# MODEL 1: Bayesian Logistic Regression
# ═══════════════════════════════════════════════════════════════════════

def run_bayesian_lr_split(ep_features, feature_cols, ho_all, split_i):
    """Bayesian LR for one holdout split using PyMC."""
    import pymc as pm

    tr = ep_features[~ep_features["episode_id"].isin(ho_all)].reset_index(drop=True)
    te = ep_features[ep_features["episode_id"].isin(ho_all)].reset_index(drop=True)

    available = [c for c in feature_cols if c in tr.columns]
    X_tr = tr[available].fillna(0).values
    X_te = te[available].fillna(0).values
    y_tr = tr["is_bubble"].values
    y_te = te["is_bubble"].values

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    n_features = X_tr_s.shape[1]

    with pm.Model() as model:
        # Informative priors:
        # - Intercept: weakly informative N(0, 2)
        # - Coefficients: regularizing N(0, 1) — like L2 with sigma=1
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_features)

        # Linear predictor
        mu = alpha + pm.math.dot(X_tr_s, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(mu))

        # Likelihood
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_tr)

        # Sample
        trace = pm.sample(2000, tune=1000, cores=1, chains=2,
                          random_seed=split_i, progressbar=False,
                          return_inferencedata=True)

    # Posterior predictive on test set
    alpha_samples = trace.posterior["alpha"].values.flatten()
    beta_samples = trace.posterior["beta"].values.reshape(-1, n_features)

    # Average prediction across posterior samples
    logits = alpha_samples[:, None] + beta_samples @ X_te_s.T  # (n_samples, n_test)
    probs = 1.0 / (1.0 + np.exp(-logits))
    mean_probs = probs.mean(axis=0)  # posterior mean prediction

    try:
        auc = roc_auc_score(y_te, mean_probs)
    except ValueError:
        auc = np.nan

    # Extract posterior mean coefficients
    result = {
        "split": split_i,
        "bayes_lr_auc": auc,
        "bayes_lr_alpha": float(np.mean(alpha_samples)),
    }
    for j, col in enumerate(available):
        result[f"bayes_beta_{col}"] = float(np.mean(beta_samples[:, j]))

    # Posterior uncertainty (width of 95% CI)
    for j, col in enumerate(available):
        lo, hi = np.percentile(beta_samples[:, j], [2.5, 97.5])
        result[f"bayes_ci_width_{col}"] = hi - lo

    return result


def run_bayesian_lr(panel, ep_features, feature_cols, splits):
    """Run Bayesian LR across all splits."""
    print(f"\n{'='*60}")
    print("MODEL 1: Bayesian Logistic Regression")
    print(f"{'='*60}")
    print(f"  Features: {len(feature_cols)}, Splits: {len(splits)}")

    # Bayesian LR is slow — run fewer splits for tractability
    n_bayes_splits = min(200, len(splits))
    print(f"  Running {n_bayes_splits} splits (Bayesian sampling is slow)...")

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_bayesian_lr_split)(ep_features, feature_cols, s, i)
        for i, s in enumerate(splits[:n_bayes_splits])
    )
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)

    auc = df["bayes_lr_auc"].dropna()
    print(f"\n  Bayesian LR Episode AUC:")
    print(f"    Median: {auc.median():.3f}")
    print(f"    95% CI: [{auc.quantile(0.025):.3f}, {auc.quantile(0.975):.3f}]")

    # Posterior coefficient summary
    beta_cols = [c for c in df.columns if c.startswith("bayes_beta_")]
    print(f"\n  Posterior mean coefficients:")
    for col in beta_cols:
        name = col.replace("bayes_beta_", "")
        vals = df[col].dropna()
        ci_col = f"bayes_ci_width_{name}"
        ci_width = df[ci_col].median() if ci_col in df.columns else np.nan
        print(f"    {name:15s}: {vals.median():+.3f} (CI width={ci_width:.3f})")

    df.to_parquet(RESULTS_DIR / "bayesian_lr_results.parquet", index=False)
    print(f"  Saved to bayesian_lr_results.parquet")
    return df


# ═══════════════════════════════════════════════════════════════════════
# MODEL 2: Hidden Markov Model
# ═══════════════════════════════════════════════════════════════════════

def fit_hmm_episode(episode_data, feature_cols, n_states=3):
    """Fit HMM to a single episode's metric trajectory, return state probabilities."""
    from hmmlearn.hmm import GaussianHMM

    X = episode_data[feature_cols].values
    # Drop rows with any NaN
    valid = np.all(np.isfinite(X), axis=1)
    if valid.sum() < 10:
        return None
    X_valid = X[valid]

    hmm = GaussianHMM(n_components=n_states, covariance_type="diag",
                       n_iter=100, random_state=42, verbose=False)
    try:
        hmm.fit(X_valid)
        state_probs = hmm.predict_proba(X_valid)  # (T, n_states)
        states = hmm.predict(X_valid)

        # Identify the "dangerous" state as the one with highest mean vol ratio
        state_means = hmm.means_  # (n_states, n_features)
        # Feature 0 is metric_a (vol ratio) — dangerous state has highest
        danger_state = np.argmax(state_means[:, 0])
        danger_prob = state_probs[:, danger_state]

        return danger_prob, valid
    except Exception:
        return None


def run_hmm_timing(panel, feature_cols, splits, n_states=3):
    """Use HMM state probabilities as timing signal."""
    print(f"\n{'='*60}")
    print(f"MODEL 2: Hidden Markov Model ({n_states} states)")
    print(f"{'='*60}")

    # Use a subset of features that have good coverage and are most informative
    hmm_features = ["metric_a", "metric_d", "metric_c", "metric_f"]
    hmm_features = [f for f in hmm_features if f in panel.columns]
    print(f"  HMM features: {hmm_features}")

    results = []
    n_hmm_splits = min(200, len(splits))

    for split_i, ho_all in enumerate(splits[:n_hmm_splits]):
        tr = panel[~panel["episode_id"].isin(ho_all)].reset_index(drop=True)
        te = panel[panel["episode_id"].isin(ho_all)].reset_index(drop=True)

        if len(te) < 10:
            continue

        # Standardize using training set
        scaler = StandardScaler()
        tr_X = tr[hmm_features].fillna(0).values
        scaler.fit(tr_X)

        # Fit one global HMM on all training episodes concatenated
        from hmmlearn.hmm import GaussianHMM

        # Prepare sequences: concatenate all training episodes
        lengths = []
        X_all = []
        for eid in tr["episode_id"].unique():
            ep_data = tr[tr["episode_id"] == eid][hmm_features].fillna(0).values
            ep_scaled = scaler.transform(ep_data)
            valid = np.all(np.isfinite(ep_scaled), axis=1)
            ep_clean = ep_scaled[valid]
            if len(ep_clean) >= 6:
                X_all.append(ep_clean)
                lengths.append(len(ep_clean))

        if not X_all:
            continue

        X_concat = np.vstack(X_all)

        hmm = GaussianHMM(n_components=n_states, covariance_type="diag",
                           n_iter=200, random_state=42, verbose=False)
        try:
            hmm.fit(X_concat, lengths)
        except Exception:
            continue

        # Identify dangerous state (highest mean on metric_a = vol ratio)
        danger_state = np.argmax(hmm.means_[:, 0])

        # Score test episodes: P(dangerous state) at each month
        te_scored = te.copy()
        te_scored["hmm_danger_prob"] = np.nan

        for eid in te["episode_id"].unique():
            mask = te_scored["episode_id"] == eid
            ep_data = te_scored.loc[mask, hmm_features].fillna(0).values
            ep_scaled = scaler.transform(ep_data)
            valid_rows = np.all(np.isfinite(ep_scaled), axis=1)

            if valid_rows.sum() < 3:
                continue

            try:
                probs = hmm.predict_proba(ep_scaled[valid_rows])
                danger_probs = probs[:, danger_state]
                idx = te_scored.index[mask][valid_rows]
                te_scored.loc[idx, "hmm_danger_prob"] = danger_probs
            except Exception:
                continue

        # Episode AUC
        ep = te_scored.groupby(["episode_id", "source"]).agg(
            hmm_mean=("hmm_danger_prob", "mean")).reset_index()
        ep["is_bubble"] = (ep["source"] == "bubble").astype(int)
        v = ep["hmm_mean"].notna()

        row = {"split": split_i}
        if v.sum() >= 4 and ep.loc[v, "is_bubble"].nunique() >= 2:
            row["hmm_episode_auc"] = roc_auc_score(ep.loc[v, "is_bubble"], ep.loc[v, "hmm_mean"])
        else:
            row["hmm_episode_auc"] = np.nan

        # Month AUC (timing within bubbles)
        bub = te_scored[te_scored["source"] == "bubble"]
        valid_bub = bub["hmm_danger_prob"].notna() & np.isfinite(bub["hmm_danger_prob"])
        if valid_bub.sum() >= 10 and bub.loc[valid_bub, EVENT_COL].nunique() >= 2:
            row["hmm_month_auc"] = roc_auc_score(
                bub.loc[valid_bub, EVENT_COL], bub.loc[valid_bub, "hmm_danger_prob"])
        else:
            row["hmm_month_auc"] = np.nan

        results.append(row)

    df = pd.DataFrame(results)
    ep_auc = df["hmm_episode_auc"].dropna()
    mo_auc = df["hmm_month_auc"].dropna()
    print(f"\n  HMM Episode AUC: {ep_auc.median():.3f} [{ep_auc.quantile(0.025):.3f}, {ep_auc.quantile(0.975):.3f}]")
    print(f"  HMM Month AUC:   {mo_auc.median():.3f} [{mo_auc.quantile(0.025):.3f}, {mo_auc.quantile(0.975):.3f}]")

    df.to_parquet(RESULTS_DIR / "hmm_results.parquet", index=False)
    print(f"  Saved to hmm_results.parquet")
    return df


# ═══════════════════════════════════════════════════════════════════════
# MODEL 5: CUSUM Monitoring
# ═══════════════════════════════════════════════════════════════════════

def cusum_score_episode(ep_data, feature_col, ref_mean, ref_std, slack=0.5):
    """
    Run one-sided CUSUM on a metric trajectory.
    Returns cumulative sum signal at each month.
    """
    vals = ep_data[feature_col].values
    valid = np.isfinite(vals)
    if valid.sum() < 3:
        return np.full(len(vals), np.nan)

    # Standardize against training reference
    z = np.where(valid, (vals - ref_mean) / max(ref_std, 1e-8), 0.0)

    # One-sided upper CUSUM: detects upward shifts
    cusum = np.zeros(len(z))
    S = 0.0
    for t in range(len(z)):
        if valid[t]:
            S = max(0, S + z[t] - slack)
        cusum[t] = S

    return cusum


def run_cusum_timing(panel, splits):
    """CUSUM monitoring on vol ratio and vol-of-vol for timing."""
    print(f"\n{'='*60}")
    print("MODEL 5: CUSUM Monitoring")
    print(f"{'='*60}")

    cusum_features = ["metric_a", "metric_d"]
    n_cusum_splits = min(500, len(splits))

    results = []
    for split_i, ho_all in enumerate(splits[:n_cusum_splits]):
        tr = panel[~panel["episode_id"].isin(ho_all)].reset_index(drop=True)
        te = panel[panel["episode_id"].isin(ho_all)].reset_index(drop=True)

        if len(te) < 10:
            continue

        # Compute reference stats from training set
        ref_stats = {}
        for col in cusum_features:
            vals = tr[col].dropna()
            ref_stats[col] = {"mean": vals.mean(), "std": vals.std()}

        # Score test episodes with CUSUM
        te_scored = te.copy()
        te_scored["cusum_score"] = 0.0

        for eid in te["episode_id"].unique():
            mask = te_scored["episode_id"] == eid
            ep_data = te_scored.loc[mask]

            combined_cusum = np.zeros(mask.sum())
            for col in cusum_features:
                cusum = cusum_score_episode(
                    ep_data, col,
                    ref_stats[col]["mean"], ref_stats[col]["std"],
                    slack=0.5
                )
                combined_cusum += cusum

            te_scored.loc[mask, "cusum_score"] = combined_cusum

        # Episode AUC
        ep = te_scored.groupby(["episode_id", "source"]).agg(
            cusum_mean=("cusum_score", "mean")).reset_index()
        ep["is_bubble"] = (ep["source"] == "bubble").astype(int)

        row = {"split": split_i}
        v = ep["cusum_mean"].notna() & np.isfinite(ep["cusum_mean"])
        if v.sum() >= 4 and ep.loc[v, "is_bubble"].nunique() >= 2:
            row["cusum_episode_auc"] = roc_auc_score(ep.loc[v, "is_bubble"], ep.loc[v, "cusum_mean"])
        else:
            row["cusum_episode_auc"] = np.nan

        # Month AUC (timing)
        bub = te_scored[te_scored["source"] == "bubble"]
        valid_bub = np.isfinite(bub["cusum_score"])
        if valid_bub.sum() >= 10 and bub.loc[valid_bub, EVENT_COL].nunique() >= 2:
            row["cusum_month_auc"] = roc_auc_score(
                bub.loc[valid_bub, EVENT_COL], bub.loc[valid_bub, "cusum_score"])
        else:
            row["cusum_month_auc"] = np.nan

        results.append(row)

    df = pd.DataFrame(results)
    ep_auc = df["cusum_episode_auc"].dropna()
    mo_auc = df["cusum_month_auc"].dropna()
    print(f"  CUSUM Episode AUC: {ep_auc.median():.3f} [{ep_auc.quantile(0.025):.3f}, {ep_auc.quantile(0.975):.3f}]")
    print(f"  CUSUM Month AUC:   {mo_auc.median():.3f} [{mo_auc.quantile(0.025):.3f}, {mo_auc.quantile(0.975):.3f}]")

    df.to_parquet(RESULTS_DIR / "cusum_results.parquet", index=False)
    print(f"  Saved to cusum_results.parquet")
    return df


# ═══════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def print_comparison(results_dict):
    """Print side-by-side comparison of all models."""
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25s} {'Episode AUC':>12s} {'Month AUC':>12s} {'N splits':>10s}")
    print("-" * 60)

    # Add baselines from existing results
    for path, label, ep_col, mo_col in [
        ("holdout_80_20_classifiers.parquet", "LR (existing)", "lr_auc", None),
        ("holdout_80_20_survival.parquet", "RSF (existing)", "rsf_episode_auc", "rsf_month_auc"),
    ]:
        fpath = RESULTS_DIR / path
        if fpath.exists():
            df = pd.read_parquet(fpath)
            # Use 'all' feature set
            sub = df[df.feature_set == "all"] if "feature_set" in df.columns else df
            ep = sub[ep_col].dropna().median() if ep_col in sub.columns else np.nan
            mo = sub[mo_col].dropna().median() if mo_col and mo_col in sub.columns else np.nan
            n = len(sub)
            print(f"  {label:<23s} {ep:>10.3f}   {mo if not np.isnan(mo) else '—':>10}   {n:>8d}")

    print(f"  {'Naive baseline (A)':<23s} {'0.780':>10s}   {'—':>10s}")
    print("-" * 60)

    for name, df in results_dict.items():
        ep_col = [c for c in df.columns if "episode_auc" in c or c.endswith("_auc")]
        mo_col = [c for c in df.columns if "month_auc" in c]

        ep_val = df[ep_col[0]].dropna().median() if ep_col else np.nan
        mo_val = df[mo_col[0]].dropna().median() if mo_col else np.nan
        n = len(df)

        mo_str = f"{mo_val:.3f}" if not np.isnan(mo_val) else "—"
        print(f"  {name:<23s} {ep_val:>10.3f}   {mo_str:>10s}   {n:>8d}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    panel, bubble_ids, near_ids = load_data()
    print(f"  Panel: {len(panel)} rows, {panel.episode_id.nunique()} episodes")
    print(f"  Bubbles: {len(bubble_ids)}, Near-bubbles: {len(near_ids)}")

    feature_cols = VOL_SEC_COLS
    ep_features = make_episode_features(panel, feature_cols)
    splits = generate_splits(bubble_ids, near_ids, N_SPLITS, DEFAULT_SEED, panel)
    print(f"  Generated {len(splits)} stratified splits")

    results = {}

    # Model 1: Bayesian LR
    results["Bayesian LR"] = run_bayesian_lr(panel, ep_features, feature_cols, splits)

    # Model 2: HMM
    results["HMM (3-state)"] = run_hmm_timing(panel, feature_cols, splits, n_states=3)

    # Model 3: CUSUM
    results["CUSUM"] = run_cusum_timing(panel, splits)

    # Comparison
    print_comparison(results)

    print(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
