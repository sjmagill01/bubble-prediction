"""
07b_timing_test.py — Time 100 splits sequential vs parallel.
"""
import warnings
warnings.filterwarnings("ignore")

import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from config import (PANELS_DIR, DEFAULT_SEED, COX_PENALIZER,
                    VOL_COLS, SEC_COLS, TEXT_CIV_COLS, ALL_FEATURE_COLS,
                    FEATURE_SETS)

ALL_COLS = ALL_FEATURE_COLS
EVENT_COL = "event_12m"
DURATION_COL = "time_to_peak"


def prepare_features(df, feature_cols, train_stats=None):
    """
    Impute and standardize features using train-set statistics only.
    If train_stats=None, compute from this data (training set).
    If train_stats provided, apply those (test set — no leakage).
    """
    # Only use columns that exist in the data
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    out = df[available].copy()
    # Add missing columns as zeros
    for c in missing:
        out[c] = 0.0
    if train_stats is None:
        # Training set: compute medians, means, stds
        train_stats = {}
        for col in out.columns:
            train_stats[col] = {
                "median": out[col].median(),
                "mean": out[col].mean(),
                "std": out[col].std(),
            }
    # Impute NaN with train-set median (not expanding window)
    for col in out.columns:
        out[col] = out[col].fillna(train_stats[col]["median"])
    out = out.fillna(0)
    # Standardize using train-set mean/std
    for col in out.columns:
        s = train_stats[col]["std"]
        m = train_stats[col]["mean"]
        if s > 1e-8:
            out[col] = (out[col] - m) / s
        else:
            out[col] = 0.0
    return out, train_stats


def run_one_split_survival(panel, feature_cols, ho_all, split_i):
    """Run Cox + RSF for one split. Returns dict of results."""
    tr = panel[~panel["episode_id"].isin(ho_all)].reset_index(drop=True)
    te = panel[panel["episode_id"].isin(ho_all)].reset_index(drop=True)
    if len(te) < 10:
        return None

    train_X, stats = prepare_features(tr, feature_cols)
    test_X, _ = prepare_features(te, feature_cols, train_stats=stats)

    # Cox
    cox_train = train_X.copy()
    cox_train["T"] = tr[DURATION_COL].values
    cox_train["E"] = tr[EVENT_COL].values
    cox_coefs = {}
    try:
        cph = CoxPHFitter(penalizer=COX_PENALIZER)
        cph.fit(cox_train, duration_col="T", event_col="E")
        cox_pred = cph.predict_partial_hazard(test_X).values.flatten()
        cox_coefs = cph.params_.to_dict()
    except Exception:
        cox_pred = np.full(len(test_X), np.nan)

    # RSF
    events = tr[EVENT_COL].astype(bool).values
    durations = tr[DURATION_COL].values.astype(float)
    train_y = np.array(list(zip(events, durations)), dtype=[("event", bool), ("duration", float)])
    try:
        rsf = RandomSurvivalForest(n_estimators=100, max_depth=5, min_samples_leaf=10,
                                    random_state=DEFAULT_SEED, n_jobs=1)  # n_jobs=1 inside parallel
        rsf.fit(train_X.values, train_y)
        rsf_pred = rsf.predict(test_X.values)
    except Exception:
        rsf_pred = np.full(len(test_X), np.nan)

    te = te.copy()
    te["cox_pred"] = cox_pred
    te["rsf_pred"] = rsf_pred

    # Episode AUC
    ep = te.groupby(["episode_id", "source"]).agg(
        cox_mean=("cox_pred", "mean"), rsf_mean=("rsf_pred", "mean")).reset_index()
    ep["is_bubble"] = (ep["source"] == "bubble").astype(int)

    # Track which episodes were held out
    ho_bubbles = sorted(te[te["source"] == "bubble"]["episode_id"].unique())
    ho_nears = sorted(te[te["source"] == "near_bubble"]["episode_id"].unique())

    result = {
        "split": split_i,
        "ho_bubbles": "|".join(ho_bubbles),
        "ho_nears": "|".join(ho_nears),
        "n_ho_bubbles": len(ho_bubbles),
        "n_ho_nears": len(ho_nears),
    }
    for model, col in [("cox", "cox_mean"), ("rsf", "rsf_mean")]:
        v = ep[col].notna()
        if v.sum() >= 4 and ep.loc[v, "is_bubble"].nunique() >= 2:
            result[f"{model}_episode_auc"] = roc_auc_score(ep.loc[v, "is_bubble"], ep.loc[v, col])
        else:
            result[f"{model}_episode_auc"] = np.nan

    # Month AUC
    bub = te[te["source"] == "bubble"]
    if len(bub) >= 10 and bub[EVENT_COL].nunique() >= 2:
        for model, pc in [("cox", "cox_pred"), ("rsf", "rsf_pred")]:
            v = np.isfinite(bub[pc].values)
            result[f"{model}_month_auc"] = roc_auc_score(bub.loc[v, EVENT_COL], bub.loc[v, pc]) if v.sum() >= 10 else np.nan
    else:
        result["cox_month_auc"] = np.nan
        result["rsf_month_auc"] = np.nan

    # Calibration
    bub_all = te[te["source"] == "bubble"]
    near_all = te[te["source"] == "near_bubble"]
    for model in ["cox", "rsf"]:
        pc = f"{model}_pred"
        result[f"{model}_mean_bubble"] = bub_all[pc].mean() if len(bub_all) > 0 else np.nan
        result[f"{model}_mean_near"] = near_all[pc].mean() if len(near_all) > 0 else np.nan

    # Cox coefficients
    if cox_coefs:
        for k, v in cox_coefs.items():
            result[f"cox_beta_{k}"] = v

    return result


def run_one_split_classifier(ep_features, feature_cols, ho_all, split_i):
    """Run LR + RF classifier for one split on episode-level data."""
    tr = ep_features[~ep_features["episode_id"].isin(ho_all)].reset_index(drop=True)
    te = ep_features[ep_features["episode_id"].isin(ho_all)].reset_index(drop=True)

    # Handle missing columns (e.g. bond-CIV not in all episodes)
    available = [c for c in feature_cols if c in tr.columns]
    missing = [c for c in feature_cols if c not in tr.columns]
    X_tr = tr[available].fillna(0).copy()
    X_te = te[available].fillna(0).copy()
    for c in missing:
        X_tr[c] = 0.0
        X_te[c] = 0.0
    X_tr = X_tr[feature_cols].values
    X_te = X_te[feature_cols].values

    y_tr = tr["is_bubble"].values
    y_te = te["is_bubble"].values

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    row = {"split": split_i}
    try:
        lr = LogisticRegressionCV(cv=3, max_iter=1000, random_state=42)
        lr.fit(X_tr_s, y_tr)
        row["lr_auc"] = roc_auc_score(y_te, lr.predict_proba(X_te_s)[:, 1])
    except Exception:
        row["lr_auc"] = np.nan
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3,
                                     random_state=42, n_jobs=1)
        rf.fit(X_tr_s, y_tr)
        row["rf_auc"] = roc_auc_score(y_te, rf.predict_proba(X_te_s)[:, 1])
    except Exception:
        row["rf_auc"] = np.nan
    return row


def generate_splits(bubble_ids, near_ids, n_splits, seed, panel=None):
    """
    Generate non-repeating 80/20 holdout splits.
    If panel provided, stratify by leverage level (high vs low).
    """
    rng = np.random.RandomState(seed)

    if panel is not None and "med_leverage" in panel.columns:
        # Stratified by leverage: compute median leverage per episode
        ep_lev = panel.groupby("episode_id")["med_leverage"].median()
        lev_median = ep_lev.median()

        # Split each group by leverage
        hi_bub = [b for b in bubble_ids if ep_lev.get(b, 0) >= lev_median]
        lo_bub = [b for b in bubble_ids if ep_lev.get(b, 0) < lev_median]
        hi_near = [n for n in near_ids if ep_lev.get(n, 0) >= lev_median]
        lo_near = [n for n in near_ids if ep_lev.get(n, 0) < lev_median]

        n_ho_hi_bub = max(1, round(len(hi_bub) * 0.2))
        n_ho_lo_bub = max(1, round(len(lo_bub) * 0.2))
        n_ho_hi_near = max(1, round(len(hi_near) * 0.2))
        n_ho_lo_near = max(1, round(len(lo_near) * 0.2))

        seen = set()
        splits = []
        while len(splits) < n_splits:
            ho = []
            if hi_bub:
                ho.extend(rng.choice(hi_bub, size=min(n_ho_hi_bub, len(hi_bub)), replace=False))
            if lo_bub:
                ho.extend(rng.choice(lo_bub, size=min(n_ho_lo_bub, len(lo_bub)), replace=False))
            if hi_near:
                ho.extend(rng.choice(hi_near, size=min(n_ho_hi_near, len(hi_near)), replace=False))
            if lo_near:
                ho.extend(rng.choice(lo_near, size=min(n_ho_lo_near, len(lo_near)), replace=False))
            key = tuple(sorted(ho))
            if key not in seen:
                seen.add(key)
                splits.append(set(ho))
        return splits
    else:
        # Unstratified fallback
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


def main():
    n_splits = 100
    n_cores = os.cpu_count()
    print(f"Timing test: {n_splits} splits, {n_cores} CPU cores available")
    print("=" * 60)

    # Load data
    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    bubble_ids = sorted(panel[panel["source"] == "bubble"]["episode_id"].unique())
    near_ids = sorted(panel[panel["source"] == "near_bubble"]["episode_id"].unique())
    splits = generate_splits(bubble_ids, near_ids, n_splits, DEFAULT_SEED)

    # Episode-level data for classifiers
    all_feat = VOL_COLS + SEC_COLS + TEXT_CIV_COLS
    ep_features = panel.groupby(["episode_id", "source"])[all_feat].mean().reset_index()
    ep_features["is_bubble"] = (ep_features["source"] == "bubble").astype(int)

    # ── SEQUENTIAL ────────────────────────────────────────────────────
    print(f"\nSEQUENTIAL ({n_splits} splits)...")
    t0 = time.time()

    seq_results = {}
    for feat_name, features in FEATURE_SETS.items():
        surv = [run_one_split_survival(panel, features, s, i) for i, s in enumerate(splits)]
        surv = [r for r in surv if r is not None]
        clf = [run_one_split_classifier(ep_features, features, s, i) for i, s in enumerate(splits)]
        clf = [r for r in clf if r is not None]
        seq_results[feat_name] = (surv, clf)

    t_seq = time.time() - t0
    print(f"  Sequential time: {t_seq:.1f}s ({t_seq/n_splits:.2f}s per split)")
    print(f"  Projected 1000 splits: {t_seq * 10:.0f}s ({t_seq * 10 / 60:.1f} min)")

    # Print results
    for feat_name, (surv, clf) in seq_results.items():
        sdf = pd.DataFrame(surv)
        cdf = pd.DataFrame(clf)
        rsf_ep = sdf["rsf_episode_auc"].dropna().median()
        rsf_mo = sdf["rsf_month_auc"].dropna().median()
        lr_ep = cdf["lr_auc"].dropna().median() if len(cdf) > 0 else np.nan
        rf_ep = cdf["rf_auc"].dropna().median() if len(cdf) > 0 else np.nan
        print(f"  {feat_name:15s}: surv_ep={rsf_ep:.3f} surv_mo={rsf_mo:.3f} "
              f"lr={lr_ep:.3f} rf={rf_ep:.3f}")

    # ── PARALLEL ──────────────────────────────────────────────────────
    print(f"\nPARALLEL ({n_splits} splits, {n_cores} cores)...")
    t0 = time.time()

    par_results = {}
    for feat_name, features in FEATURE_SETS.items():
        surv = Parallel(n_jobs=-1, backend="loky")(
            delayed(run_one_split_survival)(panel, features, s, i)
            for i, s in enumerate(splits)
        )
        surv = [r for r in surv if r is not None]

        clf = Parallel(n_jobs=-1, backend="loky")(
            delayed(run_one_split_classifier)(ep_features, features, s, i)
            for i, s in enumerate(splits)
        )
        clf = [r for r in clf if r is not None]
        par_results[feat_name] = (surv, clf)

    t_par = time.time() - t0
    speedup = t_seq / t_par if t_par > 0 else 0
    print(f"  Parallel time: {t_par:.1f}s ({t_par/n_splits:.2f}s per split)")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Projected 1000 splits: {t_par * 10:.0f}s ({t_par * 10 / 60:.1f} min)")

    # Verify results match
    print(f"\nVERIFICATION (results should match):")
    for feat_name in FEATURE_SETS:
        s_surv = pd.DataFrame(seq_results[feat_name][0])
        p_surv = pd.DataFrame(par_results[feat_name][0])
        s_ep = s_surv["rsf_episode_auc"].dropna().median()
        p_ep = p_surv["rsf_episode_auc"].dropna().median()
        match = "MATCH" if abs(s_ep - p_ep) < 0.001 else f"DIFF ({s_ep:.3f} vs {p_ep:.3f})"
        print(f"  {feat_name:15s}: seq={s_ep:.3f} par={p_ep:.3f} {match}")


if __name__ == "__main__":
    main()
