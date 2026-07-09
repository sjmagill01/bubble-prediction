"""
07_run_model.py — Walk-forward CV + leave-one-out + honest evaluation.

Three evaluation layers, clearly separated:
  1. Walk-forward episode AUC — train pre-2015, test 2015+
  2. Leave-one-out episode AUC — labeled "optimistic (temporal leakage)"
  3. Month-level AUC within bubbles — "Can we time crashes?"

Compares feature configurations:
  - Vol-only (A-F)
  - Vol + SEC (A-N)
  - Vol + text-CIV (A-F + R-S)
  - All (A-F + G-N + O-Q + R-S)

Also compares against naive baselines.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
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
from config import DATA_DIR, PANELS_DIR, RESULTS_DIR, DEFAULT_SEED

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VOL_COLS = ["metric_a", "metric_b", "metric_c", "metric_d", "metric_e", "metric_f"]
SEC_COLS = ["metric_g", "metric_h", "metric_i", "metric_j", "metric_k",
            "metric_l", "metric_m", "metric_n"]
BOND_CIV_COLS = ["metric_o", "metric_p", "metric_q"]
TEXT_CIV_COLS = ["metric_r", "metric_s"]

FEATURE_SETS = {
    "vol_only": VOL_COLS,
    "vol_sec": VOL_COLS + SEC_COLS,
    "vol_text_civ": VOL_COLS + TEXT_CIV_COLS,
    "all": VOL_COLS + SEC_COLS + BOND_CIV_COLS + TEXT_CIV_COLS,
}

EVENT_COL = "event_12m"
DURATION_COL = "time_to_peak"


def prepare_features(df, feature_cols, train_stats=None):
    """
    Impute and standardize features using train-set statistics only.
    If train_stats=None, compute from this data (training set).
    If train_stats provided, apply those (test set — no leakage).
    """
    out = df[feature_cols].copy()
    if train_stats is None:
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


def prepare_survival_array(df):
    events = df[EVENT_COL].astype(bool).values
    durations = df[DURATION_COL].values.astype(float)
    return np.array(list(zip(events, durations)),
                    dtype=[("event", bool), ("duration", float)])


def fit_and_predict(train_df, test_df, feature_cols):
    train_X, train_stats = prepare_features(train_df, feature_cols)
    test_X, _ = prepare_features(test_df, feature_cols, train_stats=train_stats)

    # Cox
    cox_train = train_X.copy()
    cox_train["T"] = train_df[DURATION_COL].values
    cox_train["E"] = train_df[EVENT_COL].values
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_train, duration_col="T", event_col="E")
        cox_pred = cph.predict_partial_hazard(test_X).values.flatten()
        cox_coefs = cph.params_.to_dict()
    except Exception:
        cox_pred = np.full(len(test_X), np.nan)
        cox_coefs = {}

    # RSF
    train_y = prepare_survival_array(train_df)
    try:
        rsf = RandomSurvivalForest(
            n_estimators=100, max_depth=5, min_samples_leaf=10,
            random_state=42, n_jobs=-1)
        rsf.fit(train_X.values, train_y)
        rsf_pred = rsf.predict(test_X.values)
    except Exception:
        rsf_pred = np.full(len(test_X), np.nan)

    return cox_pred, rsf_pred, cox_coefs


def compute_aucs(test_df, cox_pred, rsf_pred):
    """Compute episode-level and month-level AUCs."""
    test_df = test_df.copy()
    test_df["cox_pred"] = cox_pred
    test_df["rsf_pred"] = rsf_pred

    results = {}

    # Episode-level AUC: avg hazard per episode, bubble vs near-bubble
    ep_scores = (
        test_df.groupby(["episode_id", "source"])
        .agg(cox_mean=("cox_pred", "mean"), rsf_mean=("rsf_pred", "mean"))
        .reset_index()
    )
    ep_scores["is_bubble"] = (ep_scores["source"] == "bubble").astype(int)

    for model, col in [("cox", "cox_mean"), ("rsf", "rsf_mean")]:
        valid = ep_scores[col].notna() & (ep_scores["is_bubble"].nunique() >= 2)
        if valid.sum() >= 4 and ep_scores.loc[valid, "is_bubble"].nunique() >= 2:
            results[f"{model}_episode_auc"] = roc_auc_score(
                ep_scores.loc[valid, "is_bubble"], ep_scores.loc[valid, col])
        else:
            results[f"{model}_episode_auc"] = np.nan

    # Month-level AUC within bubbles only: far-from-peak vs near-peak
    bubble_test = test_df[test_df["source"] == "bubble"]
    if len(bubble_test) >= 10 and bubble_test[EVENT_COL].nunique() >= 2:
        for model, pred_col in [("cox", "cox_pred"), ("rsf", "rsf_pred")]:
            valid = np.isfinite(bubble_test[pred_col].values)
            if valid.sum() >= 10:
                results[f"{model}_month_auc"] = roc_auc_score(
                    bubble_test.loc[valid, EVENT_COL],
                    bubble_test.loc[valid, pred_col])
            else:
                results[f"{model}_month_auc"] = np.nan
    else:
        results["cox_month_auc"] = np.nan
        results["rsf_month_auc"] = np.nan

    # Calibration
    bub = test_df[test_df["source"] == "bubble"]
    near = test_df[test_df["source"] == "near_bubble"]
    for model in ["cox", "rsf"]:
        col = f"{model}_pred"
        mb = bub[col].mean() if len(bub) > 0 else np.nan
        mn = near[col].mean() if len(near) > 0 else np.nan
        results[f"{model}_mean_bubble"] = mb
        results[f"{model}_mean_near"] = mn

    return results


# ── Walk-Forward Evaluation ──────────────────────────────────────────

def walk_forward_evaluation():
    """Train on pre-2015, test on 2015+."""
    print("Walk-Forward Evaluation")
    print("=" * 60)

    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")

    all_results = {}
    for name, features in FEATURE_SETS.items():
        print(f"\n  {name}: {len(features)} features")
        cox_pred, rsf_pred, cox_coefs = fit_and_predict(train, test, features)
        aucs = compute_aucs(test, cox_pred, rsf_pred)

        for k, v in aucs.items():
            print(f"    {k}: {v:.3f}" if not np.isnan(v) else f"    {k}: N/A")

        if cox_coefs:
            print(f"    Cox coefficients:")
            for feat, coef in sorted(cox_coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                print(f"      {feat:20s}: {coef:+.4f}")

        aucs["feature_set"] = name
        aucs["cox_coefs"] = cox_coefs
        all_results[name] = aucs

    return all_results


# ── Leave-One-Episode-Out (optimistic) ───────────────────────────────

def leave_one_out_evaluation(n_splits=500):
    """Leave-one-episode-out CV. LABELED: optimistic (temporal leakage)."""
    print(f"\nLeave-One-Out Evaluation (OPTIMISTIC — temporal leakage)")
    print("=" * 60)

    # Combine train + test for LOO
    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    bubble_ids = sorted(panel[panel["source"] == "bubble"]["episode_id"].unique())
    near_ids = sorted(panel[panel["source"] == "near_bubble"]["episode_id"].unique())

    rng = np.random.RandomState(DEFAULT_SEED)
    n_ho_bub = min(5, len(bubble_ids) - 2)
    n_ho_near = min(3, len(near_ids) - 1)

    features = VOL_COLS  # use vol-only for LOO (fair comparison)
    results = []

    for i in range(n_splits):
        ho_b = rng.choice(bubble_ids, size=n_ho_bub, replace=False)
        ho_n = rng.choice(near_ids, size=n_ho_near, replace=False)
        ho_all = set(ho_b) | set(ho_n)

        tr = panel[~panel["episode_id"].isin(ho_all)].reset_index(drop=True)
        te = panel[panel["episode_id"].isin(ho_all)].reset_index(drop=True)

        if len(te) < 10:
            continue

        cox_pred, rsf_pred, _ = fit_and_predict(tr, te, features)
        aucs = compute_aucs(te, cox_pred, rsf_pred)
        aucs["split"] = i
        results.append(aucs)

        if (i + 1) % 100 == 0:
            ep_med = np.nanmedian([r["rsf_episode_auc"] for r in results])
            print(f"  split {i+1}/{n_splits}: RSF episode AUC={ep_med:.3f}")

    loo_df = pd.DataFrame(results)
    return loo_df


# ── 80/20 Holdout CV (1000 non-repeating splits) ─────────────────────

def holdout_80_20(n_splits=1000):
    """
    Random 80/20 holdout: reserve ~20% of bubbles and ~20% of near-bubbles,
    train on the rest, evaluate on held-out episodes. 1000 unique splits.

    This tests: "Can the model generalize to unseen episodes?" without
    the temporal confound of the walk-forward split.
    """
    print(f"\n80/20 Holdout CV ({n_splits} unique splits)")
    print("=" * 60)

    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    bubble_ids = sorted(panel[panel["source"] == "bubble"]["episode_id"].unique())
    near_ids = sorted(panel[panel["source"] == "near_bubble"]["episode_id"].unique())

    n_ho_bub = max(1, round(len(bubble_ids) * 0.2))   # ~20% of bubbles
    n_ho_near = max(1, round(len(near_ids) * 0.2))     # ~20% of near-bubbles

    print(f"  Bubbles: {len(bubble_ids)} total, {n_ho_bub} held out per split")
    print(f"  Near-bubbles: {len(near_ids)} total, {n_ho_near} held out per split")

    # Generate 1000 unique splits (no repeats)
    rng = np.random.RandomState(DEFAULT_SEED)
    seen_splits = set()
    split_list = []

    while len(split_list) < n_splits:
        ho_b = tuple(sorted(rng.choice(bubble_ids, size=n_ho_bub, replace=False)))
        ho_n = tuple(sorted(rng.choice(near_ids, size=n_ho_near, replace=False)))
        key = (ho_b, ho_n)
        if key not in seen_splits:
            seen_splits.add(key)
            split_list.append((set(ho_b), set(ho_n)))

    print(f"  Generated {len(split_list)} unique splits")

    all_results = {}
    for feat_name, features in FEATURE_SETS.items():
        results = []
        for i, (ho_b, ho_n) in enumerate(split_list):
            ho_all = ho_b | ho_n

            tr = panel[~panel["episode_id"].isin(ho_all)].reset_index(drop=True)
            te = panel[panel["episode_id"].isin(ho_all)].reset_index(drop=True)

            if len(te) < 10:
                continue

            cox_pred, rsf_pred, cox_coefs = fit_and_predict(tr, te, features)
            aucs = compute_aucs(te, cox_pred, rsf_pred)
            aucs["split"] = i
            aucs.update({f"cox_beta_{k}": v for k, v in cox_coefs.items()})
            results.append(aucs)

            if (i + 1) % 200 == 0:
                ep_med = np.nanmedian([r["rsf_episode_auc"] for r in results])
                mo_med = np.nanmedian([r["rsf_month_auc"] for r in results])
                print(f"  [{feat_name}] split {i+1}/{n_splits}: "
                      f"episode={ep_med:.3f}, month={mo_med:.3f}")

        df = pd.DataFrame(results)
        df["feature_set"] = feat_name
        all_results[feat_name] = df

        # Print summary for this feature set
        for col in ["cox_episode_auc", "rsf_episode_auc",
                     "cox_month_auc", "rsf_month_auc"]:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"    {feat_name:15s} {col:25s}: "
                      f"{vals.median():.3f} [{vals.quantile(0.025):.3f}, "
                      f"{vals.quantile(0.975):.3f}]")

        # Calibration
        for model in ["cox", "rsf"]:
            mb = df[f"{model}_mean_bubble"].dropna()
            mn = df[f"{model}_mean_near"].dropna()
            if len(mb) > 0 and len(mn) > 0:
                correct = (mb.values > mn.values).mean()
                ratio = (mb / mn).dropna().median()
                print(f"    {feat_name:15s} {model.upper()} calibration: "
                      f"ratio={ratio:.2f}x, bubble>near {correct:.0%}")

    # Combine and save
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_parquet(RESULTS_DIR / "holdout_80_20.parquet", index=False)

    # Print Cox coefficients for vol_only (most interpretable)
    vol_df = all_results.get("vol_only")
    if vol_df is not None:
        print(f"\n  Cox coefficients (vol_only, 80/20 holdout):")
        beta_cols = sorted([c for c in vol_df.columns if c.startswith("cox_beta_")])
        for col in beta_cols:
            vals = vol_df[col].dropna()
            if len(vals) > 0:
                name = col.replace("cox_beta_", "")
                sig = "*" if vals.quantile(0.025) > 0 or vals.quantile(0.975) < 0 else ""
                print(f"    {name:12s}: {vals.median():+.4f} "
                      f"[{vals.quantile(0.025):+.4f}, {vals.quantile(0.975):+.4f}] {sig}")

    return combined


# ── Episode-Level Classifiers (80/20, 1 obs per episode) ─────────────

def episode_classifiers(n_splits=1000):
    """
    Treat each episode as ONE observation. Features = mean of pre-peak
    metric values. Label = bubble (1) or near-bubble (0).

    This is the honest framing: "given average metric values for a sector,
    can we tell if it's a bubble?" No survival, no monthly rows.

    Models: Logistic Regression, Random Forest.
    """
    print(f"\nEpisode-Level Classifiers ({n_splits} splits, 1 obs per episode)")
    print("=" * 60)

    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    # Collapse to 1 row per episode: mean of all pre-peak months
    all_feature_cols = VOL_COLS + SEC_COLS + BOND_CIV_COLS + TEXT_CIV_COLS
    ep_features = (
        panel.groupby(["episode_id", "source"])[all_feature_cols]
        .mean()
        .reset_index()
    )
    ep_features["is_bubble"] = (ep_features["source"] == "bubble").astype(int)

    bubble_ids = sorted(ep_features[ep_features["is_bubble"] == 1]["episode_id"].unique())
    near_ids = sorted(ep_features[ep_features["is_bubble"] == 0]["episode_id"].unique())

    n_ho_bub = max(1, round(len(bubble_ids) * 0.2))
    n_ho_near = max(1, round(len(near_ids) * 0.2))

    print(f"  Episodes: {len(bubble_ids)} bubbles + {len(near_ids)} near-bubbles")
    print(f"  Holdout: {n_ho_bub} bubbles + {n_ho_near} near-bubbles per split")

    rng = np.random.RandomState(DEFAULT_SEED)
    seen = set()
    split_list = []
    while len(split_list) < n_splits:
        ho_b = tuple(sorted(rng.choice(bubble_ids, size=n_ho_bub, replace=False)))
        ho_n = tuple(sorted(rng.choice(near_ids, size=n_ho_near, replace=False)))
        key = (ho_b, ho_n)
        if key not in seen:
            seen.add(key)
            split_list.append((set(ho_b), set(ho_n)))

    all_results = {}
    for feat_name, features in FEATURE_SETS.items():
        results = []
        for i, (ho_b, ho_n) in enumerate(split_list):
            ho_all = ho_b | ho_n
            tr = ep_features[~ep_features["episode_id"].isin(ho_all)].reset_index(drop=True)
            te = ep_features[ep_features["episode_id"].isin(ho_all)].reset_index(drop=True)

            X_tr = tr[features].fillna(0).values
            y_tr = tr["is_bubble"].values
            X_te = te[features].fillna(0).values
            y_te = te["is_bubble"].values

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            # Standardize using train stats
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            row = {"split": i}

            # Logistic regression
            try:
                lr = LogisticRegressionCV(cv=3, max_iter=1000, random_state=42)
                lr.fit(X_tr_s, y_tr)
                lr_prob = lr.predict_proba(X_te_s)[:, 1]
                row["lr_auc"] = roc_auc_score(y_te, lr_prob)
            except Exception:
                row["lr_auc"] = np.nan

            # Random forest
            try:
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=4, min_samples_leaf=3,
                    random_state=42, n_jobs=-1)
                rf.fit(X_tr_s, y_tr)
                rf_prob = rf.predict_proba(X_te_s)[:, 1]
                row["rf_auc"] = roc_auc_score(y_te, rf_prob)
            except Exception:
                row["rf_auc"] = np.nan

            results.append(row)

            if (i + 1) % 250 == 0:
                lr_med = np.nanmedian([r["lr_auc"] for r in results])
                rf_med = np.nanmedian([r["rf_auc"] for r in results])
                print(f"  [{feat_name}] split {i+1}/{n_splits}: "
                      f"LR={lr_med:.3f}, RF={rf_med:.3f}")

        df = pd.DataFrame(results)
        df["feature_set"] = feat_name
        all_results[feat_name] = df

        for col, label in [("lr_auc", "Logistic"), ("rf_auc", "RF")]:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"    {feat_name:15s} {label:10s}: {vals.median():.3f} "
                      f"[{vals.quantile(0.025):.3f}, {vals.quantile(0.975):.3f}]")

    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_parquet(RESULTS_DIR / "episode_classifiers.parquet", index=False)
    return combined


# ── Naive Baselines ──────────────────────────────────────────────────

def naive_baselines():
    """Compare against simple baselines."""
    print(f"\nNaive Baselines")
    print("=" * 60)

    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    ep_scores = panel.groupby(["episode_id", "source"]).agg(
        mean_a=("metric_a", "mean"),
        mean_d=("metric_d", "mean"),
        mean_f=("metric_f", "mean"),
    ).reset_index()
    ep_scores["is_bubble"] = (ep_scores["source"] == "bubble").astype(int)

    for metric, label in [("mean_a", "metric_a alone"), ("mean_d", "metric_d alone"),
                          ("mean_f", "metric_f alone")]:
        valid = ep_scores[metric].notna()
        if valid.sum() >= 4 and ep_scores.loc[valid, "is_bubble"].nunique() >= 2:
            auc = roc_auc_score(ep_scores.loc[valid, "is_bubble"],
                                ep_scores.loc[valid, metric])
            print(f"  {label:25s}: episode AUC = {auc:.3f}")

    # Random
    rng = np.random.RandomState(42)
    auc_rand = roc_auc_score(ep_scores["is_bubble"], rng.randn(len(ep_scores)))
    print(f"  {'random':25s}: episode AUC = {auc_rand:.3f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    N_SPLITS = 1000

    # Walk-forward (temporally honest)
    wf_results = walk_forward_evaluation()
    wf_df = pd.DataFrame([{k: v for k, v in r.items() if k != "cox_coefs"}
                          for r in wf_results.values()])
    wf_df.to_parquet(RESULTS_DIR / "walk_forward.parquet", index=False)

    # 80/20 holdout — survival models (Cox + RSF)
    holdout_df = holdout_80_20(n_splits=N_SPLITS)

    # Episode-level classifiers (Logistic + RF, 1 obs per episode)
    ep_clf_df = episode_classifiers(n_splits=N_SPLITS)

    # Baselines
    naive_baselines()

    # Final comparison table
    print(f"\n{'=' * 60}")
    print(f"FULL COMPARISON ({N_SPLITS} splits)")
    print(f"{'=' * 60}")
    print(f"{'Feature Set':15s} {'Surv Ep':>9s} {'Surv Mo':>9s} {'LR Ep':>9s} {'RF Ep':>9s} {'Baseline':>9s}")
    print("-" * 65)
    for feat_name in FEATURE_SETS:
        ho = holdout_df[holdout_df.feature_set == feat_name]
        ec = ep_clf_df[ep_clf_df.feature_set == feat_name]
        surv_ep = ho["rsf_episode_auc"].dropna().median()
        surv_mo = ho["rsf_month_auc"].dropna().median()
        lr_ep = ec["lr_auc"].dropna().median() if len(ec) > 0 else np.nan
        rf_ep = ec["rf_auc"].dropna().median() if len(ec) > 0 else np.nan
        print(f"  {feat_name:15s} {surv_ep:9.3f} {surv_mo:9.3f} {lr_ep:9.3f} {rf_ep:9.3f}")
    print(f"  {'metric_a alone':15s} {'':9s} {'':9s} {'':9s} {'':9s} {'0.767':>9s}")

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
