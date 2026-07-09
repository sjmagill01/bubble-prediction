"""
11b_permutation_cv_comparison.py — Compare fixed-C vs CV'd-C permutation test.

Runs 200 splits x 500 perms with both LogisticRegression(C=1.0) and
LogisticRegressionCV(cv=3) on the "all" feature set, to verify that
the fixed-C optimization doesn't change the p-value.
"""
import warnings
warnings.filterwarnings("ignore")

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from config import PANELS_DIR, RESULTS_DIR, DEFAULT_SEED, ALL_FEATURE_COLS, FEATURE_SETS

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "timing_test", Path(__file__).resolve().parent / "07b_timing_test.py")
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)
generate_splits = tt.generate_splits


def _run_one_split(ep_np, feature_idx, is_bubble, episode_ids,
                   split_ho, n_perms, base_seed, use_cv):
    """Run one split with either fixed-C or CV'd-C LR."""
    ho_set = set(split_ho)
    mask_ho = np.array([eid in ho_set for eid in episode_ids])
    mask_tr = ~mask_ho

    X_tr = ep_np[mask_tr][:, feature_idx]
    X_te = ep_np[mask_ho][:, feature_idx]
    y_tr_real = is_bubble[mask_tr]
    y_te = is_bubble[mask_ho]

    if len(np.unique(y_tr_real)) < 2 or len(np.unique(y_te)) < 2:
        return None

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    def make_lr():
        if use_cv:
            return LogisticRegressionCV(cv=3, max_iter=1000, random_state=42)
        else:
            return LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)

    # Observed
    try:
        lr = make_lr()
        lr.fit(X_tr_s, y_tr_real)
        obs_auc = roc_auc_score(y_te, lr.predict_proba(X_te_s)[:, 1])
    except Exception:
        return None

    # Null
    null_aucs = []
    rng = np.random.RandomState(base_seed)
    for _ in range(n_perms):
        y_tr_perm = rng.permutation(y_tr_real)
        if len(np.unique(y_tr_perm)) < 2:
            continue
        try:
            lr_perm = make_lr()
            lr_perm.fit(X_tr_s, y_tr_perm)
            null_aucs.append(roc_auc_score(y_te, lr_perm.predict_proba(X_te_s)[:, 1]))
        except Exception:
            continue

    return (obs_auc, null_aucs)


def run_test(ep_np, feature_idx, is_bubble, episode_ids, splits_tuples,
             n_perms, use_cv, label):
    t0 = time.time()
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_run_one_split)(
            ep_np, feature_idx, is_bubble, episode_ids,
            sp, n_perms, DEFAULT_SEED + i * 10000, use_cv
        )
        for i, sp in enumerate(splits_tuples)
    )

    observed = []
    null_aucs = []
    for r in results:
        if r is None:
            continue
        observed.append(r[0])
        null_aucs.extend(r[1])

    obs_median = np.median(observed)
    p_value = (np.array(null_aucs) >= obs_median).mean()
    elapsed = time.time() - t0

    print(f"\n  [{label}] ({elapsed:.0f}s)")
    print(f"    Observed median: {obs_median:.4f} [{np.percentile(observed, 2.5):.4f}, {np.percentile(observed, 97.5):.4f}]")
    print(f"    Null median: {np.median(null_aucs):.4f}, 95th: {np.percentile(null_aucs, 95):.4f}")
    print(f"    P-value: {p_value:.6f}")
    print(f"    N observed: {len(observed)}, N null: {len(null_aucs):,}")

    return {"label": label, "obs_median": obs_median, "p_value": p_value,
            "null_median": np.median(null_aucs), "null_95": np.percentile(null_aucs, 95),
            "n_obs": len(observed), "n_null": len(null_aucs), "time_s": elapsed}


def main():
    N_SPLITS = 200
    N_PERMS = 500
    print(f"CV vs Fixed-C comparison: {N_SPLITS} splits x {N_PERMS} perms, feature set = all")
    print("=" * 70)

    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    all_feat = [c for c in ALL_FEATURE_COLS if c in panel.columns]
    ep_features = panel.groupby(["episode_id", "source"])[all_feat].mean().reset_index()
    ep_features["is_bubble"] = (ep_features["source"] == "bubble").astype(int)

    bubble_ids = sorted(ep_features[ep_features["is_bubble"] == 1]["episode_id"].unique())
    near_ids = sorted(ep_features[ep_features["is_bubble"] == 0]["episode_id"].unique())

    splits = generate_splits(bubble_ids, near_ids, N_SPLITS, DEFAULT_SEED, panel=panel)
    splits_tuples = [tuple(sorted(s)) for s in splits]

    features = FEATURE_SETS["all"]
    features = [f for f in features if f in ep_features.columns]
    feature_idx = [all_feat.index(f) for f in features]
    ep_np = ep_features[all_feat].fillna(0).values
    episode_ids = ep_features["episode_id"].values
    is_bubble = ep_features["is_bubble"].values

    print(f"  {len(bubble_ids)} bubbles + {len(near_ids)} near-bubbles, {len(features)} features")

    # Run fixed-C first (fast)
    r_fixed = run_test(ep_np, feature_idx, is_bubble, episode_ids, splits_tuples,
                       N_PERMS, use_cv=False, label="Fixed C=1.0")

    # Run CV'd-C (slow)
    r_cv = run_test(ep_np, feature_idx, is_bubble, episode_ids, splits_tuples,
                    N_PERMS, use_cv=True, label="LogisticRegressionCV")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'':20s} {'Fixed C=1.0':>15s} {'CV (cv=3)':>15s}")
    print(f"{'-' * 50}")
    print(f"{'Observed median':20s} {r_fixed['obs_median']:15.4f} {r_cv['obs_median']:15.4f}")
    print(f"{'Null median':20s} {r_fixed['null_median']:15.4f} {r_cv['null_median']:15.4f}")
    print(f"{'Null 95th':20s} {r_fixed['null_95']:15.4f} {r_cv['null_95']:15.4f}")
    print(f"{'P-value':20s} {r_fixed['p_value']:15.6f} {r_cv['p_value']:15.6f}")
    print(f"{'Time (s)':20s} {r_fixed['time_s']:15.0f} {r_cv['time_s']:15.0f}")
    print(f"{'Speedup':20s} {'':15s} {r_cv['time_s']/r_fixed['time_s']:14.1f}x")

    pd.DataFrame([r_fixed, r_cv]).to_csv(
        RESULTS_DIR / "permutation_cv_comparison.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'permutation_cv_comparison.csv'}")


if __name__ == "__main__":
    main()
