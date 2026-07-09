"""
11_permutation_test.py — Permutation test for episode-level AUC.

Shuffles bubble/near-bubble labels and re-runs the LR classifier
to build a null distribution. Tests whether the observed AUC of 0.844
is statistically distinguishable from what you'd get by chance given
the class balance (27/42) and feature dimensionality (23).

Uses the same leverage-stratified 80/20 holdout and standardization
pipeline as 07c_run_parallel.py.

Usage:
    python 11_permutation_test.py                    # defaults: 200 splits x 500 perms
    python 11_permutation_test.py --splits 1000 --perms 1000
    python 11_permutation_test.py --splits 10000 --perms 500
"""
import warnings
warnings.filterwarnings("ignore")

import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from config import PANELS_DIR, RESULTS_DIR, DEFAULT_SEED, ALL_FEATURE_COLS, FEATURE_SETS

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import leverage-stratified split generator
import importlib.util
spec = importlib.util.spec_from_file_location(
    "timing_test", Path(__file__).resolve().parent / "07b_timing_test.py")
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)
generate_splits = tt.generate_splits


def _run_one_split(ep_features_np, feature_idx, is_bubble, episode_ids,
                   split_ho, n_perms, base_seed):
    """
    Run observed + all permutations for ONE split.
    Returns (observed_auc, [null_auc_1, ..., null_auc_n_perms]).

    Takes numpy arrays (not DataFrames) to avoid serialization overhead.

    Uses fixed-C LogisticRegression (not LogisticRegressionCV) for both
    observed and null. The permutation test asks whether the *signal* is
    real, not what the *exact* AUC is — both sides must use the same model,
    but CV to pick C is wasted work (~30x overhead) that doesn't change
    the permutation p-value.
    """
    ho_set = set(split_ho)
    mask_ho = np.array([eid in ho_set for eid in episode_ids])
    mask_tr = ~mask_ho

    X_tr = ep_features_np[mask_tr][:, feature_idx]
    X_te = ep_features_np[mask_ho][:, feature_idx]
    y_tr_real = is_bubble[mask_tr]
    y_te = is_bubble[mask_ho]

    # Need both classes in train and test
    if len(np.unique(y_tr_real)) < 2 or len(np.unique(y_te)) < 2:
        return None

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # --- Observed ---
    try:
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
        lr.fit(X_tr_s, y_tr_real)
        obs_auc = roc_auc_score(y_te, lr.predict_proba(X_te_s)[:, 1])
    except Exception:
        return None

    # --- Null (shuffle training labels) ---
    null_aucs = []
    rng = np.random.RandomState(base_seed)
    for _ in range(n_perms):
        y_tr_perm = rng.permutation(y_tr_real)
        if len(np.unique(y_tr_perm)) < 2:
            continue
        try:
            lr.fit(X_tr_s, y_tr_perm)  # reuse same LR object
            null_auc = roc_auc_score(y_te, lr.predict_proba(X_te_s)[:, 1])
            null_aucs.append(null_auc)
        except Exception:
            continue

    return (obs_auc, null_aucs)


def main():
    parser = argparse.ArgumentParser(description="Permutation test for episode-level AUC")
    parser.add_argument("--splits", type=int, default=200, help="Number of CV splits")
    parser.add_argument("--perms", type=int, default=500, help="Permutations per split")
    parser.add_argument("--feature-sets", nargs="+", default=["vol_only", "all"],
                        help="Feature sets to test")
    args = parser.parse_args()

    N_CV_SPLITS = args.splits
    N_PERMUTATIONS = args.perms

    total_fits = N_CV_SPLITS * (1 + N_PERMUTATIONS) * len(args.feature_sets)
    print(f"Permutation test: {N_CV_SPLITS} CV splits x {N_PERMUTATIONS} permutations")
    print(f"  Total LR fits: {total_fits:,} across {len(args.feature_sets)} feature set(s)")
    print("=" * 70)

    # Load data
    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    # Episode-level features (1 row per episode)
    all_feat = [c for c in ALL_FEATURE_COLS if c in panel.columns]
    ep_features = panel.groupby(["episode_id", "source"])[all_feat].mean().reset_index()
    ep_features["is_bubble"] = (ep_features["source"] == "bubble").astype(int)

    bubble_ids = sorted(ep_features[ep_features["is_bubble"] == 1]["episode_id"].unique())
    near_ids = sorted(ep_features[ep_features["is_bubble"] == 0]["episode_id"].unique())
    print(f"  {len(bubble_ids)} bubbles + {len(near_ids)} near-bubbles = {len(ep_features)} episodes")

    # Generate splits (leverage-stratified)
    t_split = time.time()
    splits = generate_splits(bubble_ids, near_ids, N_CV_SPLITS, DEFAULT_SEED, panel=panel)
    print(f"  {len(splits)} leverage-stratified splits generated ({time.time() - t_split:.1f}s)")

    # Pre-convert to numpy for fast serialization
    episode_ids = ep_features["episode_id"].values
    is_bubble = ep_features["is_bubble"].values

    # Convert splits from sets to sorted tuples (hashable, smaller serialization)
    splits_tuples = [tuple(sorted(s)) for s in splits]

    # Test each feature set
    for feat_name in args.feature_sets:
        features = FEATURE_SETS[feat_name]
        features = [f for f in features if f in ep_features.columns]
        feature_idx = [all_feat.index(f) for f in features]
        print(f"\n  Feature set: {feat_name} ({len(features)} features)")
        print(f"  {'-' * 50}")

        # Pre-extract numpy feature matrix (fillna 0)
        ep_np = ep_features[all_feat].fillna(0).values

        t0 = time.time()

        # Run all splits in parallel — each split does observed + all permutations
        results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(_run_one_split)(
                ep_np, feature_idx, is_bubble, episode_ids,
                sp, N_PERMUTATIONS, DEFAULT_SEED + i * 10000
            )
            for i, sp in enumerate(splits_tuples)
        )

        # Unpack
        observed = []
        null_aucs = []
        for r in results:
            if r is None:
                continue
            obs_auc, nulls = r
            observed.append(obs_auc)
            null_aucs.extend(nulls)

        obs_median = np.median(observed)
        null_median = np.median(null_aucs)
        null_95 = np.percentile(null_aucs, 95)
        null_99 = np.percentile(null_aucs, 99)

        # P-value: fraction of null AUCs >= observed median
        p_value = (np.array(null_aucs) >= obs_median).mean()

        elapsed = time.time() - t0

        print(f"\n  Observed ({len(observed)} splits):")
        print(f"    Median AUC: {obs_median:.4f} "
              f"[{np.percentile(observed, 2.5):.4f}, {np.percentile(observed, 97.5):.4f}]")
        print(f"  Null distribution ({len(null_aucs):,} values):")
        print(f"    Median: {null_median:.4f}")
        print(f"    95th percentile: {null_95:.4f}")
        print(f"    99th percentile: {null_99:.4f}")
        print(f"  P-value (null >= observed): {p_value:.6f}")
        print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Save results
        results_dict = {
            "feature_set": feat_name,
            "n_cv_splits": N_CV_SPLITS,
            "n_permutations": N_PERMUTATIONS,
            "observed_median": obs_median,
            "observed_2.5pct": np.percentile(observed, 2.5),
            "observed_97.5pct": np.percentile(observed, 97.5),
            "null_median": null_median,
            "null_95pct": null_95,
            "null_99pct": null_99,
            "p_value": p_value,
            "n_observed": len(observed),
            "n_null": len(null_aucs),
        }

        # Save summary CSV
        pd.DataFrame([results_dict]).to_csv(
            RESULTS_DIR / f"permutation_test_{feat_name}.csv", index=False)

        # Save full distributions (compressed numpy)
        np.savez_compressed(
            RESULTS_DIR / f"permutation_test_{feat_name}.npz",
            observed=np.array(observed),
            null=np.array(null_aucs))

        # Save per-split detail (observed + per-split null stats)
        split_records = []
        idx = 0
        for i, r in enumerate(results):
            if r is None:
                continue
            obs_auc, nulls = r
            split_records.append({
                "split_idx": i,
                "observed_auc": obs_auc,
                "n_null": len(nulls),
                "null_median": np.median(nulls) if nulls else np.nan,
                "null_mean": np.mean(nulls) if nulls else np.nan,
                "null_std": np.std(nulls) if nulls else np.nan,
                "null_5pct": np.percentile(nulls, 5) if nulls else np.nan,
                "null_95pct": np.percentile(nulls, 95) if nulls else np.nan,
                "p_value_this_split": np.mean(np.array(nulls) >= obs_auc) if nulls else np.nan,
            })
        pd.DataFrame(split_records).to_parquet(
            RESULTS_DIR / f"permutation_test_{feat_name}_per_split.parquet", index=False)

        print(f"  Saved: permutation_test_{feat_name}.csv, .npz, _per_split.parquet")

    print(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
