"""
07c_run_parallel.py — Full 1000-split parallel 80/20 holdout.

Based on timing test: ~28 min expected (3.3x speedup over sequential).
"""
import warnings
warnings.filterwarnings("ignore")

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from config import PANELS_DIR, RESULTS_DIR, DEFAULT_SEED

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import the split runners from timing test
import importlib.util
spec = importlib.util.spec_from_file_location(
    "timing_test", Path(__file__).resolve().parent / "07b_timing_test.py")
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)

run_one_split_survival = tt.run_one_split_survival
run_one_split_classifier = tt.run_one_split_classifier
generate_splits = tt.generate_splits
VOL_COLS = tt.VOL_COLS
SEC_COLS = tt.SEC_COLS
TEXT_CIV_COLS = tt.TEXT_CIV_COLS
ALL_COLS = tt.ALL_COLS
FEATURE_SETS = tt.FEATURE_SETS


def main():
    N_SPLITS = 1000
    print(f"Full parallel 80/20 holdout: {N_SPLITS} splits")
    print("=" * 60)

    # Load data
    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    bubble_ids = sorted(panel[panel["source"] == "bubble"]["episode_id"].unique())
    near_ids = sorted(panel[panel["source"] == "near_bubble"]["episode_id"].unique())

    n_ho_bub = max(1, round(len(bubble_ids) * 0.2))
    n_ho_near = max(1, round(len(near_ids) * 0.2))
    print(f"  {len(bubble_ids)} bubbles (hold out {n_ho_bub}), "
          f"{len(near_ids)} near-bubbles (hold out {n_ho_near})")

    splits = generate_splits(bubble_ids, near_ids, N_SPLITS, DEFAULT_SEED, panel=panel)
    print(f"  {len(splits)} unique splits generated (leverage-stratified)")

    # Episode-level data for classifiers — use ALL feature columns
    from config import ALL_FEATURE_COLS as ALL_FEAT
    all_feat = [c for c in ALL_FEAT if c in panel.columns]
    ep_features = panel.groupby(["episode_id", "source"])[all_feat].mean().reset_index()
    ep_features["is_bubble"] = (ep_features["source"] == "bubble").astype(int)

    t0 = time.time()

    all_surv = {}
    all_clf = {}

    for feat_name, features in FEATURE_SETS.items():
        t1 = time.time()
        print(f"\n  {feat_name} ({len(features)} features)...")

        # Survival models (parallel)
        surv = Parallel(n_jobs=-1, backend="loky")(
            delayed(run_one_split_survival)(panel, features, s, i)
            for i, s in enumerate(splits)
        )
        surv = [r for r in surv if r is not None]
        surv_df = pd.DataFrame(surv)
        surv_df["feature_set"] = feat_name
        all_surv[feat_name] = surv_df

        # Episode classifiers (parallel)
        clf = Parallel(n_jobs=-1, backend="loky")(
            delayed(run_one_split_classifier)(ep_features, features, s, i)
            for i, s in enumerate(splits)
        )
        clf = [r for r in clf if r is not None]
        clf_df = pd.DataFrame(clf)
        clf_df["feature_set"] = feat_name
        all_clf[feat_name] = clf_df

        elapsed = time.time() - t1

        # Print summary
        for col, label in [
            ("rsf_episode_auc", "RSF ep"),
            ("cox_episode_auc", "Cox ep"),
            ("rsf_month_auc", "RSF mo"),
            ("cox_month_auc", "Cox mo"),
        ]:
            vals = surv_df[col].dropna()
            if len(vals) > 0:
                print(f"    {label:8s}: {vals.median():.3f} "
                      f"[{vals.quantile(0.025):.3f}, {vals.quantile(0.975):.3f}]")

        for col, label in [("lr_auc", "LR ep"), ("rf_auc", "RF ep")]:
            vals = clf_df[col].dropna()
            if len(vals) > 0:
                print(f"    {label:8s}: {vals.median():.3f} "
                      f"[{vals.quantile(0.025):.3f}, {vals.quantile(0.975):.3f}]")

        print(f"    ({elapsed:.0f}s)")

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Total time: {total:.0f}s ({total/60:.1f} min)")

    # Save
    surv_combined = pd.concat(all_surv.values(), ignore_index=True)
    clf_combined = pd.concat(all_clf.values(), ignore_index=True)
    surv_combined.to_parquet(RESULTS_DIR / "holdout_80_20_survival.parquet", index=False)
    clf_combined.to_parquet(RESULTS_DIR / "holdout_80_20_classifiers.parquet", index=False)

    # Cox coefficients for vol_only
    vol_surv = all_surv.get("vol_only")
    if vol_surv is not None:
        print(f"\nCox coefficients (vol_only):")
        beta_cols = sorted([c for c in vol_surv.columns if c.startswith("cox_beta_")])
        for col in beta_cols:
            vals = vol_surv[col].dropna()
            if len(vals) > 0:
                name = col.replace("cox_beta_", "")
                sig = "*" if vals.quantile(0.025) > 0 or vals.quantile(0.975) < 0 else ""
                print(f"  {name:12s}: {vals.median():+.4f} "
                      f"[{vals.quantile(0.025):+.4f}, {vals.quantile(0.975):+.4f}] {sig}")

    # Final comparison table
    print(f"\n{'=' * 60}")
    print(f"FULL COMPARISON ({N_SPLITS} splits, 80/20 holdout)")
    print(f"{'=' * 60}")
    print(f"{'Features':15s} {'RSF Ep':>9s} {'Cox Ep':>9s} {'RSF Mo':>9s} {'LR Ep':>9s} {'RF Ep':>9s}")
    print("-" * 65)
    for feat_name in FEATURE_SETS:
        s = all_surv[feat_name]
        c = all_clf[feat_name]
        vals = [
            s["rsf_episode_auc"].dropna().median(),
            s["cox_episode_auc"].dropna().median(),
            s["rsf_month_auc"].dropna().median(),
            c["lr_auc"].dropna().median(),
            c["rf_auc"].dropna().median(),
        ]
        print(f"  {feat_name:15s}" + "".join(f" {v:9.3f}" for v in vals))
    print(f"  {'baseline(A)':15s} {'':9s} {'':9s} {'':9s} {'0.780':>9s} {'':9s}")

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
