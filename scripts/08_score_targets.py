"""
08_score_targets.py — Score AI, quantum, nuclear2 with full-sample trained model.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from config import PANELS_DIR, RESULTS_DIR

VOL_COLS = ["metric_a", "metric_b", "metric_c", "metric_d", "metric_e", "metric_f"]
SEC_COLS = ["metric_g", "metric_h", "metric_i", "metric_j", "metric_k",
            "metric_l", "metric_m", "metric_n"]
TEXT_CIV_COLS = ["metric_r", "metric_s"]
ALL_COLS = VOL_COLS + SEC_COLS + TEXT_CIV_COLS


def prepare_features(df, feature_cols, train_stats=None):
    out = df[feature_cols].copy()
    if train_stats is None:
        train_stats = {}
        for col in out.columns:
            train_stats[col] = {
                "median": out[col].median(),
                "mean": out[col].mean(),
                "std": out[col].std(),
            }
    for col in out.columns:
        out[col] = out[col].fillna(train_stats[col]["median"])
    out = out.fillna(0)
    for col in out.columns:
        s = train_stats[col]["std"]
        m = train_stats[col]["mean"]
        if s > 1e-8:
            out[col] = (out[col] - m) / s
        else:
            out[col] = 0.0
    return out, train_stats


def main():
    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    target = pd.read_parquet(PANELS_DIR / "target.parquet")

    # Train on ALL labeled data (train + test) for target scoring
    full = pd.concat([train, test], ignore_index=True)

    for name, features in [("vol_only", VOL_COLS), ("all", ALL_COLS)]:
        print(f"\n{'='*60}")
        print(f"Scoring targets with {name} ({len(features)} features)")
        print(f"{'='*60}")

        train_X, train_stats = prepare_features(full, features)

        cox_train = train_X.copy()
        cox_train["T"] = full["time_to_peak"].values
        cox_train["E"] = full["event_12m"].values

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_train, duration_col="T", event_col="E")

        for tid in target["episode_id"].unique():
            t_df = target[target["episode_id"] == tid].copy()
            t_X, _ = prepare_features(t_df, features, train_stats=train_stats)
            t_df["hazard"] = cph.predict_partial_hazard(t_X).values.flatten()

            print(f"\n  {tid}:")
            # Show last 6 months
            recent = t_df.sort_values("date").tail(6)
            for _, row in recent.iterrows():
                date_str = row["date"].strftime("%Y-%m")
                h = row["hazard"]
                print(f"    {date_str}: hazard={h:.3f}")

            # Save
            out = t_df[["date", "episode_id", "hazard"]].copy()
            out["feature_set"] = name
            out.to_parquet(RESULTS_DIR / f"target_{tid}_{name}.parquet", index=False)

    print(f"\nTarget scores saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
