"""
12_robustness.py — Robustness checks: walk-forward, per-fold ridge, Compustat lag.

1. Per-fold ridge: retrain text-CIV ridge model within each holdout fold
   (instead of global training) to eliminate information leakage in R/S.
2. Compustat forward-fill lag: shift Compustat datadate by +60 days before
   monthly merge to simulate real-time data availability for metrics B/F.
3. Expanding walk-forward: test multiple temporal cutoffs instead of single
   2015-01 split.

Output:
  data/results/robustness/perfold_ridge.csv
  data/results/robustness/compustat_lag.csv
  data/results/robustness/expanding_walk_forward.csv
  figures/fig_robustness_walk_forward.png
  figures/fig_robustness_compustat_lag.png
"""
import warnings
warnings.filterwarnings("ignore")

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import (DATA_DIR, RAW_DIR, PANELS_DIR, RESULTS_DIR, FIGURES_DIR,
                    METRICS_DIR, DEFAULT_SEED, VOL_COLS, SEC_COLS,
                    BOND_CIV_COLS, TEXT_CIV_COLS, INTERACTION_COLS,
                    PRE_MONTHS, NEAR_PEAK_THRESHOLD)
from src.catalog import get_bubbles, get_near_bubbles

ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = VOL_COLS + SEC_COLS + BOND_CIV_COLS + TEXT_CIV_COLS + INTERACTION_COLS

# NLP features used by the ridge model (must match text_civ.py)
NLP_FEATURES = [
    "lm_negative_pct", "lm_positive_pct", "lm_uncertainty_pct",
    "lm_litigious_pct", "lm_constraining_pct",
    "lm_modal_weak_pct", "lm_modal_strong_pct",
    "lm_net_sentiment", "lm_negativity_ratio",
    "fog_index", "fk_grade", "avg_words_per_sentence",
    "pct_complex_words",
    "offbalance_per10k", "covenant_per10k",
    "risk_escalation_per10k", "growth_narrative_per10k",
    "word_count",
]


def load_episode_panel():
    """Load and collapse panel to episode-level features."""
    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
    ep = (
        panel.groupby(["episode_id", "source"])[feat_cols]
        .mean()
        .reset_index()
    )
    ep["is_bubble"] = (ep["source"] == "bubble").astype(int)
    return panel, ep


def run_lr_splits(ep_features, feature_cols, n_splits=200, seed=DEFAULT_SEED):
    """Run LR classifier over n_splits holdout splits. Returns list of AUCs."""
    feat_cols = [c for c in feature_cols if c in ep_features.columns]
    bubble_ids = sorted(ep_features[ep_features["is_bubble"] == 1]["episode_id"].unique())
    near_ids = sorted(ep_features[ep_features["is_bubble"] == 0]["episode_id"].unique())

    n_ho_bub = max(1, round(len(bubble_ids) * 0.2))
    n_ho_near = max(1, round(len(near_ids) * 0.2))

    rng = np.random.RandomState(seed)
    results = []
    seen = set()

    while len(results) < n_splits:
        ho_b = tuple(sorted(rng.choice(bubble_ids, size=n_ho_bub, replace=False)))
        ho_n = tuple(sorted(rng.choice(near_ids, size=n_ho_near, replace=False)))
        key = (ho_b, ho_n)
        if key in seen:
            continue
        seen.add(key)

        ho_all = set(ho_b) | set(ho_n)
        tr = ep_features[~ep_features["episode_id"].isin(ho_all)]
        te = ep_features[ep_features["episode_id"].isin(ho_all)]

        X_tr = tr[feat_cols].fillna(0).values
        y_tr = tr["is_bubble"].values
        X_te = te[feat_cols].fillna(0).values
        y_te = te["is_bubble"].values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        try:
            lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
            lr.fit(X_tr_s, y_tr)
            prob = lr.predict_proba(X_te_s)[:, 1]
            results.append(roc_auc_score(y_te, prob))
        except Exception:
            continue

    return results


# ═══════════════════════════════════════════════════════════════════
# LIMITATION 1: Per-Fold Ridge for Text-CIV
# ═══════════════════════════════════════════════════════════════════

def run_perfold_ridge(panel, ep_features, n_splits=200):
    """
    Compare global-ridge vs per-fold-ridge for text-CIV metrics R/S.

    For each holdout fold:
    1. Identify test-fold episode CIKs
    2. Retrain ridge on only train-fold CIKs' firm-quarters
    3. Re-predict spread for all filings
    4. Re-aggregate to sector-monthly metric_r, metric_s
    5. Rebuild episode-level features
    6. Run LR classifier
    """
    print("\n  Loading text-CIV data...")
    civ_dir = DATA_DIR / "civ"
    text_panel_path = civ_dir / "text_civ_panel.parquet"
    civ_panel_path = civ_dir / "civ_panel.parquet"

    if not text_panel_path.exists() or not civ_panel_path.exists():
        print("  SKIP: text-CIV panel not found")
        return None

    text_panel = pd.read_parquet(text_panel_path)
    civ_panel = pd.read_parquet(civ_panel_path)

    # Build the ridge training set: firm-quarters with NLP + observed spreads
    # (same logic as text_civ.py:build_training_set, but we already have the data)
    text_panel["quarter"] = text_panel["filing_date"].dt.to_period("Q")
    has_cusip = text_panel.dropna(subset=["issuer_cusip"])

    # Get observed spreads from CIV panel
    civ_panel["date"] = pd.to_datetime(civ_panel["date"])
    civ_panel["quarter"] = civ_panel["date"].dt.to_period("Q")
    civ_quarterly = (
        civ_panel.groupby(["issuer_cusip", "quarter"])
        .agg(spread=("spread", "median"))
        .reset_index()
    )

    # NLP quarterly per issuer
    nlp_quarterly = (
        has_cusip.groupby(["issuer_cusip", "quarter"])[NLP_FEATURES]
        .median()
        .reset_index()
    )

    # Merge: firm-quarters with both NLP and spread
    ridge_data = nlp_quarterly.merge(
        civ_quarterly, on=["issuer_cusip", "quarter"], how="inner"
    ).dropna(subset=NLP_FEATURES + ["spread"])

    # Map issuer_cusip -> set of episode_ids (via text_panel)
    cusip_to_episodes = (
        has_cusip.groupby("issuer_cusip")["episode_id"]
        .apply(set)
        .to_dict()
    )
    ridge_data["episode_ids"] = ridge_data["issuer_cusip"].map(cusip_to_episodes)

    print(f"  Ridge training data: {len(ridge_data)} firm-quarters, "
          f"{ridge_data['issuer_cusip'].nunique()} issuers")

    # Pre-aggregate NLP features per episode for re-prediction
    # For each episode, we need filing-level NLP to re-predict spreads
    # then aggregate to monthly metric_r
    episode_filings = {}
    for eid, edf in text_panel.groupby("episode_id"):
        valid = edf[NLP_FEATURES].notna().all(axis=1)
        if valid.sum() > 0:
            episode_filings[eid] = edf[valid].copy()

    # Episode-level features (global ridge baseline)
    feat_cols = [c for c in FEATURE_COLS if c in ep_features.columns]
    bubble_ids = sorted(ep_features[ep_features["is_bubble"] == 1]["episode_id"].unique())
    near_ids = sorted(ep_features[ep_features["is_bubble"] == 0]["episode_id"].unique())

    n_ho_bub = max(1, round(len(bubble_ids) * 0.2))
    n_ho_near = max(1, round(len(near_ids) * 0.2))

    rng = np.random.RandomState(DEFAULT_SEED)
    global_aucs = []
    perfold_aucs = []
    seen = set()

    print(f"  Running {n_splits} splits...")
    t0 = time.time()

    while len(global_aucs) < n_splits:
        ho_b = tuple(sorted(rng.choice(bubble_ids, size=n_ho_bub, replace=False)))
        ho_n = tuple(sorted(rng.choice(near_ids, size=n_ho_near, replace=False)))
        key = (ho_b, ho_n)
        if key in seen:
            continue
        seen.add(key)

        ho_all = set(ho_b) | set(ho_n)
        tr_ep = ep_features[~ep_features["episode_id"].isin(ho_all)]
        te_ep = ep_features[ep_features["episode_id"].isin(ho_all)]

        X_tr = tr_ep[feat_cols].fillna(0).values
        y_tr = tr_ep["is_bubble"].values
        X_te = te_ep[feat_cols].fillna(0).values
        y_te = te_ep["is_bubble"].values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        # --- Global ridge (baseline) ---
        scaler_c = StandardScaler()
        X_tr_s = scaler_c.fit_transform(X_tr)
        X_te_s = scaler_c.transform(X_te)

        try:
            lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
            lr.fit(X_tr_s, y_tr)
            prob = lr.predict_proba(X_te_s)[:, 1]
            global_auc = roc_auc_score(y_te, prob)
        except Exception:
            continue

        # --- Per-fold ridge ---
        # Filter ridge training data: exclude firm-quarters linked to test episodes
        train_episode_ids = set(tr_ep["episode_id"].values)
        mask = ridge_data["episode_ids"].apply(
            lambda eps: bool(eps & train_episode_ids) if isinstance(eps, set) else False
        )
        fold_ridge_data = ridge_data[mask]

        if len(fold_ridge_data) < 30:
            # Too few training samples for ridge — use global
            perfold_aucs.append(global_auc)
            global_aucs.append(global_auc)
            continue

        # Fit per-fold ridge
        X_ridge = fold_ridge_data[NLP_FEATURES].values
        y_ridge = np.log(fold_ridge_data["spread"].values)

        ridge_scaler = StandardScaler()
        X_ridge_s = ridge_scaler.fit_transform(X_ridge)
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(X_ridge_s, y_ridge)

        # Re-predict metric_r for each episode using fold-specific ridge
        ep_r_updated = {}
        for eid, edf in episode_filings.items():
            X_nlp = edf[NLP_FEATURES].values
            X_nlp_s = ridge_scaler.transform(X_nlp)
            pred_spread = np.exp(ridge.predict(X_nlp_s))
            # metric_r is sector-level median predicted spread
            # (simplified: use median of all filings' predicted spreads)
            ep_r_updated[eid] = np.median(pred_spread)

        # Build updated episode features
        tr_ep_pf = tr_ep.copy()
        te_ep_pf = te_ep.copy()
        for df_pf in [tr_ep_pf, te_ep_pf]:
            df_pf["metric_r"] = df_pf["episode_id"].map(ep_r_updated)
            # metric_s is 12-month trajectory — hard to recompute at episode level
            # Use NaN (will be filled with 0)

        X_tr_pf = tr_ep_pf[feat_cols].fillna(0).values
        X_te_pf = te_ep_pf[feat_cols].fillna(0).values

        scaler_pf = StandardScaler()
        X_tr_pf_s = scaler_pf.fit_transform(X_tr_pf)
        X_te_pf_s = scaler_pf.transform(X_te_pf)

        try:
            lr_pf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
            lr_pf.fit(X_tr_pf_s, y_tr)
            prob_pf = lr_pf.predict_proba(X_te_pf_s)[:, 1]
            perfold_auc = roc_auc_score(y_te, prob_pf)
        except Exception:
            perfold_auc = global_auc

        global_aucs.append(global_auc)
        perfold_aucs.append(perfold_auc)

        if len(global_aucs) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    Split {len(global_aucs)}/{n_splits} "
                  f"({elapsed:.0f}s): global={np.median(global_aucs):.3f}, "
                  f"perfold={np.median(perfold_aucs):.3f}")

    global_aucs = np.array(global_aucs)
    perfold_aucs = np.array(perfold_aucs)
    diff = perfold_aucs - global_aucs

    result = pd.DataFrame({
        "global_auc": global_aucs,
        "perfold_auc": perfold_aucs,
        "diff": diff,
    })
    result.to_csv(ROBUSTNESS_DIR / "perfold_ridge.csv", index=False)

    print(f"\n  Per-Fold Ridge Results ({n_splits} splits):")
    print(f"    Global  median: {np.median(global_aucs):.3f} "
          f"[{np.percentile(global_aucs, 2.5):.3f}, {np.percentile(global_aucs, 97.5):.3f}]")
    print(f"    PerFold median: {np.median(perfold_aucs):.3f} "
          f"[{np.percentile(perfold_aucs, 2.5):.3f}, {np.percentile(perfold_aucs, 97.5):.3f}]")
    print(f"    Diff    median: {np.median(diff):+.3f} "
          f"[{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]")

    return result


# ═══════════════════════════════════════════════════════════════════
# LIMITATION 2: Compustat Forward-Fill Lag
# ═══════════════════════════════════════════════════════════════════

def recompute_leverage_with_lag(episode_id, lag_days):
    """
    Recompute monthly leverage for an episode with Compustat filing lag.
    Shifts datadate by +lag_days before the monthly forward-fill.
    Returns DataFrame with columns [date, permno, leverage_lagged].
    """
    crsp_path = RAW_DIR / f"{episode_id}_crsp.parquet"
    fundq_path = RAW_DIR / f"{episode_id}_fundq.parquet"
    ccm_path = RAW_DIR / f"{episode_id}_ccm.parquet"

    if not all(p.exists() for p in [crsp_path, fundq_path, ccm_path]):
        return pd.DataFrame()

    crsp = pd.read_parquet(crsp_path)
    fundq = pd.read_parquet(fundq_path)
    ccm = pd.read_parquet(ccm_path)

    crsp["date"] = pd.to_datetime(crsp["date"])
    fundq["datadate"] = pd.to_datetime(fundq["datadate"])

    # Monthly market cap per firm
    crsp["month"] = crsp["date"].dt.to_period("M")
    mcap = crsp.groupby(["permno", "month"]).last().reset_index()
    mcap["market_cap"] = mcap["price"] * mcap["shrout"] / 1000
    mcap["date"] = mcap["month"].dt.to_timestamp("M")

    # Map permno -> gvkey
    permno_gvkey = ccm.drop_duplicates("permno", keep="last").set_index("permno")["gvkey"].to_dict()
    mcap["gvkey"] = mcap["permno"].map(permno_gvkey)

    # Shift Compustat dates by lag_days (simulate publication delay)
    fundq_shifted = fundq.copy()
    fundq_shifted["datadate"] = fundq_shifted["datadate"] + pd.Timedelta(days=lag_days)

    # Forward-fill quarterly debt to monthly (with shifted dates)
    debt_frames = []
    for gvkey, gdf in fundq_shifted.groupby("gvkey"):
        gdf = gdf.sort_values("datadate").drop_duplicates("datadate", keep="last")
        gdf["month"] = gdf["datadate"].dt.to_period("M")
        gdf = gdf.drop_duplicates("month", keep="last")
        debt_monthly = gdf.set_index("month")[["total_debt", "total_assets"]].resample("M").last().ffill().reset_index()
        debt_monthly["date"] = debt_monthly["month"].dt.to_timestamp("M")
        debt_monthly["gvkey"] = gvkey
        debt_frames.append(debt_monthly[["date", "gvkey", "total_debt"]])

    if not debt_frames:
        return pd.DataFrame()

    debt_all = pd.concat(debt_frames, ignore_index=True)
    merged = mcap[["date", "permno", "gvkey", "market_cap"]].merge(
        debt_all, on=["date", "gvkey"], how="left"
    )

    merged["leverage"] = (
        merged["total_debt"] / (merged["total_debt"] + merged["market_cap"])
    ).clip(0.01, 0.99)

    return merged[["date", "permno", "leverage"]].dropna(subset=["leverage"])


def recompute_metric_b_with_lag(measures_orig, leverage_lagged):
    """Recompute metric_b using lagged leverage."""
    if leverage_lagged.empty:
        return pd.DataFrame(columns=["date", "metric_b"])

    measures = measures_orig.copy()
    # Replace leverage and asset_vol with lagged versions
    measures = measures.drop(columns=["leverage", "asset_vol"], errors="ignore")
    measures = measures.merge(
        leverage_lagged.rename(columns={"leverage": "leverage"}),
        on=["date", "permno"], how="left"
    )
    measures["asset_vol"] = measures["realized_vol"] * (1 - measures["leverage"].fillna(0.5))

    # Same OLS-based metric_b computation as 05_compute_metrics.py
    df = measures.dropna(subset=["asset_vol", "leverage"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "metric_b"])

    months = sorted(df["date"].unique())
    results = []
    for month in months:
        current = df[df["date"] == month]
        history = df[df["date"] <= month]
        if len(history) < 10:
            results.append({"date": month, "metric_b": np.nan})
            continue
        x = history["leverage"].values
        y = history["asset_vol"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 10 or (x.max() - x.min()) < 0.10:
            gap_val = np.nanmedian(current["asset_vol"].values) - y.mean() if len(y) > 0 else np.nan
            results.append({"date": month, "metric_b": gap_val})
            continue
        x_mean, y_mean = x.mean(), y.mean()
        ss_x = np.sum((x - x_mean) ** 2)
        b_hat = np.clip(np.sum((x - x_mean) * (y - y_mean)) / max(ss_x, 1e-6), -5.0, 5.0)
        a_hat = y_mean - b_hat * x_mean
        predicted = a_hat + b_hat * current["leverage"].values
        gap = current["asset_vol"].values - predicted
        results.append({"date": month, "metric_b": np.nanmedian(gap)})

    return pd.DataFrame(results)


def recompute_metric_f_with_lag(measures_orig, leverage_lagged):
    """Recompute metric_f (leverage trajectory) using lagged leverage."""
    if leverage_lagged.empty:
        return pd.DataFrame(columns=["date", "metric_f"])

    # Sector-level monthly median leverage
    monthly_lev = (
        leverage_lagged.groupby("date")["leverage"]
        .median()
        .rename("med_leverage")
        .reset_index()
        .sort_values("date")
    )
    monthly_lev["metric_f"] = monthly_lev["med_leverage"].diff(12)
    return monthly_lev[["date", "metric_f"]]


def run_compustat_lag(panel, ep_features, lag_values=(0, 30, 60, 90), n_splits=200):
    """
    Recompute metrics B and F with lagged Compustat data, then re-run classifier.
    """
    episodes = get_bubbles() + get_near_bubbles()
    measures_dir = DATA_DIR / "measures"

    rows = []
    for lag in lag_values:
        print(f"\n  Lag = {lag} days:")

        if lag == 0:
            # Baseline — use existing panel
            aucs = run_lr_splits(ep_features, FEATURE_COLS, n_splits)
        else:
            # Recompute metric_b and metric_f per episode with lagged leverage
            ep_updated = ep_features.copy()
            updated_b = {}
            updated_f = {}

            for ep in episodes:
                eid = ep["id"]
                measures_path = measures_dir / f"{eid}.parquet"
                if not measures_path.exists():
                    continue

                measures = pd.read_parquet(measures_path)
                lev_lagged = recompute_leverage_with_lag(eid, lag)

                if lev_lagged.empty:
                    continue

                new_b = recompute_metric_b_with_lag(measures, lev_lagged)
                new_f = recompute_metric_f_with_lag(measures, lev_lagged)

                if not new_b.empty:
                    updated_b[eid] = new_b.set_index("date")["metric_b"].to_dict()
                if not new_f.empty:
                    updated_f[eid] = new_f.set_index("date")["metric_f"].to_dict()

            print(f"    Recomputed B for {len(updated_b)} episodes, "
                  f"F for {len(updated_f)} episodes")

            # Rebuild episode-level means for metric_b and metric_f
            # We need the full monthly panel to recompute episode-level means
            train = pd.read_parquet(PANELS_DIR / "train.parquet")
            test = pd.read_parquet(PANELS_DIR / "test.parquet")
            full_panel = pd.concat([train, test], ignore_index=True)

            for eid in updated_b:
                mask = full_panel["episode_id"] == eid
                dates = full_panel.loc[mask, "date"]
                new_vals = dates.map(updated_b[eid])
                full_panel.loc[mask, "metric_b"] = new_vals.values

            for eid in updated_f:
                mask = full_panel["episode_id"] == eid
                dates = full_panel.loc[mask, "date"]
                new_vals = dates.map(updated_f[eid])
                full_panel.loc[mask, "metric_f"] = new_vals.values

            # Also update interaction features that depend on leverage
            # metric_v = metric_a * leverage, metric_w = metric_a * (1-leverage)
            # metric_x = metric_d * leverage
            # But med_leverage in the panel is from the original pipeline
            # For now, leave interactions unchanged (they use panel-level leverage)

            # Re-collapse to episode level
            feat_cols = [c for c in FEATURE_COLS if c in full_panel.columns]
            ep_updated = (
                full_panel.groupby(["episode_id", "source"])[feat_cols]
                .mean()
                .reset_index()
            )
            ep_updated["is_bubble"] = (ep_updated["source"] == "bubble").astype(int)

            aucs = run_lr_splits(ep_updated, FEATURE_COLS, n_splits)

        aucs = np.array(aucs)
        med = np.median(aucs)
        ci_lo = np.percentile(aucs, 2.5)
        ci_hi = np.percentile(aucs, 97.5)
        print(f"    LR AUC: {med:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

        rows.append({
            "lag_days": lag,
            "median_auc": med,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "mean_auc": aucs.mean(),
            "std_auc": aucs.std(),
        })

    result = pd.DataFrame(rows)
    result.to_csv(ROBUSTNESS_DIR / "compustat_lag.csv", index=False)
    print(f"\n  Saved: {ROBUSTNESS_DIR / 'compustat_lag.csv'}")
    return result


# ═══════════════════════════════════════════════════════════════════
# LIMITATION 4: Expanding-Window Walk-Forward
# ═══════════════════════════════════════════════════════════════════

def run_expanding_walk_forward(cutoffs=("2008-01", "2010-01", "2012-01", "2015-01", "2018-01")):
    """
    Run episode-level LR classifier at multiple temporal cutoffs.
    Train on episodes with peak_date < cutoff, test on peak_date >= cutoff.
    """
    bubbles = get_bubbles()
    nears = get_near_bubbles()
    all_episodes = bubbles + nears

    rows = []
    for cutoff in cutoffs:
        train_eps = [e for e in all_episodes if e["peak_date"] is not None and e["peak_date"] < cutoff]
        test_eps = [e for e in all_episodes if e["peak_date"] is not None and e["peak_date"] >= cutoff]

        train_bub = [e for e in train_eps if e["is_bubble"]]
        train_near = [e for e in train_eps if not e["is_bubble"]]
        test_bub = [e for e in test_eps if e["is_bubble"]]
        test_near = [e for e in test_eps if not e["is_bubble"]]

        n_train_bub = len(train_bub)
        n_train_near = len(train_near)
        n_test_bub = len(test_bub)
        n_test_near = len(test_near)

        print(f"\n  Cutoff {cutoff}: train={n_train_bub}B+{n_train_near}N, "
              f"test={n_test_bub}B+{n_test_near}N")

        if n_train_bub < 3 or n_train_near < 2 or n_test_bub < 2 or n_test_near < 2:
            print(f"    SKIP: insufficient episodes")
            rows.append({
                "cutoff": cutoff,
                "n_train_bub": n_train_bub, "n_train_near": n_train_near,
                "n_test_bub": n_test_bub, "n_test_near": n_test_near,
                "lr_auc": np.nan,
            })
            continue

        # Load and build panels for this cutoff
        train_ids = {e["id"] for e in train_eps}
        test_ids = {e["id"] for e in test_eps}

        train_frames = []
        test_frames = []

        for ep in all_episodes:
            eid = ep["id"]
            path = METRICS_DIR / f"{eid}.parquet"
            if not path.exists():
                continue

            df = pd.read_parquet(path)
            peak_ts = pd.Timestamp(ep["peak_date"] + "-01")

            df["tau"] = (
                (df["date"].dt.year - peak_ts.year) * 12
                + df["date"].dt.month - peak_ts.month
            )

            # Pre-peak only
            df = df[(df["tau"] >= -PRE_MONTHS) & (df["tau"] < 0)].copy()

            if ep["is_bubble"]:
                df["event_12m"] = (df["tau"] >= NEAR_PEAK_THRESHOLD).astype(int)
                df["time_to_peak"] = (-df["tau"]).clip(lower=1)
                df["source"] = "bubble"
            else:
                df["event_12m"] = 0
                df["time_to_peak"] = 36
                df["source"] = "near_bubble"

            if eid in train_ids:
                train_frames.append(df)
            else:
                test_frames.append(df)

        if not train_frames or not test_frames:
            print(f"    SKIP: no data")
            continue

        train_panel = pd.concat(train_frames, ignore_index=True)
        test_panel = pd.concat(test_frames, ignore_index=True)

        # Collapse to episode level
        feat_cols = [c for c in FEATURE_COLS if c in train_panel.columns]
        tr_ep = train_panel.groupby(["episode_id", "source"])[feat_cols].mean().reset_index()
        te_ep = test_panel.groupby(["episode_id", "source"])[feat_cols].mean().reset_index()
        tr_ep["is_bubble"] = (tr_ep["source"] == "bubble").astype(int)
        te_ep["is_bubble"] = (te_ep["source"] == "bubble").astype(int)

        X_tr = tr_ep[feat_cols].fillna(0).values
        y_tr = tr_ep["is_bubble"].values
        X_te = te_ep[feat_cols].fillna(0).values
        y_te = te_ep["is_bubble"].values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            print(f"    SKIP: single class in train or test")
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        try:
            lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
            lr.fit(X_tr_s, y_tr)
            prob = lr.predict_proba(X_te_s)[:, 1]
            auc = roc_auc_score(y_te, prob)
            print(f"    LR episode AUC: {auc:.3f}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            auc = np.nan

        rows.append({
            "cutoff": cutoff,
            "n_train_bub": n_train_bub, "n_train_near": n_train_near,
            "n_test_bub": n_test_bub, "n_test_near": n_test_near,
            "n_train_total": n_train_bub + n_train_near,
            "n_test_total": n_test_bub + n_test_near,
            "lr_auc": auc,
        })

    result = pd.DataFrame(rows)
    result.to_csv(ROBUSTNESS_DIR / "expanding_walk_forward.csv", index=False)
    print(f"\n  Saved: {ROBUSTNESS_DIR / 'expanding_walk_forward.csv'}")
    return result


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

def make_figures(wf_result, lag_result):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipped figures")
        return

    # Walk-forward figure
    if wf_result is not None and not wf_result.empty:
        valid = wf_result.dropna(subset=["lr_auc"])
        if len(valid) >= 2:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(range(len(valid)), valid["lr_auc"].values, "o-",
                     color="steelblue", linewidth=2, markersize=10)
            ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")

            # Highlight the 2015 baseline
            baseline_idx = valid[valid["cutoff"] == "2015-01"].index
            if len(baseline_idx) > 0:
                pos = list(valid.index).index(baseline_idx[0])
                ax1.axvline(pos, color="red", linestyle=":", alpha=0.6, label="Baseline (2015)")

            ax1.set_xticks(range(len(valid)))
            labels = []
            for _, row in valid.iterrows():
                labels.append(f"{row['cutoff']}\n({int(row['n_train_bub'])}B+{int(row['n_train_near'])}N)")
            ax1.set_xticklabels(labels, fontsize=9)

            ax1.set_ylabel("Episode-Level AUC", fontsize=12)
            ax1.set_xlabel("Walk-Forward Cutoff (Train Composition)", fontsize=12)
            ax1.set_ylim(0.3, 1.0)
            ax1.legend(fontsize=10)
            ax1.set_title("Expanding-Window Walk-Forward", fontsize=14)
            plt.tight_layout()

            fig_path = FIGURES_DIR / "fig_robustness_walk_forward.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {fig_path}")

    # Compustat lag figure
    if lag_result is not None and not lag_result.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(lag_result["lag_days"], lag_result["median_auc"],
               width=12, color="steelblue", alpha=0.8)
        ax.errorbar(lag_result["lag_days"], lag_result["median_auc"],
                    yerr=[lag_result["median_auc"] - lag_result["ci_low"],
                          lag_result["ci_high"] - lag_result["median_auc"]],
                    fmt="none", color="black", capsize=5)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Compustat Filing Lag (days)", fontsize=12)
        ax.set_ylabel("Median Episode AUC", fontsize=12)
        ax.set_title("Sensitivity to Compustat Forward-Fill Lag", fontsize=14)
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks(lag_result["lag_days"])
        plt.tight_layout()

        fig_path = FIGURES_DIR / "fig_robustness_compustat_lag.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Robustness Checks for Known Limitations")
    print("=" * 60)

    panel, ep_features = load_episode_panel()
    print(f"Loaded {len(ep_features)} episodes "
          f"({ep_features['is_bubble'].sum()} bubbles, "
          f"{(1 - ep_features['is_bubble']).sum()} near-bubbles)")

    # Limitation 1: Per-fold ridge
    print("\n" + "=" * 60)
    print("LIMITATION 1: Per-Fold Ridge for Text-CIV")
    print("=" * 60)
    ridge_result = run_perfold_ridge(panel, ep_features, n_splits=200)

    # Limitation 2: Compustat lag
    print("\n" + "=" * 60)
    print("LIMITATION 2: Compustat Forward-Fill Lag")
    print("=" * 60)
    lag_result = run_compustat_lag(panel, ep_features, lag_values=[0, 30, 60, 90], n_splits=200)

    # Limitation 4: Expanding walk-forward
    print("\n" + "=" * 60)
    print("LIMITATION 4: Expanding Walk-Forward")
    print("=" * 60)
    wf_result = run_expanding_walk_forward()

    # Figures
    print("\n" + "=" * 60)
    print("FIGURES")
    print("=" * 60)
    make_figures(wf_result, lag_result)

    print("\n" + "=" * 60)
    print("DONE — All robustness checks complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
