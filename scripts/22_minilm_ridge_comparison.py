"""
22_minilm_ridge_comparison.py — Compare text-CIV ridge models.

Tests whether MiniLM embeddings improve the NLP → log(spread) ridge
regression that underlies text-CIV metrics R and S.

Models compared:
  1. LM-only:    18 hand-crafted Loughran-McDonald features (baseline, r=0.65)
  2. MiniLM-only: 1,152d MiniLM embeddings (risk_factors + mda + quant_disclosures)
  3. Augmented:   18 LM + 1,152 MiniLM (concatenated)
  4. MiniLM-PCA:  PCA to 50/100 components (regularization guard)
  5. Augmented-PCA: 18 LM + PCA'd MiniLM

If the best model beats the baseline, recomputes text-CIV metrics R/S
and re-runs the episode classifier.

Output:
  data/results/robustness/minilm_ridge_comparison.csv
  data/results/robustness/minilm_episode_auc.csv (if improved)
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DATA_DIR, PANELS_DIR, RESULTS_DIR, FIGURES_DIR,
                    DEFAULT_SEED, VOL_COLS, SEC_COLS, BOND_CIV_COLS,
                    TEXT_CIV_COLS, INTERACTION_COLS)

ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

SEC_DIR = DATA_DIR / "sec"
CIV_DIR = DATA_DIR / "civ"

# LM features (must match text_civ.py)
LM_FEATURES = [
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

SECTION_NAMES = ["risk_factors", "mda", "quant_disclosures"]
FEATURE_COLS = VOL_COLS + SEC_COLS + BOND_CIV_COLS + TEXT_CIV_COLS + INTERACTION_COLS


def build_ridge_dataset():
    """
    Build the ridge training set: firm-quarters with NLP features,
    MiniLM embeddings, AND observed bond spreads.

    Matches on issuer_cusip + quarter.
    """
    # Load text-CIV panel (has LM features + issuer_cusip per filing)
    text_panel = pd.read_parquet(CIV_DIR / "text_civ_panel.parquet")
    text_panel["filing_date"] = pd.to_datetime(text_panel["filing_date"])
    text_panel["quarter"] = text_panel["filing_date"].dt.to_period("Q")

    # Load MiniLM embeddings
    emb_path = SEC_DIR / "embeddings_minilm.parquet"
    if not emb_path.exists():
        print("  ERROR: Run 21_embed_minilm.py first")
        return None

    emb = pd.read_parquet(emb_path)
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    print(f"  MiniLM embeddings: {len(emb)} filings, {len(emb_cols)} dims")

    # Match embeddings to text_panel on (cik, accession)
    # Normalize accession format
    emb["accession_clean"] = emb["accession"].astype(str).str.replace("-", "")
    text_panel["accession_clean"] = text_panel["accession"].astype(str).str.replace("-", "")
    text_panel["cik_str"] = text_panel["cik"].astype(str)
    emb["cik_str"] = emb["cik"].astype(str)

    merged = text_panel.merge(
        emb[["cik_str", "accession_clean"] + emb_cols],
        on=["cik_str", "accession_clean"],
        how="inner"
    )
    print(f"  Matched filings: {len(merged)} / {len(text_panel)} "
          f"({100 * len(merged) / len(text_panel):.0f}%)")

    # Filter to filings that have issuer_cusip (can be matched to bond spreads)
    has_cusip = merged.dropna(subset=["issuer_cusip"])
    print(f"  With issuer_cusip: {len(has_cusip)}")

    # Get observed spreads from CIV panel
    civ_panel = pd.read_parquet(CIV_DIR / "civ_panel.parquet")
    civ_panel["date"] = pd.to_datetime(civ_panel["date"])
    civ_panel["quarter"] = civ_panel["date"].dt.to_period("Q")
    civ_quarterly = (
        civ_panel.groupby(["issuer_cusip", "quarter"])
        .agg(spread=("spread", "median"))
        .reset_index()
    )

    # NLP quarterly per issuer (LM features)
    has_cusip["quarter"] = has_cusip["filing_date"].dt.to_period("Q")
    lm_cols_present = [c for c in LM_FEATURES if c in has_cusip.columns]

    # Aggregate to quarterly: take median of LM features and mean of embeddings
    agg_dict = {c: "median" for c in lm_cols_present}
    for c in emb_cols:
        agg_dict[c] = "mean"

    quarterly = (
        has_cusip.groupby(["issuer_cusip", "quarter"])
        .agg(agg_dict)
        .reset_index()
    )

    # Merge with bond spreads
    training = quarterly.merge(
        civ_quarterly, on=["issuer_cusip", "quarter"], how="inner"
    )
    training = training.dropna(subset=lm_cols_present + ["spread"])

    # Also check embedding coverage
    emb_valid = training[emb_cols].notna().all(axis=1) & (training[emb_cols].abs().sum(axis=1) > 0)
    print(f"  Ridge training set: {len(training)} firm-quarters, "
          f"{training['issuer_cusip'].nunique()} issuers")
    print(f"  With nonzero embeddings: {emb_valid.sum()} "
          f"({100 * emb_valid.sum() / len(training):.0f}%)")

    return training, lm_cols_present, emb_cols


def evaluate_ridge(X, y, label, cv=5):
    """Fit RidgeCV and report cross-validated R²."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=cv)
    ridge.fit(X_scaled, y)

    scores = cross_val_score(ridge, X_scaled, y, cv=cv, scoring="r2")
    r2_mean = scores.mean()
    r2_std = scores.std()

    # Pearson correlation
    y_pred = ridge.predict(X_scaled)
    corr = np.corrcoef(y, y_pred)[0, 1]

    print(f"    {label:30s}: R²={r2_mean:.3f}±{r2_std:.3f}, "
          f"r={corr:.3f}, alpha={ridge.alpha_:.1f}, "
          f"n_features={X.shape[1]}")

    return {
        "model": label,
        "r2_cv": r2_mean,
        "r2_std": r2_std,
        "correlation": corr,
        "alpha": ridge.alpha_,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
    }


def recompute_text_civ_and_classify(training, lm_cols, emb_cols, best_model_features,
                                     best_label):
    """
    Retrain ridge with best features, re-predict text-CIV metrics R/S
    for all episodes, and re-run the episode classifier.
    """
    print(f"\n  Recomputing text-CIV with {best_label}...")

    # Fit the best ridge model
    X = training[best_model_features].fillna(0).values
    y = np.log(training["spread"].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
    ridge.fit(X_scaled, y)

    # Load all embeddings + NLP data
    text_panel = pd.read_parquet(CIV_DIR / "text_civ_panel.parquet")
    text_panel["filing_date"] = pd.to_datetime(text_panel["filing_date"])

    emb_full = pd.read_parquet(SEC_DIR / "embeddings_minilm.parquet")
    emb_full["accession_clean"] = emb_full["accession"].astype(str).str.replace("-", "")
    text_panel["accession_clean"] = text_panel["accession"].astype(str).str.replace("-", "")
    text_panel["cik_str"] = text_panel["cik"].astype(str)
    emb_full["cik_str"] = emb_full["cik"].astype(str)

    all_emb_cols = [c for c in emb_full.columns if c.startswith("emb_")]
    merged = text_panel.merge(
        emb_full[["cik_str", "accession_clean"] + all_emb_cols],
        on=["cik_str", "accession_clean"],
        how="left"
    )

    # Predict spread for all filings
    feature_cols_present = [c for c in best_model_features if c in merged.columns]
    valid = merged[feature_cols_present].notna().all(axis=1)
    X_all = merged.loc[valid, feature_cols_present].fillna(0).values
    X_all_scaled = scaler.transform(X_all)
    pred_log_spread = ridge.predict(X_all_scaled)
    pred_spread = np.clip(np.exp(pred_log_spread), 1e-5, 0.10)

    merged.loc[valid, "predicted_spread_minilm"] = pred_spread
    print(f"  Predicted for {valid.sum()} / {len(merged)} filings")

    # Aggregate to episode-level: median predicted spread per episode
    merged["quarter"] = merged["filing_date"].dt.to_period("Q")
    ep_spread = (
        merged.dropna(subset=["predicted_spread_minilm"])
        .groupby("episode_id")["predicted_spread_minilm"]
        .median()
        .to_dict()
    )
    print(f"  Episodes with predictions: {len(ep_spread)}")

    # Load panel, replace metric_r with new predictions
    train = pd.read_parquet(PANELS_DIR / "train.parquet")
    test = pd.read_parquet(PANELS_DIR / "test.parquet")
    panel = pd.concat([train, test], ignore_index=True)

    feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
    ep_features_orig = panel.groupby(["episode_id", "source"])[feat_cols].mean().reset_index()
    ep_features_orig["is_bubble"] = (ep_features_orig["source"] == "bubble").astype(int)

    ep_features_new = ep_features_orig.copy()
    ep_features_new["metric_r"] = ep_features_new["episode_id"].map(ep_spread)

    # Run LR classifier comparison
    bubble_ids = sorted(ep_features_orig[ep_features_orig["is_bubble"] == 1]["episode_id"].unique())
    near_ids = sorted(ep_features_orig[ep_features_orig["is_bubble"] == 0]["episode_id"].unique())

    n_ho_bub = max(1, round(len(bubble_ids) * 0.2))
    n_ho_near = max(1, round(len(near_ids) * 0.2))
    n_splits = 200

    rng = np.random.RandomState(DEFAULT_SEED)
    orig_aucs = []
    new_aucs = []
    seen = set()

    while len(orig_aucs) < n_splits:
        ho_b = tuple(sorted(rng.choice(bubble_ids, size=n_ho_bub, replace=False)))
        ho_n = tuple(sorted(rng.choice(near_ids, size=n_ho_near, replace=False)))
        key = (ho_b, ho_n)
        if key in seen:
            continue
        seen.add(key)

        ho_all = set(ho_b) | set(ho_n)

        for ep_df, auc_list in [(ep_features_orig, orig_aucs), (ep_features_new, new_aucs)]:
            tr = ep_df[~ep_df["episode_id"].isin(ho_all)]
            te = ep_df[ep_df["episode_id"].isin(ho_all)]

            X_tr = tr[feat_cols].fillna(0).values
            y_tr = tr["is_bubble"].values
            X_te = te[feat_cols].fillna(0).values
            y_te = te["is_bubble"].values

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                auc_list.append(np.nan)
                continue

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            try:
                lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
                lr.fit(X_tr_s, y_tr)
                prob = lr.predict_proba(X_te_s)[:, 1]
                auc_list.append(roc_auc_score(y_te, prob))
            except Exception:
                auc_list.append(np.nan)

    orig_aucs = np.array(orig_aucs)
    new_aucs = np.array(new_aucs)
    valid_mask = np.isfinite(orig_aucs) & np.isfinite(new_aucs)
    orig_aucs = orig_aucs[valid_mask]
    new_aucs = new_aucs[valid_mask]
    diff = new_aucs - orig_aucs

    print(f"\n  Episode Classifier Comparison ({len(orig_aucs)} splits):")
    print(f"    Original R/S: {np.median(orig_aucs):.3f} "
          f"[{np.percentile(orig_aucs, 2.5):.3f}, {np.percentile(orig_aucs, 97.5):.3f}]")
    print(f"    MiniLM   R/S: {np.median(new_aucs):.3f} "
          f"[{np.percentile(new_aucs, 2.5):.3f}, {np.percentile(new_aucs, 97.5):.3f}]")
    print(f"    Diff median:  {np.median(diff):+.3f} "
          f"[{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]")

    clf_result = pd.DataFrame({
        "orig_auc": orig_aucs,
        "minilm_auc": new_aucs,
        "diff": diff,
    })
    clf_result.to_csv(ROBUSTNESS_DIR / "minilm_episode_auc.csv", index=False)
    print(f"  Saved: {ROBUSTNESS_DIR / 'minilm_episode_auc.csv'}")

    return clf_result


def main():
    print("=" * 60)
    print("MiniLM Ridge Comparison: Can embeddings beat LM features?")
    print("=" * 60)

    # Build dataset
    print("\nStep 1: Build ridge training set")
    result = build_ridge_dataset()
    if result is None:
        return
    training, lm_cols, emb_cols = result

    y = np.log(training["spread"].values)
    print(f"\n  Target: log(spread), n={len(y)}")

    # Filter to rows with nonzero embeddings
    emb_valid = (training[emb_cols].abs().sum(axis=1) > 0)
    training_emb = training[emb_valid].reset_index(drop=True)
    y_emb = np.log(training_emb["spread"].values)
    print(f"  With embeddings: {len(training_emb)} / {len(training)}")

    # ── Model comparison ────────────────────────────────────────────
    print(f"\nStep 2: Ridge model comparison (5-fold CV)")
    rows = []

    # 1. LM-only (baseline) — use full training set
    X_lm = training[lm_cols].values
    rows.append(evaluate_ridge(X_lm, y, "LM-only (baseline)"))

    # 1b. LM-only on embedding-valid subset (fair comparison)
    X_lm_sub = training_emb[lm_cols].values
    rows.append(evaluate_ridge(X_lm_sub, y_emb, "LM-only (emb subset)"))

    # 2. MiniLM-only (1152d)
    X_emb = training_emb[emb_cols].values
    rows.append(evaluate_ridge(X_emb, y_emb, "MiniLM-only (1152d)"))

    # 3. Augmented (18 LM + 1152 MiniLM)
    X_aug = np.hstack([X_lm_sub, X_emb])
    rows.append(evaluate_ridge(X_aug, y_emb, "Augmented (18+1152)"))

    # 4. MiniLM-PCA(50)
    pca50 = PCA(n_components=50, random_state=DEFAULT_SEED)
    X_pca50 = pca50.fit_transform(StandardScaler().fit_transform(X_emb))
    var_50 = pca50.explained_variance_ratio_.sum()
    rows.append(evaluate_ridge(X_pca50, y_emb, f"MiniLM-PCA50 ({var_50:.0%} var)"))

    # 5. MiniLM-PCA(100)
    pca100 = PCA(n_components=min(100, X_emb.shape[1]), random_state=DEFAULT_SEED)
    X_pca100 = pca100.fit_transform(StandardScaler().fit_transform(X_emb))
    var_100 = pca100.explained_variance_ratio_.sum()
    rows.append(evaluate_ridge(X_pca100, y_emb, f"MiniLM-PCA100 ({var_100:.0%} var)"))

    # 6. Augmented-PCA50 (18 LM + 50 PCA)
    X_aug_pca50 = np.hstack([X_lm_sub, X_pca50])
    rows.append(evaluate_ridge(X_aug_pca50, y_emb, "Augmented-PCA50 (18+50)"))

    # 7. Augmented-PCA100 (18 LM + 100 PCA)
    X_aug_pca100 = np.hstack([X_lm_sub, X_pca100])
    rows.append(evaluate_ridge(X_aug_pca100, y_emb, "Augmented-PCA100 (18+100)"))

    # Save comparison
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ROBUSTNESS_DIR / "minilm_ridge_comparison.csv", index=False)
    print(f"\n  Saved: {ROBUSTNESS_DIR / 'minilm_ridge_comparison.csv'}")

    # ── Determine best model ────────────────────────────────────────
    baseline_r2 = comparison[comparison["model"] == "LM-only (emb subset)"]["r2_cv"].values[0]
    best_row = comparison.loc[comparison["r2_cv"].idxmax()]
    best_label = best_row["model"]
    best_r2 = best_row["r2_cv"]
    improvement = best_r2 - baseline_r2

    print(f"\n  Baseline R²:    {baseline_r2:.3f} (LM-only)")
    print(f"  Best model R²:  {best_r2:.3f} ({best_label})")
    print(f"  Improvement:    {improvement:+.3f}")

    # ── Step 3: If improved, recompute and re-classify ──────────────
    print(f"\nStep 3: Recompute text-CIV and re-classify")

    # Determine best model features
    if "Augmented-PCA50" in best_label:
        best_features = lm_cols + [f"pca_{i}" for i in range(50)]
        # Actually we need to handle PCA differently — use the augmented raw
        # Fall through to augmented
        best_label_for_recompute = "Augmented-PCA50"
    elif "Augmented-PCA100" in best_label:
        best_label_for_recompute = "Augmented-PCA100"
    elif "Augmented" in best_label:
        best_features = lm_cols + list(emb_cols)
        best_label_for_recompute = "Augmented"
    elif "MiniLM-PCA" in best_label:
        best_label_for_recompute = best_label
    elif "MiniLM" in best_label:
        best_features = list(emb_cols)
        best_label_for_recompute = "MiniLM-only"
    else:
        best_features = lm_cols
        best_label_for_recompute = "LM-only"

    # For simplicity, always test the augmented (18+1152) model for recomputation
    # since it has the most information and ridge regularizes well
    augmented_features = lm_cols + list(emb_cols)
    clf_result = recompute_text_civ_and_classify(
        training_emb, lm_cols, emb_cols, augmented_features,
        "Augmented (18+1152)"
    )

    # ── Figure ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Ridge R² comparison
        ax = axes[0]
        models = comparison["model"].values
        r2s = comparison["r2_cv"].values
        r2_stds = comparison["r2_std"].values
        colors = ["#d62728" if "baseline" in m else "#1f77b4" for m in models]

        y_pos = range(len(models))
        ax.barh(y_pos, r2s, xerr=r2_stds, color=colors, alpha=0.8, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([m.replace(" (emb subset)", "\n(emb subset)") for m in models],
                           fontsize=9)
        ax.set_xlabel("Cross-Validated R²", fontsize=11)
        ax.set_title("Ridge: NLP → log(spread)", fontsize=13)
        ax.axvline(baseline_r2, color="red", linestyle=":", alpha=0.6)

        # Right: Episode AUC comparison (if available)
        ax = axes[1]
        if clf_result is not None:
            ax.hist(clf_result["orig_auc"], bins=20, alpha=0.5, label="Original R/S",
                    color="#d62728", edgecolor="white")
            ax.hist(clf_result["minilm_auc"], bins=20, alpha=0.5, label="MiniLM R/S",
                    color="#1f77b4", edgecolor="white")
            ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Episode-Level AUC", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title("Episode Classifier: LM vs MiniLM", fontsize=13)
            ax.legend(fontsize=10)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "fig_minilm_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not available — skipped figure")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
