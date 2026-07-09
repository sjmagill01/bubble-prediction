"""
text_civ.py — Text-implied volatility from SEC filings.

Novel approach: predict bond spreads from filing NLP features, then
Merton-invert the predicted spread to get text-implied asset volatility.

Pipeline:
  1. Build training set: firms with BOTH NLP scores AND observed bond spreads
  2. Train ridge regression: NLP features -> log(spread)
  3. Predict spread for ALL firms with NLP scores (even those without bonds)
  4. Merton-invert predicted spread -> text-CIV
  5. Aggregate to sector-monthly

This gives a credit-risk-implied vol measure for every firm that files a 10-K,
dramatically expanding coverage beyond the ~5,000 bond issuers in TRACE.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from config import DATA_DIR

CIV_DIR = DATA_DIR / "civ"
NLP_DIR = DATA_DIR / "sec" / "nlp_scores"
CIV_DIR.mkdir(parents=True, exist_ok=True)

# NLP features to use for spread prediction
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


def build_training_set():
    """
    Build training set of firm-quarters with BOTH NLP scores AND
    observed bond spreads.

    Matches on: issuer_cusip (bonds) ↔ CIK (SEC filings) via
    the CIK-to-issuer-cusip mapping we built.
    """
    civ_panel_path = CIV_DIR / "civ_panel.parquet"
    if not civ_panel_path.exists():
        print("  No bond-CIV panel. Run 03_fetch_bond_civ.py first.")
        return None, None

    # Load bond-CIV panel (has issuer_cusip, date, spread, leverage, civ)
    civ = pd.read_parquet(civ_panel_path)
    civ["quarter"] = civ["date"].dt.to_period("Q")

    # Quarterly median spread per issuer
    civ_quarterly = (
        civ.groupby(["issuer_cusip", "quarter"])
        .agg(spread=("spread", "median"), leverage=("leverage", "median"),
             civ=("civ", "median"))
        .reset_index()
    )

    # Load issuer CUSIP -> permno mapping
    cusip_map_path = CIV_DIR / "issuer_cusip_mapping.parquet"
    if not cusip_map_path.exists():
        print("  No issuer CUSIP mapping.")
        return None, None
    cusip_map = pd.read_parquet(cusip_map_path)

    # Load CIK mapping (has ticker -> CIK)
    cik_map_path = DATA_DIR / "sec" / "cik_mapping.parquet"
    if not cik_map_path.exists():
        print("  No CIK mapping.")
        return None, None
    cik_map = pd.read_parquet(cik_map_path)

    # Load all NLP scores and tag with CIK
    nlp_frames = []
    for nlp_path in sorted(NLP_DIR.glob("*.parquet")):
        df = pd.read_parquet(nlp_path)
        df["episode_id"] = nlp_path.stem
        nlp_frames.append(df)

    if not nlp_frames:
        print("  No NLP scores.")
        return None, None

    all_nlp = pd.concat(nlp_frames, ignore_index=True)
    all_nlp["filing_date"] = pd.to_datetime(all_nlp["filing_date"])
    all_nlp["quarter"] = all_nlp["filing_date"].dt.to_period("Q")

    # Build CIK -> issuer_cusip bridge
    # CIK map: episode_id, ticker, cik
    # CUSIP map: episode_id, permno, issuer_cusip
    # We also have firms files: episode_id, ticker, permno
    # So: CIK -> (episode, ticker) -> (episode, permno) -> issuer_cusip

    from config import RAW_DIR
    ticker_permno = {}
    for firms_path in RAW_DIR.glob("*_firms.parquet"):
        eid = firms_path.stem.replace("_firms", "")
        firms = pd.read_parquet(firms_path)
        if "ticker" in firms.columns and "permno" in firms.columns:
            for _, row in firms.iterrows():
                ticker_permno[(eid, row["ticker"])] = int(row["permno"])

    # Map CIK -> issuer_cusip
    cik_to_cusip = {}
    for _, row in cik_map.iterrows():
        if pd.isna(row["cik"]):
            continue
        key = (row["episode_id"], row["ticker"])
        permno = ticker_permno.get(key)
        if permno is not None:
            cusips = cusip_map[
                (cusip_map["episode_id"] == row["episode_id"]) &
                (cusip_map["permno"] == permno)
            ]["issuer_cusip"]
            if not cusips.empty:
                cik_to_cusip[str(row["cik"])] = cusips.iloc[0]

    print(f"  CIK -> issuer_cusip mapping: {len(cik_to_cusip)} CIKs linked")

    # Map NLP CIK -> issuer_cusip
    all_nlp["issuer_cusip"] = all_nlp["cik"].astype(str).map(cik_to_cusip)

    # Quarterly aggregate NLP per issuer
    nlp_with_cusip = all_nlp.dropna(subset=["issuer_cusip"])
    if nlp_with_cusip.empty:
        print("  No NLP-bond matches.")
        return None, None

    nlp_quarterly = (
        nlp_with_cusip.groupby(["issuer_cusip", "quarter"])[NLP_FEATURES]
        .median()
        .reset_index()
    )

    # Merge NLP with bond spreads
    training = nlp_quarterly.merge(
        civ_quarterly[["issuer_cusip", "quarter", "spread", "leverage", "civ"]],
        on=["issuer_cusip", "quarter"],
        how="inner",
    )
    training = training.dropna(subset=NLP_FEATURES + ["spread", "leverage"])

    print(f"  Training set: {len(training)} firm-quarters, "
          f"{training.issuer_cusip.nunique()} issuers")

    return training, all_nlp


def train_spread_model(training):
    """Train ridge regression: NLP features -> log(spread)."""
    X = training[NLP_FEATURES].values
    y = np.log(training["spread"].values)  # log-spread for better regression

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    model.fit(X_scaled, y)

    # Cross-val R²
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    r2_mean = cv_scores.mean()
    r2_std = cv_scores.std()

    # Feature importance
    coef_df = pd.DataFrame({
        "feature": NLP_FEATURES,
        "coefficient": model.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)

    print(f"  Ridge CV R²: {r2_mean:.3f} +/- {r2_std:.3f}")
    print(f"  Best alpha: {model.alpha_:.1f}")
    print(f"  Top features:")
    for _, row in coef_df.head(5).iterrows():
        print(f"    {row['feature']:30s}: {row['coefficient']:+.4f}")

    # Save model for reproducibility
    import pickle
    model_path = CIV_DIR / "text_civ_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": NLP_FEATURES,
                      "alpha": model.alpha_, "r2_cv": r2_mean, "coef_df": coef_df}, f)
    print(f"  Model saved to {model_path}")

    return model, scaler


def _build_firm_leverage_lookup():
    """
    Build CIK -> monthly leverage lookup from CRSP + Compustat.

    Returns dict: cik_str -> DataFrame(date, leverage)
    """
    from config import RAW_DIR, DATA_DIR

    cik_map = pd.read_parquet(DATA_DIR / "sec" / "cik_mapping.parquet")
    cik_to_leverage = {}

    for eid in cik_map["episode_id"].unique():
        crsp_path = RAW_DIR / f"{eid}_crsp.parquet"
        fundq_path = RAW_DIR / f"{eid}_fundq.parquet"
        ccm_path = RAW_DIR / f"{eid}_ccm.parquet"
        firms_path = RAW_DIR / f"{eid}_firms.parquet"

        if not all(p.exists() for p in [crsp_path, fundq_path, ccm_path, firms_path]):
            continue

        crsp = pd.read_parquet(crsp_path)
        fundq = pd.read_parquet(fundq_path)
        ccm = pd.read_parquet(ccm_path)
        firms = pd.read_parquet(firms_path)

        ep_ciks = cik_map[
            (cik_map["episode_id"] == eid) & cik_map["cik"].notna()
        ]

        for _, crow in ep_ciks.iterrows():
            cik_str = str(crow["cik"])
            ticker = crow["ticker"]
            if cik_str in cik_to_leverage:
                continue  # already computed

            # Find permno for this ticker
            if "ticker" not in firms.columns or "permno" not in firms.columns:
                continue
            pmatch = firms[firms["ticker"] == ticker]
            if pmatch.empty:
                continue
            permno = int(pmatch["permno"].iloc[0])

            # Market cap from CRSP
            fc = crsp[crsp["permno"] == permno].copy()
            if fc.empty:
                continue
            fc["month"] = fc["date"].dt.to_period("M")
            mcap = fc.groupby("month").last()
            mcap["market_cap"] = mcap["price"] * mcap["shrout"] / 1000
            mcap = mcap[["market_cap"]].reset_index()
            mcap["date"] = mcap["month"].dt.to_timestamp("M")

            # Debt from Compustat
            gm = ccm[ccm["permno"] == permno]
            if gm.empty:
                continue
            gvkey = gm["gvkey"].iloc[0]
            ff = fundq[fundq["gvkey"] == gvkey].copy()
            if ff.empty:
                continue
            ff["month"] = ff["datadate"].dt.to_period("M")
            ff = ff.sort_values("datadate").drop_duplicates("month", keep="last")
            debt = ff.set_index("month")[["total_debt"]].resample("M").last().ffill().reset_index()
            debt["date"] = debt["month"].dt.to_timestamp("M")

            merged = mcap.merge(debt[["date", "total_debt"]], on="date", how="left")
            merged["total_debt"] = merged["total_debt"].ffill()
            merged["leverage"] = (
                merged["total_debt"] / (merged["total_debt"] + merged["market_cap"])
            ).clip(0.01, 0.99)

            cik_to_leverage[cik_str] = merged[["date", "leverage"]]

    return cik_to_leverage


def predict_text_civ(model, scaler, all_nlp):
    """
    Predict spread -> Merton-invert -> text-CIV for ALL filings.

    Produces two measures:
    1. predicted_spread — raw NLP-predicted credit spread (no Merton)
    2. text_civ — Merton inversion using ACTUAL firm leverage from Compustat
       (falls back to sector median leverage if firm-specific unavailable)
    """
    from src.merton import invert_spread_to_civ

    # Predict spread for every filing
    nlp_features = all_nlp[NLP_FEATURES].copy()
    valid = nlp_features.notna().all(axis=1)
    nlp_valid = nlp_features[valid]

    if nlp_valid.empty:
        return pd.DataFrame()

    X = scaler.transform(nlp_valid.values)
    log_spread_pred = model.predict(X)
    spread_pred = np.clip(np.exp(log_spread_pred), 1e-5, 0.10)

    result = all_nlp[valid].copy()
    result["predicted_spread"] = spread_pred

    # Build firm-level leverage lookup
    print("  Building firm leverage lookup from Compustat...")
    cik_to_leverage = _build_firm_leverage_lookup()
    print(f"  Leverage available for {len(cik_to_leverage)} CIKs")

    # Match each filing to its firm's leverage at filing date
    leverage_vals = np.full(len(result), np.nan)
    filing_dates = pd.to_datetime(result["filing_date"].values)
    cik_vals = result["cik"].astype(str).values

    for i, (cik, fdate) in enumerate(zip(cik_vals, filing_dates)):
        if cik not in cik_to_leverage:
            continue
        lev_df = cik_to_leverage[cik]
        # Find closest month
        month_end = fdate + pd.offsets.MonthEnd(0)
        match = lev_df[lev_df["date"] <= month_end]
        if not match.empty:
            leverage_vals[i] = match.iloc[-1]["leverage"]

    result["firm_leverage"] = leverage_vals

    # Fill missing leverage with sector median (from firms that have it)
    median_lev_by_episode = result.dropna(subset=["firm_leverage"]).groupby(
        "episode_id")["firm_leverage"].median()
    mask = result["firm_leverage"].isna()
    result.loc[mask, "firm_leverage"] = (
        result.loc[mask, "episode_id"].map(median_lev_by_episode)
    )
    # Fill any remaining with global default
    result["firm_leverage"] = result["firm_leverage"].fillna(0.40)

    # Merton inversion with ACTUAL leverage
    MATURITY = 5.0
    L = result["firm_leverage"].values
    tau = np.full(len(L), MATURITY)

    text_civ = invert_spread_to_civ(spread_pred, L, tau)
    result["text_civ"] = text_civ

    n_with_lev = (leverage_vals > 0).sum()  # before fill
    n_valid_civ = np.isfinite(text_civ).sum()
    print(f"  Firm-specific leverage matched: {n_with_lev}/{len(result)} "
          f"({100 * n_with_lev / len(result):.0f}%)")
    print(f"  Text-CIV computed: {n_valid_civ}/{len(result)} "
          f"({100 * n_valid_civ / len(result):.0f}%)")

    return result


def aggregate_sector_text_civ(text_civ_df):
    """
    Aggregate filing-level text-CIV to sector-monthly.

    Outputs both:
    - predicted_spread: raw NLP-predicted credit spread (sector median)
    - text_civ: leverage-adjusted Merton inversion (sector median)
    """
    if text_civ_df.empty:
        return {}

    df = text_civ_df.copy()
    df["quarter"] = df["filing_date"].dt.to_period("Q")

    results = {}
    for eid in sorted(df["episode_id"].unique()):
        ep_df = df[df["episode_id"] == eid]

        quarterly = (
            ep_df.groupby("quarter")
            .agg(
                text_civ=("text_civ", "median"),
                predicted_spread=("predicted_spread", "median"),
                firm_leverage=("firm_leverage", "median"),
                n_filings=("cik", "nunique"),
            )
            .reset_index()
        )
        quarterly["date"] = quarterly["quarter"].dt.to_timestamp("M")

        if quarterly.empty:
            continue

        # Expand quarterly to monthly via forward-fill
        start = quarterly["date"].min()
        end = quarterly["date"].max() + pd.DateOffset(months=3)
        monthly_idx = pd.date_range(start=start, end=end, freq="ME")
        monthly = pd.DataFrame({"date": monthly_idx})
        monthly = monthly.merge(quarterly.drop(columns=["quarter"]),
                                on="date", how="left").ffill()
        monthly["episode_id"] = eid

        out_path = CIV_DIR / f"{eid}_text_civ.parquet"
        monthly.to_parquet(out_path, index=False)
        results[eid] = monthly

        n_months = len(monthly.dropna(subset=["text_civ"]))
        med_civ = monthly["text_civ"].median()
        med_spread = monthly["predicted_spread"].median()
        print(f"  {eid:25s}: {n_months} months, "
              f"spread={med_spread:.4f}, text-CIV={med_civ:.1%}")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Text-CIV: SEC Filing -> Predicted Spread -> Merton Inversion")
    print("=" * 60)

    print("\nStep 1: Build training set (firms with NLP + bond data)")
    training, all_nlp = build_training_set()

    if training is None or len(training) < 50:
        print("  Insufficient training data for text-CIV model")
        return

    print(f"\nStep 2: Train NLP -> spread model")
    model, scaler = train_spread_model(training)

    print(f"\nStep 3: Predict text-CIV for all filings")
    text_civ_df = predict_text_civ(model, scaler, all_nlp)

    if text_civ_df.empty:
        print("  No text-CIV predictions")
        return

    text_civ_df.to_parquet(CIV_DIR / "text_civ_panel.parquet", index=False)
    print(f"  Saved text-CIV panel: {len(text_civ_df)} filings")

    print(f"\nStep 4: Aggregate to sector-monthly")
    results = aggregate_sector_text_civ(text_civ_df)

    n_episodes = len(results)
    print(f"\nDone. {n_episodes} episodes with text-CIV data")

    # Coverage comparison
    n_bond = len(list(CIV_DIR.glob("*_bond_civ.parquet")))
    n_text = len(list(CIV_DIR.glob("*_text_civ.parquet")))
    print(f"\nCoverage: bond-CIV={n_bond} episodes, text-CIV={n_text} episodes")


if __name__ == "__main__":
    main()
