"""
05_compute_metrics.py — Compute all 20 sector-level metrics (A-T) per episode.

Combines:
  A-F: Volatility-structural (from firm measures + CRSP daily)
  G-N: SEC textual (from NLP scores)
  O-Q: Bond-CIV (from TRACE bond spreads)
  R-S: Text-CIV (from NLP-predicted spreads + actual leverage)

Output: data/metrics/{episode_id}.parquet (one combined file per episode)
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import (DATA_DIR, RAW_DIR, MEASURES_DIR, METRICS_DIR,
                    VOL_WINDOW, CORR_WINDOW, MIN_FIRMS_CORR, MIN_DAYS_CORR)
from src.catalog import get_all

METRICS_DIR = DATA_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEC_METRICS_DIR = DATA_DIR / "sec" / "nlp_scores"
CIV_DIR = DATA_DIR / "civ"


# ── Vol Metrics A-F ──────────────────────────────────────────────────

def compute_market_vol():
    mkt_path = RAW_DIR / "market_daily.parquet"
    if not mkt_path.exists():
        return pd.DataFrame()
    mkt = pd.read_parquet(mkt_path)
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt["month"] = mkt["date"].dt.to_period("M")
    market_vol = (
        mkt.groupby("month")["vwretd"]
        .agg(lambda x: x.std() * np.sqrt(252) if len(x) >= 15 else np.nan)
        .rename("market_vol").reset_index()
    )
    market_vol["date"] = market_vol["month"].dt.to_timestamp("M")
    return market_vol


def metric_a(measures, market_vol):
    sector = measures.groupby("date")["realized_vol"].median().rename("sector_vol")
    merged = sector.reset_index().merge(market_vol[["date", "market_vol"]], on="date", how="left")
    merged["metric_a"] = merged["sector_vol"] / merged["market_vol"]
    return merged[["date", "sector_vol", "market_vol", "metric_a"]]


def metric_b(measures):
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


def metric_c(episode_id):
    crsp_path = RAW_DIR / f"{episode_id}_crsp.parquet"
    if not crsp_path.exists():
        return pd.DataFrame()
    daily = pd.read_parquet(crsp_path)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ret"] = daily["ret"].clip(-2.0, 2.0)
    daily["month"] = daily["date"].dt.to_period("M")
    months = sorted(daily["month"].unique())
    results = []
    for month in months:
        month_end = month.to_timestamp("M")
        month_start = month_end - pd.Timedelta(days=100)
        window = daily[(daily["date"] >= month_start) & (daily["date"] <= month_end)]
        window = window.sort_values("date").groupby("permno").tail(CORR_WINDOW)
        pivot = window.pivot_table(index="date", columns="permno", values="ret")
        valid_firms = pivot.columns[pivot.notna().sum() >= MIN_DAYS_CORR]
        if len(valid_firms) < MIN_FIRMS_CORR:
            results.append({"month": month, "metric_c": np.nan, "n_firms_corr": 0})
            continue
        corr_matrix = pivot[valid_firms].corr()
        n = len(valid_firms)
        upper = corr_matrix.values[np.triu_indices(n, k=1)]
        results.append({"month": month, "metric_c": np.nanmean(upper), "n_firms_corr": n})
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["date"] = result_df["month"].dt.to_timestamp("M")
    return result_df


def metric_d(measures):
    return measures.groupby("date")["vol_of_vol"].median().rename("metric_d").reset_index()


def metric_e(measures):
    monthly_assets = measures.groupby("date")["total_assets"].median().rename("med_assets").reset_index().sort_values("date")
    monthly_assets["metric_e"] = monthly_assets["med_assets"].pct_change(12)
    return monthly_assets[["date", "metric_e"]]


def metric_f(measures):
    monthly_lev = measures.groupby("date")["leverage"].median().rename("med_leverage").reset_index().sort_values("date")
    monthly_lev["metric_f"] = monthly_lev["med_leverage"].diff(12)
    return monthly_lev[["date", "med_leverage", "metric_f"]]


# ── SEC Metrics G-N ──────────────────────────────────────────────────

METRIC_MAP = {
    "metric_g": "lm_net_sentiment",
    "metric_h": "lm_uncertainty_pct",
    "metric_i": "fog_index",
    "metric_j": "offbalance_per10k",
    "metric_k": "risk_escalation_per10k",
    "metric_l": "growth_narrative_per10k",
    "metric_n": "lm_negativity_ratio",
}


def compute_sec_metrics(episode_id):
    nlp_path = SEC_METRICS_DIR / f"{episode_id}.parquet"
    if not nlp_path.exists():
        return pd.DataFrame()
    nlp = pd.read_parquet(nlp_path)
    nlp["filing_date"] = pd.to_datetime(nlp["filing_date"])
    nlp["quarter"] = nlp["filing_date"].dt.to_period("Q")
    nlp = nlp.sort_values("filing_date").groupby(["cik", "quarter"]).last().reset_index()

    agg_cols = list(METRIC_MAP.values()) + ["word_count"]
    agg_cols = [c for c in agg_cols if c in nlp.columns]
    quarterly = nlp.groupby("quarter")[agg_cols].median().reset_index()
    quarterly["n_filings"] = nlp.groupby("quarter")["cik"].nunique().values
    quarterly = quarterly.sort_values("quarter")

    # Expand to monthly
    if quarterly.empty:
        return pd.DataFrame()
    quarterly["date"] = quarterly["quarter"].dt.to_timestamp("M")
    start = quarterly["date"].min()
    end = quarterly["date"].max() + pd.DateOffset(months=3)
    monthly_idx = pd.date_range(start=start, end=end, freq="ME")
    monthly = pd.DataFrame({"date": monthly_idx})
    monthly = monthly.merge(quarterly.drop(columns=["quarter"]), on="date", how="left").ffill()

    # Rename to metric labels
    for metric_name, nlp_col in METRIC_MAP.items():
        if nlp_col in monthly.columns:
            monthly[metric_name] = monthly[nlp_col]

    # Filing length trend (metric_m)
    if "word_count" in monthly.columns:
        monthly["metric_m"] = monthly["word_count"].pct_change(12)

    return monthly


# ── CIV Metrics O-Q (bond) and R-S (text) ───────────────────────────

def compute_civ_metrics(episode_id, measures):
    result = pd.DataFrame()

    # Bond-CIV
    bond_path = CIV_DIR / f"{episode_id}_bond_civ.parquet"
    if bond_path.exists():
        bond = pd.read_parquet(bond_path)
        bond["date"] = pd.to_datetime(bond["date"])
        bond = bond.rename(columns={"civ_median": "metric_o"})
        bond["metric_q"] = bond["metric_o"].diff(12)  # 12-month trajectory
        result = bond[["date", "metric_o", "metric_q", "n_firms_civ"]]

    # CIV-equity wedge (metric_p)
    if not result.empty and not measures.empty:
        equity_avol = measures.groupby("date")["asset_vol"].median().rename("equity_asset_vol").reset_index()
        result = result.merge(equity_avol, on="date", how="left")
        result["metric_p"] = result["metric_o"] - result["equity_asset_vol"]

    # Text-CIV
    text_path = CIV_DIR / f"{episode_id}_text_civ.parquet"
    if text_path.exists():
        text_civ = pd.read_parquet(text_path)
        text_civ["date"] = pd.to_datetime(text_civ["date"])
        text_civ = text_civ.rename(columns={"text_civ": "metric_r", "predicted_spread": "metric_r_spread"})
        text_civ["metric_s"] = text_civ["metric_r"].diff(12)  # trajectory

        if result.empty:
            result = text_civ[["date", "metric_r", "metric_r_spread", "metric_s"]]
        else:
            text_cols = ["date", "metric_r", "metric_r_spread", "metric_s"]
            result = result.merge(text_civ[[c for c in text_cols if c in text_civ.columns]],
                                  on="date", how="outer")

    return result


# ── Main: combine all metrics ────────────────────────────────────────

def process_episode(episode_id, market_vol):
    out_path = METRICS_DIR / f"{episode_id}.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        print(f"  {episode_id:25s}: CACHED ({len(df)} months, "
              f"{sum(1 for c in df.columns if c.startswith('metric_'))} metrics)")
        return df

    measures_path = DATA_DIR / "measures" / f"{episode_id}.parquet"
    if not measures_path.exists():
        print(f"  {episode_id:25s}: NO MEASURES")
        return None

    measures = pd.read_parquet(measures_path)

    # Vol metrics A-F
    a = metric_a(measures, market_vol)
    b = metric_b(measures)
    c = metric_c(episode_id)
    d = metric_d(measures)
    e = metric_e(measures)
    f = metric_f(measures)

    result = a[["date", "sector_vol", "market_vol", "metric_a"]].copy()
    for df_m, cols in [(b, ["metric_b"]), (d, ["metric_d"]),
                       (e, ["metric_e"]), (f, ["med_leverage", "metric_f"])]:
        if not df_m.empty and "date" in df_m.columns:
            result = result.merge(df_m[["date"] + cols], on="date", how="left")
    if not c.empty and "date" in c.columns:
        result = result.merge(c[["date", "metric_c", "n_firms_corr"]], on="date", how="left")
    else:
        result["metric_c"] = np.nan
        result["n_firms_corr"] = 0

    # SEC metrics G-N
    sec = compute_sec_metrics(episode_id)
    if not sec.empty:
        sec_cols = [c for c in sec.columns if c.startswith("metric_") or c == "date"]
        result = result.merge(sec[sec_cols], on="date", how="left")

    # CIV metrics O-S
    civ = compute_civ_metrics(episode_id, measures)
    if not civ.empty:
        civ_cols = [c for c in civ.columns if c.startswith("metric_") or c == "date" or c == "n_firms_civ"]
        result = result.merge(civ[civ_cols], on="date", how="left")

    # ── Leverage-conditional interaction features (U-X) ─────────────
    # These let the model learn that vol means different things at
    # different leverage levels (fragility vs mania)
    lev = result["med_leverage"].fillna(0.0) if "med_leverage" in result.columns else 0.0

    # U: Leverage level (the level itself, not trajectory)
    result["metric_u"] = lev

    # V: Fragility vol = vol_ratio × leverage (high for levered volatile sectors)
    result["metric_v"] = result["metric_a"].fillna(0) * lev

    # W: Mania vol = vol_ratio × (1 - leverage) (high for unlevered volatile sectors)
    result["metric_w"] = result["metric_a"].fillna(0) * (1 - lev)

    # X: Fragility instability = vol_of_vol × leverage
    result["metric_x"] = result["metric_d"].fillna(0) * lev

    result["episode_id"] = episode_id
    result = result.sort_values("date").reset_index(drop=True)
    result.to_parquet(out_path, index=False)

    n_metrics = sum(1 for c in result.columns if c.startswith("metric_"))
    coverage = {c: result[c].notna().mean() for c in result.columns if c.startswith("metric_")}
    good = sum(1 for v in coverage.values() if v > 0.3)
    print(f"  {episode_id:25s}: {len(result)} months, {n_metrics} metrics ({good} with >30% coverage)")

    return result


def main():
    episodes = get_all()
    print(f"Computing sector metrics for {len(episodes)} episodes\n")

    market_vol = compute_market_vol()
    if market_vol.empty:
        print("ERROR: No market daily data")
        return

    for ep in sorted(episodes, key=lambda x: x["peak_date"] or "9999"):
        process_episode(ep["id"], market_vol)

    n_done = len(list(METRICS_DIR.glob("*.parquet")))
    print(f"\nDone. {n_done} episodes with metrics")


if __name__ == "__main__":
    main()
