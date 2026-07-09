"""
vol_measures.py — Firm-level monthly volatility and leverage panel.

From CRSP daily returns + Compustat quarterly fundamentals, computes:
  - realized_vol: 63-day rolling std of daily returns, annualized
  - market_cap: price * shrout / 1000 (millions $)
  - leverage: total_debt / (total_debt + market_cap), clipped (0.01, 0.99)
  - asset_vol: realized_vol * (1 - leverage) [Merton decomposition]
  - vol_of_vol: 6-month rolling std of realized_vol

Saves to data/measures/{episode_id}.parquet
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from config import DATA_DIR, RAW_DIR, VOL_WINDOW, MIN_DAYS_PER_MONTH, VOL_OF_VOL_WINDOW, ANNUALIZE

MEASURES_DIR = DATA_DIR / "measures"
MEASURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_firm_measures(episode_id):
    """Compute firm-level monthly measures for one episode."""
    out_path = MEASURES_DIR / f"{episode_id}.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        print(f"  {episode_id:25s}: CACHED ({len(df)} rows)")
        return df

    crsp_path = RAW_DIR / f"{episode_id}_crsp.parquet"
    fundq_path = RAW_DIR / f"{episode_id}_fundq.parquet"
    ccm_path = RAW_DIR / f"{episode_id}_ccm.parquet"

    if not crsp_path.exists():
        print(f"  {episode_id:25s}: NO CRSP DATA")
        return None

    crsp = pd.read_parquet(crsp_path)
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["ret"] = crsp["ret"].clip(-2.0, 2.0)  # cap extreme returns

    # Monthly realized vol per firm
    crsp["month"] = crsp["date"].dt.to_period("M")
    monthly_frames = []

    for permno, firm_df in crsp.groupby("permno"):
        firm_df = firm_df.sort_values("date")

        # Rolling realized vol (63-day window)
        firm_df = firm_df.copy()
        firm_df["rolling_vol"] = (
            firm_df["ret"].rolling(VOL_WINDOW, min_periods=MIN_DAYS_PER_MONTH).std()
            * ANNUALIZE
        )

        # Monthly aggregation: last day of month
        monthly = firm_df.groupby("month").agg(
            realized_vol=("rolling_vol", "last"),
            price=("price", "last"),
            shrout=("shrout", "last"),
            n_days=("ret", "count"),
        ).reset_index()

        # Filter months with too few days
        monthly = monthly[monthly["n_days"] >= MIN_DAYS_PER_MONTH]

        # Market cap (millions $)
        monthly["market_cap"] = monthly["price"] * monthly["shrout"] / 1000

        # Cap realized vol at 500% annualized
        monthly["realized_vol"] = monthly["realized_vol"].clip(upper=5.0)

        monthly["permno"] = permno
        monthly_frames.append(monthly)

    if not monthly_frames:
        print(f"  {episode_id:25s}: NO VALID MONTHS")
        return None

    measures = pd.concat(monthly_frames, ignore_index=True)
    measures["date"] = measures["month"].dt.to_timestamp("M")

    # Merge Compustat debt for leverage
    if fundq_path.exists() and ccm_path.exists():
        fundq = pd.read_parquet(fundq_path)
        ccm = pd.read_parquet(ccm_path)
        fundq["datadate"] = pd.to_datetime(fundq["datadate"])

        # Map permno -> gvkey
        permno_gvkey = ccm.drop_duplicates("permno", keep="last").set_index("permno")["gvkey"].to_dict()
        measures["gvkey"] = measures["permno"].map(permno_gvkey)

        # Forward-fill quarterly debt to monthly
        debt_frames = []
        for gvkey, gdf in fundq.groupby("gvkey"):
            gdf = gdf.sort_values("datadate").drop_duplicates("datadate", keep="last")
            gdf["month"] = gdf["datadate"].dt.to_period("M")
            gdf = gdf.drop_duplicates("month", keep="last")
            debt_monthly = gdf.set_index("month")[["total_debt", "total_assets"]].resample("M").last().ffill().reset_index()
            debt_monthly["date"] = debt_monthly["month"].dt.to_timestamp("M")
            debt_monthly["gvkey"] = gvkey
            debt_frames.append(debt_monthly[["date", "gvkey", "total_debt", "total_assets"]])

        if debt_frames:
            debt_all = pd.concat(debt_frames, ignore_index=True)
            measures = measures.merge(debt_all, on=["date", "gvkey"], how="left")
        else:
            measures["total_debt"] = np.nan
            measures["total_assets"] = np.nan

        # Leverage
        measures["leverage"] = (
            measures["total_debt"] / (measures["total_debt"] + measures["market_cap"])
        ).clip(0.01, 0.99)

        # Asset vol (Merton decomposition)
        measures["asset_vol"] = measures["realized_vol"] * (1 - measures["leverage"])
    else:
        measures["total_debt"] = np.nan
        measures["total_assets"] = np.nan
        measures["leverage"] = np.nan
        measures["asset_vol"] = np.nan

    # Vol of vol (6-month rolling std of realized_vol)
    vov_frames = []
    for permno, fdf in measures.groupby("permno"):
        fdf = fdf.sort_values("date").copy()
        fdf["vol_of_vol"] = fdf["realized_vol"].rolling(
            VOL_OF_VOL_WINDOW, min_periods=3
        ).std()
        vov_frames.append(fdf)
    measures = pd.concat(vov_frames, ignore_index=True)

    # Clean up
    keep_cols = ["date", "permno", "realized_vol", "market_cap", "leverage",
                 "asset_vol", "vol_of_vol", "total_debt", "total_assets", "n_days"]
    measures = measures[[c for c in keep_cols if c in measures.columns]]
    measures = measures.sort_values(["permno", "date"]).reset_index(drop=True)

    measures.to_parquet(out_path, index=False)

    n_firms = measures["permno"].nunique()
    n_months = measures["date"].nunique()
    lev_cov = measures["leverage"].notna().mean()
    print(f"  {episode_id:25s}: {len(measures)} rows, {n_firms} firms, "
          f"{n_months} months, leverage={lev_cov:.0%}")

    return measures


def compute_all():
    """Compute measures for all episodes."""
    from src.catalog import get_all
    episodes = get_all()
    print(f"Computing firm-level measures for {len(episodes)} episodes\n")

    for ep in sorted(episodes, key=lambda x: x["peak_date"] or "9999"):
        compute_firm_measures(ep["id"])

    n_done = len(list(MEASURES_DIR.glob("*.parquet")))
    print(f"\nDone. {n_done} episodes with measures")


if __name__ == "__main__":
    compute_all()
