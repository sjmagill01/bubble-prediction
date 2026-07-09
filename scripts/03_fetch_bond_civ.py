"""
03_fetch_bond_civ.py — Pull bond spreads from WRDS and compute bond-CIV.

Pipeline:
  1. Map bubble firm tickers → issuer CUSIPs (via permno → bondcrsp_link)
  2. Pull bond spreads from wrdsapps_bondret.bondret
  3. Aggregate to firm × month × maturity bucket
  4. Merge leverage from Compustat
  5. Merton-invert spread → CIV
  6. Aggregate to sector-monthly metrics

All data pulled fresh from WRDS.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.wrds_utils import _connect

import pandas as pd
import numpy as np
from sqlalchemy import text
from config import DATA_DIR, RAW_DIR

CIV_DIR = DATA_DIR / "civ"
CIV_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 200
MATURITY_BUCKETS = [3.0, 5.0, 10.0]
MATURITY_EDGES = [0, 4, 7, 30]


# ── Step 1: Map tickers → issuer CUSIPs ──────────────────────────────

def build_issuer_cusip_mapping(conn):
    """Map bubble firm permnos → issuer CUSIPs via bondcrsp_link."""
    cache = CIV_DIR / "issuer_cusip_mapping.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"Issuer CUSIP mapping: CACHED ({len(df)} firms)")
        return df

    # Get all permnos from our equity data
    from src.catalog import get_all
    all_permnos = set()
    permno_to_episode = {}

    for ep in get_all():
        firms_path = RAW_DIR / f"{ep['id']}_firms.parquet"
        if not firms_path.exists():
            continue
        firms = pd.read_parquet(firms_path)
        if "permno" in firms.columns:
            for p in firms["permno"].unique():
                all_permnos.add(int(p))
                permno_to_episode.setdefault(int(p), []).append(ep["id"])

    if not all_permnos:
        print("  No permnos found in firms files")
        return pd.DataFrame()

    permnos = sorted(all_permnos)
    print(f"  Looking up {len(permnos)} permnos in bondcrsp_link...")

    # Query bondcrsp_link — table has bond-level cusip (9-char), we derive
    # issuer_cusip as first 6 characters
    frames = []
    for i in range(0, len(permnos), CHUNK_SIZE):
        chunk = permnos[i:i + CHUNK_SIZE]
        permno_list = ", ".join(str(p) for p in chunk)
        sql = text(f"""
        SELECT DISTINCT SUBSTRING(cusip, 1, 6) AS issuer_cusip, permno
        FROM wrdsapps.bondcrsp_link
        WHERE permno IN ({permno_list})
          AND cusip IS NOT NULL
        """)
        with conn.engine.connect() as c:
            frames.append(pd.read_sql(sql, c))

    if not frames:
        print("  No bond-CRSP links found")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).drop_duplicates()

    # Add episode mapping
    rows = []
    for _, row in df.iterrows():
        p = int(row["permno"])
        for eid in permno_to_episode.get(p, []):
            rows.append({
                "episode_id": eid,
                "permno": p,
                "issuer_cusip": row["issuer_cusip"],
            })
    result = pd.DataFrame(rows)
    result.to_parquet(cache, index=False)

    n_eps = result["episode_id"].nunique()
    n_issuers = result["issuer_cusip"].nunique()
    print(f"  Mapped {n_issuers} issuer CUSIPs across {n_eps} episodes")

    return result


# ── Step 2: Pull bond spreads ────────────────────────────────────────

def fetch_bondret(conn, issuer_cusips, start, end):
    """Pull bond spread data from wrdsapps_bondret.bondret."""
    frames = []
    cusips = sorted(set(issuer_cusips))

    for i in range(0, len(cusips), CHUNK_SIZE):
        chunk = cusips[i:i + CHUNK_SIZE]
        cusip_list = ", ".join(f"'{c}'" for c in chunk)
        sql = text(f"""
        SELECT date, cusip, SUBSTRING(cusip, 1, 6) AS issuer_cusip,
               t_spread, tmt, rating_num, rating_cat, amount_outstanding
        FROM wrdsapps_bondret.bondret
        WHERE SUBSTRING(cusip, 1, 6) IN ({cusip_list})
          AND date >= '{start}' AND date <= '{end}'
          AND t_spread IS NOT NULL
          AND t_spread > 0
          AND t_spread <= 0.10
        ORDER BY cusip, date
        """)
        with conn.engine.connect() as c:
            frames.append(pd.read_sql(sql, c))

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_to_firm_monthly(bondret_df):
    """Aggregate bond-level spreads to firm × month × maturity bucket."""
    df = bondret_df.copy()

    # Assign maturity bucket
    df["mat_bucket"] = pd.cut(
        df["tmt"], bins=MATURITY_EDGES, labels=MATURITY_BUCKETS,
        include_lowest=True
    ).astype(float)
    df = df.dropna(subset=["mat_bucket"])

    # Weighted average spread by issuer × month × maturity bucket
    df["weight"] = df["amount_outstanding"].fillna(1.0)

    def _agg_group(g):
        w = g["weight"].values
        if w.sum() <= 0:
            w = np.ones(len(w))
        return pd.Series({
            "spread": np.average(g["t_spread"].values, weights=w),
            "n_bonds": len(g),
            "rating": g["rating_cat"].mode().iloc[0] if len(g["rating_cat"].mode()) > 0 else None,
        })

    agg = (
        df.groupby(["issuer_cusip", "date", "mat_bucket"])
        .apply(_agg_group, include_groups=False)
        .reset_index()
    )
    return agg


# ── Step 3: Merge leverage and compute CIV ───────────────────────────

def merge_leverage_and_invert(firm_monthly, cusip_mapping):
    """Merge leverage from Compustat and Merton-invert to CIV."""
    from src.merton import invert_spread_to_civ

    # Get leverage for each issuer_cusip via permno → equity data
    # We already have fundq + crsp data in RAW_DIR
    leverage_rows = []
    for eid in cusip_mapping["episode_id"].unique():
        crsp_path = RAW_DIR / f"{eid}_crsp.parquet"
        fundq_path = RAW_DIR / f"{eid}_fundq.parquet"
        ccm_path = RAW_DIR / f"{eid}_ccm.parquet"

        if not all(p.exists() for p in [crsp_path, fundq_path, ccm_path]):
            continue

        crsp = pd.read_parquet(crsp_path)
        fundq = pd.read_parquet(fundq_path)
        ccm = pd.read_parquet(ccm_path)

        # Get permnos for this episode's issuer cusips
        ep_cusips = cusip_mapping[cusip_mapping["episode_id"] == eid]

        for _, row in ep_cusips.iterrows():
            permno = row["permno"]
            issuer = row["issuer_cusip"]

            # Market cap from CRSP (month-end)
            firm_crsp = crsp[crsp["permno"] == permno].copy()
            if firm_crsp.empty:
                continue
            firm_crsp["month"] = firm_crsp["date"].dt.to_period("M")
            mcap = (
                firm_crsp.groupby("month")
                .last()
                .assign(market_cap=lambda x: x["price"] * x["shrout"] / 1000)  # millions
                [["market_cap"]]
                .reset_index()
            )
            mcap["date"] = mcap["month"].dt.to_timestamp("M")

            # Debt from Compustat
            gvkey_match = ccm[ccm["permno"] == permno]
            if gvkey_match.empty:
                continue
            gvkey = gvkey_match["gvkey"].iloc[0]
            firm_fundq = fundq[fundq["gvkey"] == gvkey].copy()
            if firm_fundq.empty:
                continue
            firm_fundq["month"] = firm_fundq["datadate"].dt.to_period("M")
            firm_fundq = firm_fundq.sort_values("datadate").drop_duplicates("month", keep="last")

            # Forward-fill quarterly debt to monthly
            debt_monthly = (
                firm_fundq.set_index("month")[["total_debt"]]
                .resample("M").last().ffill()
                .reset_index()
            )
            debt_monthly["date"] = debt_monthly["month"].dt.to_timestamp("M")

            # Merge
            merged = mcap.merge(debt_monthly[["date", "total_debt"]], on="date", how="left")
            merged["total_debt"] = merged["total_debt"].ffill()
            merged["leverage"] = merged["total_debt"] / (
                merged["total_debt"] + merged["market_cap"]
            )
            merged["leverage"] = merged["leverage"].clip(0.01, 0.99)
            merged["issuer_cusip"] = issuer

            leverage_rows.append(merged[["date", "issuer_cusip", "leverage", "market_cap"]])

    if not leverage_rows:
        return pd.DataFrame()

    leverage_df = pd.concat(leverage_rows, ignore_index=True)
    leverage_df = leverage_df.drop_duplicates(["date", "issuer_cusip"], keep="last")

    # Merge onto firm_monthly bond data
    merged = firm_monthly.merge(leverage_df, on=["date", "issuer_cusip"], how="left")
    merged = merged.dropna(subset=["spread", "leverage", "mat_bucket"])

    if merged.empty:
        return pd.DataFrame()

    # Merton inversion
    merged["civ"] = invert_spread_to_civ(
        merged["spread"].values,
        merged["leverage"].values,
        merged["mat_bucket"].values,
    )

    n_valid = merged["civ"].notna().sum()
    print(f"  CIV inversion: {n_valid}/{len(merged)} success "
          f"({100 * n_valid / len(merged):.0f}%)")

    return merged


# ── Step 4: Aggregate to sector-monthly ──────────────────────────────

def compute_sector_civ(civ_panel, cusip_mapping):
    """Aggregate firm-level CIV to sector-monthly for each episode."""
    from src.catalog import get_all

    episodes = {e["id"]: e for e in get_all()}
    results = {}

    for eid in sorted(cusip_mapping["episode_id"].unique()):
        ep_cusips = cusip_mapping[cusip_mapping["episode_id"] == eid]["issuer_cusip"].unique()
        ep_civ = civ_panel[civ_panel["issuer_cusip"].isin(ep_cusips)]

        if ep_civ.empty:
            continue

        # Prefer 5-year maturity, else median across maturities
        monthly = (
            ep_civ.groupby(["issuer_cusip", "date"])
            .apply(
                lambda g: g[g["mat_bucket"] == 5.0]["civ"].median()
                if (g["mat_bucket"] == 5.0).any()
                else g["civ"].median(),
                include_groups=False,
            )
            .rename("civ")
            .reset_index()
        )

        sector = (
            monthly.groupby("date")
            .agg(
                civ_median=("civ", "median"),
                n_firms_civ=("issuer_cusip", "nunique"),
            )
            .reset_index()
        )
        sector["episode_id"] = eid

        # Save per-episode
        out_path = CIV_DIR / f"{eid}_bond_civ.parquet"
        sector.to_parquet(out_path, index=False)
        results[eid] = sector

        n_months = len(sector)
        med_firms = sector["n_firms_civ"].median()
        med_civ = sector["civ_median"].median()
        print(f"  {eid:25s}: {n_months} months, {med_firms:.0f} firms, "
              f"CIV={med_civ:.1%}")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 1: Map tickers to issuer CUSIPs")
    print("=" * 60)
    conn = _connect()
    cusip_mapping = build_issuer_cusip_mapping(conn)

    if cusip_mapping.empty:
        print("No issuer CUSIP mappings. Exiting.")
        conn.close()
        return

    print(f"\n{'=' * 60}")
    print("Step 2: Pull bond spreads from WRDS")
    print("=" * 60)

    all_cusips = cusip_mapping["issuer_cusip"].unique().tolist()
    print(f"  Pulling bondret for {len(all_cusips)} issuer CUSIPs...")

    bondret = fetch_bondret(conn, all_cusips, "1993-01-01", "2026-12-31")
    conn.close()

    if bondret.empty:
        print("  No bond data found")
        return

    print(f"  Bondret: {len(bondret):,} rows, "
          f"{bondret.issuer_cusip.nunique()} issuers, "
          f"{bondret.date.min().date()} to {bondret.date.max().date()}")

    # Save raw bondret
    bondret.to_parquet(CIV_DIR / "bondret_raw.parquet", index=False)

    print(f"\n{'=' * 60}")
    print("Step 3: Aggregate to firm × month × maturity")
    print("=" * 60)
    firm_monthly = aggregate_to_firm_monthly(bondret)
    print(f"  Firm-monthly: {len(firm_monthly):,} rows, "
          f"{firm_monthly.issuer_cusip.nunique()} issuers")

    print(f"\n{'=' * 60}")
    print("Step 4: Merge leverage and Merton-invert")
    print("=" * 60)
    civ_panel = merge_leverage_and_invert(firm_monthly, cusip_mapping)

    if civ_panel.empty:
        print("  No CIV data produced")
        return

    civ_panel.to_parquet(CIV_DIR / "civ_panel.parquet", index=False)
    print(f"  CIV panel: {len(civ_panel):,} rows, "
          f"{civ_panel.issuer_cusip.nunique()} firms")

    print(f"\n{'=' * 60}")
    print("Step 5: Aggregate to sector-monthly")
    print("=" * 60)
    compute_sector_civ(civ_panel, cusip_mapping)

    # Summary
    n_files = len(list(CIV_DIR.glob("*_bond_civ.parquet")))
    print(f"\nDone. {n_files} episodes with bond-CIV data")


if __name__ == "__main__":
    main()
