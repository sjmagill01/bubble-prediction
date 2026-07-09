"""
02_fetch_equity.py — Pull CRSP daily returns + Compustat quarterly from WRDS.

For each episode:
  1. Resolve tickers → permnos via CRSP msenames
  2. Pull CRSP daily (permno, date, ret, price, shrout)
  3. Pull CCM link (permno → gvkey)
  4. Pull Compustat quarterly (gvkey, datadate, total_debt, total_assets)

Window: episode_start - 3 years to episode_end + 2 years.
All data pulled fresh from WRDS.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.wrds_utils import _connect

import pandas as pd
from sqlalchemy import text
from config import DATA_DIR, RAW_DIR
from src.catalog import get_all

RAW_DIR.mkdir(parents=True, exist_ok=True)

PRE_BUFFER = 3
POST_BUFFER = 2
CHUNK_SIZE = 200


def episode_window(ep):
    start = ep["period"][0] - PRE_BUFFER
    peak = ep["period"][1] or 2026
    end = (ep["period"][2] or peak) + POST_BUFFER
    return f"{start}-01-01", f"{end}-12-31"


def resolve_tickers_to_permnos(conn, tickers, start, end):
    """Map tickers to CRSP permnos using msenames."""
    ticker_list = ", ".join(f"'{t}'" for t in tickers)
    sql = text(f"""
    SELECT DISTINCT ticker, permno, comnam, siccd,
           namedt, nameendt
    FROM crsp.msenames
    WHERE ticker IN ({ticker_list})
      AND shrcd IN (10, 11, 12)
      AND namedt <= '{end}'
      AND nameendt >= '{start}'
    ORDER BY ticker, permno
    """)
    with conn.engine.connect() as c:
        df = pd.read_sql(sql, c)

    if df.empty:
        return pd.DataFrame(), []  # (firms_df, permnos_list)

    # Pick the permno with longest coverage per ticker
    df["span"] = (pd.to_datetime(df["nameendt"]) - pd.to_datetime(df["namedt"])).dt.days
    best = df.sort_values("span", ascending=False).drop_duplicates("ticker", keep="first")
    permnos = sorted(best["permno"].unique().tolist())
    return best, permnos


def fetch_crsp_daily(conn, permnos, start, end):
    frames = []
    for i in range(0, len(permnos), CHUNK_SIZE):
        chunk = permnos[i:i + CHUNK_SIZE]
        permno_list = ", ".join(str(p) for p in chunk)
        sql = text(f"""
        SELECT permno, date, ret, ABS(prc) AS price, shrout
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date >= '{start}' AND date <= '{end}'
          AND ret IS NOT NULL
        ORDER BY permno, date
        """)
        with conn.engine.connect() as c:
            frames.append(pd.read_sql(sql, c))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_ccm_and_fundq(conn, permnos, start, end):
    # CCM link
    frames = []
    for i in range(0, len(permnos), CHUNK_SIZE):
        chunk = permnos[i:i + CHUNK_SIZE]
        permno_list = ", ".join(str(p) for p in chunk)
        sql = text(f"""
        SELECT DISTINCT a.gvkey, b.permno
        FROM crsp_a_ccm.ccmxpf_lnkhist a
        JOIN (SELECT unnest(ARRAY[{permno_list}]) AS permno) b
          ON a.lpermno = b.permno
        WHERE a.linktype IN ('LC', 'LU', 'LS')
          AND a.linkprim IN ('P', 'C')
          AND a.linkdt <= '{end}'
          AND COALESCE(a.linkenddt, '2026-12-31') >= '{start}'
        """)
        with conn.engine.connect() as c:
            frames.append(pd.read_sql(sql, c))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    ccm = pd.concat(frames, ignore_index=True).drop_duplicates()
    gvkeys = ccm["gvkey"].unique().tolist()

    if not gvkeys:
        return ccm, pd.DataFrame()

    # Compustat quarterly
    frames = []
    for i in range(0, len(gvkeys), CHUNK_SIZE):
        chunk = gvkeys[i:i + CHUNK_SIZE]
        gvkey_list = ", ".join(f"'{g}'" for g in chunk)
        sql = text(f"""
        SELECT gvkey, datadate,
               COALESCE(dlcq, 0) + COALESCE(dlttq, 0) AS total_debt,
               atq AS total_assets,
               cshoq AS shares_outstanding
        FROM comp.fundq
        WHERE gvkey IN ({gvkey_list})
          AND datadate >= '{start}' AND datadate <= '{end}'
          AND indfmt = 'INDL' AND datafmt = 'STD'
          AND popsrc = 'D' AND consol = 'C'
        ORDER BY gvkey, datadate
        """)
        with conn.engine.connect() as c:
            frames.append(pd.read_sql(sql, c))
    if not frames:
        return ccm, pd.DataFrame()
    fundq = pd.concat(frames, ignore_index=True)
    fundq["datadate"] = pd.to_datetime(fundq["datadate"])
    return ccm, fundq


def fetch_market_daily(conn):
    """Fetch CRSP value-weighted market daily returns."""
    out_path = RAW_DIR / "market_daily.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        print(f"Market daily: CACHED ({len(df):,} rows)")
        return df

    sql = text("""
    SELECT date, vwretd
    FROM crsp.dsi
    WHERE date >= '1990-01-01'
    ORDER BY date
    """)
    with conn.engine.connect() as c:
        df = pd.read_sql(sql, c)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(out_path, index=False)
    print(f"Market daily: {len(df):,} rows ({df.date.min().date()} to {df.date.max().date()})")
    return df


def process_episode(conn, ep):
    eid = ep["id"]
    crsp_path = RAW_DIR / f"{eid}_crsp.parquet"
    fundq_path = RAW_DIR / f"{eid}_fundq.parquet"
    ccm_path = RAW_DIR / f"{eid}_ccm.parquet"
    firms_path = RAW_DIR / f"{eid}_firms.parquet"

    if crsp_path.exists() and fundq_path.exists():
        crsp = pd.read_parquet(crsp_path)
        print(f"  {eid:25s}: CACHED ({len(crsp):,} CRSP rows)")
        return

    start, end = episode_window(ep)
    tickers = ep["tickers"]

    # Resolve tickers
    firms_df, permnos = resolve_tickers_to_permnos(conn, tickers, start, end)
    if not permnos:
        print(f"  {eid:25s}: NO permnos found for tickers {tickers}")
        return

    # Save firms mapping
    firms_df.to_parquet(firms_path, index=False)

    # CRSP daily
    crsp = fetch_crsp_daily(conn, permnos, start, end)
    if crsp.empty:
        print(f"  {eid:25s}: NO CRSP data")
        return
    crsp.to_parquet(crsp_path, index=False)

    # CCM + Compustat
    ccm, fundq = fetch_ccm_and_fundq(conn, permnos, start, end)
    if not ccm.empty:
        ccm.to_parquet(ccm_path, index=False)
    if not fundq.empty:
        fundq.to_parquet(fundq_path, index=False)

    n_permnos = crsp["permno"].nunique()
    n_gvkeys = len(ccm["gvkey"].unique()) if not ccm.empty else 0
    n_fundq = len(fundq)
    print(f"  {eid:25s}: {len(crsp):,} CRSP rows ({n_permnos} permnos), "
          f"{n_fundq:,} fundq rows ({n_gvkeys} gvkeys)")


def main():
    episodes = get_all()
    print(f"Fetching equity data for {len(episodes)} episodes")
    print("=" * 60)

    conn = _connect()

    # Market daily first
    fetch_market_daily(conn)

    # Each episode
    for ep in sorted(episodes, key=lambda x: x["peak_date"] or "9999"):
        try:
            process_episode(conn, ep)
        except Exception as e:
            print(f"  {ep['id']:25s}: ERROR — {e}")
            # Reconnect in case of dropped connection
            try:
                conn.close()
            except Exception:
                pass
            conn = _connect()

    conn.close()

    # Summary
    n_crsp = len(list(RAW_DIR.glob("*_crsp.parquet")))
    print(f"\nDone. {n_crsp} episodes with CRSP data in {RAW_DIR}")


if __name__ == "__main__":
    main()
