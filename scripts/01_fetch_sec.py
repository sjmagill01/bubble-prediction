"""
01_fetch_sec.py — Pull SEC filings from EDGAR for all 59 episodes.

Three phases:
  1. Map tickers → CIKs (SEC bulk ticker mapping + WRDS fallback)
  2. Fetch filing indices from EDGAR submissions API (10-K, 10-Q)
  3. Download and clean filing text
  4. Run L-M NLP scoring

All data pulled fresh. No cache reuse.
"""
import sys
import os
import re
import json
import time
import requests
from pathlib import Path
from html.parser import HTMLParser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import DATA_DIR, SEC_DIR, SEC_USER_AGENT, SEC_RATE_LIMIT

INDEX_DIR = SEC_DIR / "filings_index"
TEXT_DIR = SEC_DIR / "filings_text"
NLP_DIR = SEC_DIR / "nlp_scores"

for d in [SEC_DIR, INDEX_DIR, TEXT_DIR, NLP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}


# ── HTML Stripper ─────────────────────────────────────────────────────

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            self.result.append(data)

    def get_text(self):
        return " ".join(self.result)


def strip_html(html_text):
    stripper = HTMLStripper()
    try:
        stripper.feed(html_text)
        return stripper.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html_text)
        return re.sub(r"\s+", " ", text).strip()


# ── Phase 1: CIK Mapping ─────────────────────────────────────────────

def load_sec_ticker_mapping():
    """Load SEC bulk CIK-ticker mapping from EDGAR."""
    cache = SEC_DIR / "sec_cik_tickers.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    print("  Downloading SEC ticker-CIK mapping...")
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=SEC_HEADERS)
    r.raise_for_status()
    data = r.json()

    mapping = {}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"])
        mapping[ticker] = cik

    with open(cache, "w") as f:
        json.dump(mapping, f)
    print(f"  SEC ticker map: {len(mapping)} tickers")
    return mapping


def build_cik_mapping():
    """Map all episode tickers to SEC CIKs."""
    from src.catalog import get_all

    sec_map = load_sec_ticker_mapping()

    # Try WRDS linkage as supplement
    wrds_map = {}
    try:
        from src.wrds_utils import _connect
        from sqlalchemy import text
        conn = _connect()
        with conn.engine.connect() as c:
            df = pd.read_sql(text("""
                SELECT DISTINCT cik, ticker
                FROM wrdssec_common.wciklink_gvkey w
                JOIN comp.company c ON w.gvkey = c.gvkey
                WHERE cik IS NOT NULL AND ticker IS NOT NULL
            """), c)
        conn.close()
        for _, row in df.iterrows():
            t = str(row["ticker"]).upper()
            if t not in wrds_map:
                wrds_map[t] = str(row["cik"])
        print(f"  WRDS CIK supplement: {len(wrds_map)} tickers")
    except Exception as e:
        print(f"  WRDS linkage unavailable ({e}), using SEC mapping only")

    rows = []
    episodes = get_all()
    for ep in episodes:
        for ticker in ep["tickers"]:
            t = ticker.upper()
            cik = sec_map.get(t) or wrds_map.get(t)
            rows.append({
                "episode_id": ep["id"],
                "ticker": t,
                "cik": cik,
            })

    df = pd.DataFrame(rows)
    out_path = SEC_DIR / "cik_mapping.parquet"
    df.to_parquet(out_path, index=False)

    n_total = len(df)
    n_mapped = df["cik"].notna().sum()
    n_episodes = df[df["cik"].notna()]["episode_id"].nunique()
    print(f"  CIK mapping: {n_mapped}/{n_total} tickers mapped ({n_mapped/n_total:.0%})")
    print(f"  Episodes with >= 1 CIK: {n_episodes}/{df['episode_id'].nunique()}")

    # Per-episode summary
    for eid in sorted(df["episode_id"].unique()):
        sub = df[df["episode_id"] == eid]
        mapped = sub["cik"].notna().sum()
        print(f"    {eid:25s}: {mapped}/{len(sub)} tickers mapped")

    return df


# ── Phase 2: Filing Indices ───────────────────────────────────────────

def fetch_filing_index(cik, start_date, end_date):
    """Fetch filing metadata from EDGAR submissions API."""
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    time.sleep(SEC_RATE_LIMIT)
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    df = pd.DataFrame(recent)
    if df.empty or "form" not in df.columns:
        return pd.DataFrame()

    df = df[df["form"].isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])].copy()
    if df.empty:
        return df

    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df = df[(df["filingDate"] >= start_date) & (df["filingDate"] <= end_date)].copy()
    df["cik"] = str(cik)

    keep = ["cik", "accessionNumber", "filingDate", "form", "primaryDocument"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def build_filing_indices(cik_mapping):
    """Build filing indices for all episodes."""
    from src.catalog import get_all

    episodes = get_all()
    ep_dict = {e["id"]: e for e in episodes}

    for eid in sorted(cik_mapping["episode_id"].unique()):
        out_path = INDEX_DIR / f"{eid}.parquet"
        if out_path.exists():
            df = pd.read_parquet(out_path)
            print(f"  {eid:25s}: CACHED ({len(df)} filings)")
            continue

        ep = ep_dict.get(eid)
        if ep is None:
            continue

        # Window: 5 years pre to 2 years post
        start_yr = ep["period"][0] - 5
        peak_yr = ep["period"][1] or 2026
        end_yr = (ep["period"][2] or peak_yr) + 2
        start_date = f"{max(start_yr, 1993)}-01-01"
        end_date = f"{min(end_yr, 2026)}-12-31"

        ciks = cik_mapping[
            (cik_mapping["episode_id"] == eid) & cik_mapping["cik"].notna()
        ]["cik"].unique()

        if len(ciks) == 0:
            print(f"  {eid:25s}: NO CIKs")
            continue

        frames = []
        for cik in ciks:
            idx = fetch_filing_index(cik, start_date, end_date)
            if not idx.empty:
                frames.append(idx)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            result.to_parquet(out_path, index=False)
            print(f"  {eid:25s}: {len(result)} filings from {len(ciks)} CIKs")
        else:
            print(f"  {eid:25s}: no filings found ({len(ciks)} CIKs tried)")


# ── Phase 3: Download Filing Text ─────────────────────────────────────

def download_filing_text(cik, accession, primary_doc):
    """Download a single filing from EDGAR and return cleaned text."""
    accession_clean = accession.replace("-", "")
    url = (f"https://www.sec.gov/Archives/edgar/data/"
           f"{cik}/{accession_clean}/{primary_doc}")

    time.sleep(SEC_RATE_LIMIT)
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=60)
        if r.status_code != 200:
            return None
        raw = r.text
    except Exception:
        return None

    if "<html" in raw.lower() or "<body" in raw.lower():
        text = strip_html(raw)
    else:
        text = raw

    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_all_texts():
    """Download filing text for all indexed filings."""
    index_files = sorted(INDEX_DIR.glob("*.parquet"))
    total = 0
    cached = 0
    failed = 0

    for idx_path in index_files:
        bid = idx_path.stem
        df = pd.read_parquet(idx_path)

        for _, row in df.iterrows():
            cik = str(row["cik"])
            accession = row["accessionNumber"]
            doc = row.get("primaryDocument", "")
            if not doc:
                continue

            safe_acc = accession.replace("-", "")
            text_path = TEXT_DIR / f"{cik}_{safe_acc}.txt"

            if text_path.exists():
                cached += 1
                continue

            text = download_filing_text(cik, accession, doc)
            if text and len(text) > 1000:
                text_path.write_text(text, encoding="utf-8")
                total += 1
            else:
                failed += 1

            if (total + failed) % 100 == 0:
                print(f"  Downloaded: {total}, cached: {cached}, failed: {failed}")

    print(f"  FINAL: downloaded={total}, cached={cached}, failed={failed}")


# ── Phase 4: NLP Scoring ──────────────────────────────────────────────

def run_nlp_scoring():
    """Score all downloaded filings with L-M NLP pipeline."""
    from src.sec_nlp import process_all_episodes
    process_all_episodes()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 1: CIK Mapping")
    print("=" * 60)
    cik_mapping = build_cik_mapping()

    print(f"\n{'=' * 60}")
    print("Phase 2: Filing Indices")
    print("=" * 60)
    build_filing_indices(cik_mapping)

    # Summary
    n_filings = 0
    n_episodes = 0
    for f in INDEX_DIR.glob("*.parquet"):
        n = len(pd.read_parquet(f))
        if n > 0:
            n_filings += n
            n_episodes += 1
    print(f"\nTotal: {n_filings} filings across {n_episodes} episodes")

    print(f"\n{'=' * 60}")
    print("Phase 3: Download Filing Text")
    print("=" * 60)
    download_all_texts()

    print(f"\n{'=' * 60}")
    print("Phase 4: NLP Scoring")
    print("=" * 60)
    run_nlp_scoring()

    print("\nDone.")


if __name__ == "__main__":
    main()
