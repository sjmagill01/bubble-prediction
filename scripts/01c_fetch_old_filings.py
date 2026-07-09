"""
01c_fetch_old_filings.py — Fetch older SEC filings from EDGAR batch files.

The EDGAR submissions API only returns the most recent 1,000 filings in the
"recent" endpoint. Older filings are in additional batch files listed in the
"files" array. This script fetches those for episodes with pre-2017 windows.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
import re
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

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip = False
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"): self.skip = True
    def handle_endtag(self, tag):
        if tag in ("script", "style"): self.skip = False
    def handle_data(self, data):
        if not self.skip: self.result.append(data)
    def get_text(self):
        return " ".join(self.result)


def strip_html(html_text):
    stripper = HTMLStripper()
    try:
        stripper.feed(html_text)
        return stripper.get_text()
    except Exception:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_text)).strip()


def fetch_full_filing_index(cik, start_date, end_date):
    """Fetch ALL filings for a CIK, including older batch files."""
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

    # Collect from "recent" + batch files
    all_frames = []

    recent = data.get("filings", {}).get("recent", {})
    if recent and "form" in recent:
        df = pd.DataFrame(recent)
        all_frames.append(df)

    # Fetch older batch files
    files = data.get("filings", {}).get("files", [])
    for batch in files:
        batch_name = batch.get("name", "")
        if not batch_name:
            continue
        batch_url = f"https://data.sec.gov/submissions/{batch_name}"
        time.sleep(SEC_RATE_LIMIT)
        try:
            br = requests.get(batch_url, headers=SEC_HEADERS, timeout=30)
            if br.status_code == 200:
                batch_data = br.json()
                if batch_data and "form" in batch_data:
                    bdf = pd.DataFrame(batch_data)
                    all_frames.append(bdf)
        except Exception:
            pass

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)

    # Filter to 10-K/10-Q
    df = df[df["form"].isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])].copy()
    if df.empty:
        return df

    df.loc[:, "filingDate"] = pd.to_datetime(df["filingDate"])
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df = df[(df["filingDate"] >= start_ts) & (df["filingDate"] <= end_ts)].copy()
    if df.empty:
        return df

    df.loc[:, "cik"] = str(cik)

    keep = ["cik", "accessionNumber", "filingDate", "form", "primaryDocument"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def download_filing_text(cik, accession, primary_doc):
    accession_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{primary_doc}"
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
    return re.sub(r"\s+", " ", text).strip()


def main():
    from src.catalog import get_all

    # Episodes that need older filings (EDGAR "recent" only has last 1000)
    NEEDS_OLD = [
        "commodity_super", "daytrading", "internet1", "solar1",
        "nb_hmo_1990s", "nb_tobacco",
        "banks_gfc", "telecom", "subprime", "homebuilders",
        "shipping", "monolines", "nb_defense_gwot", "uranium",
        # New mania near-bubbles with pre-2010 filing windows
        "nb_restaurants_1990s", "nb_pharma_generics", "nb_consumer_staples_1990s",
        "nb_medical_devices", "nb_online_retail_2005", "nb_discount_retail_2007",
        "nb_ecommerce_2010",
    ]

    episodes = {e["id"]: e for e in get_all()}
    cik_df = pd.read_parquet(SEC_DIR / "cik_mapping.parquet")

    print("Fetching full filing history (including batch files)...")
    print("=" * 60)

    for eid in NEEDS_OLD:
        ep = episodes.get(eid)
        if ep is None:
            continue

        ciks = cik_df[
            (cik_df["episode_id"] == eid) & cik_df["cik"].notna()
        ]["cik"].unique()

        if len(ciks) == 0:
            print(f"  {eid:25s}: no CIKs")
            continue

        start_yr = ep["period"][0] - 5
        peak_yr = ep["period"][1] or 2026
        end_yr = (ep["period"][2] or peak_yr) + 2
        start_date = f"{max(start_yr, 1993)}-01-01"
        end_date = f"{min(end_yr, 2026)}-12-31"

        frames = []
        for cik in ciks:
            idx = fetch_full_filing_index(cik, start_date, end_date)
            if not idx.empty:
                frames.append(idx)

        if frames:
            result = pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=["cik", "accessionNumber"])

            # Merge with existing index if any
            existing_path = INDEX_DIR / f"{eid}.parquet"
            if existing_path.exists():
                old = pd.read_parquet(existing_path)
                result = pd.concat([old, result], ignore_index=True).drop_duplicates(
                    subset=["cik", "accessionNumber"])

            result.to_parquet(INDEX_DIR / f"{eid}.parquet", index=False)
            n_ciks = result["cik"].nunique()
            print(f"  {eid:25s}: {len(result)} filings from {n_ciks} CIKs")
        else:
            print(f"  {eid:25s}: no filings found")

    # Download texts
    print(f"\nDownloading missing filing texts...")
    total_new = 0
    for eid in NEEDS_OLD:
        idx_path = INDEX_DIR / f"{eid}.parquet"
        if not idx_path.exists():
            continue
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
                continue
            text = download_filing_text(cik, accession, doc)
            if text and len(text) > 1000:
                text_path.write_text(text, encoding="utf-8")
                total_new += 1
                if total_new % 50 == 0:
                    print(f"  Downloaded {total_new}...")

    print(f"  Downloaded {total_new} new filing texts")

    # NLP scoring
    print(f"\nNLP scoring...")
    from src.sec_nlp import load_lm_dictionary, process_episode
    lm_dict, syllable_map = load_lm_dictionary()

    for eid in NEEDS_OLD:
        nlp_path = NLP_DIR / f"{eid}.parquet"
        if nlp_path.exists():
            nlp_path.unlink()  # Force re-score with new data
        process_episode(eid, lm_dict, syllable_map)

    # Final report
    print(f"\n{'=' * 60}")
    print("UPDATED REPORT")
    n_nlp = len(list(NLP_DIR.glob("*.parquet")))
    n_txt = len(list(TEXT_DIR.glob("*.txt")))
    print(f"NLP scores: {n_nlp} episodes, {n_txt} text files")

    for eid in NEEDS_OLD:
        nlp_path = NLP_DIR / f"{eid}.parquet"
        if nlp_path.exists():
            ndf = pd.read_parquet(nlp_path)
            print(f"  {eid:25s}: {len(ndf)} NLP filings, {ndf['cik'].nunique()} firms")
        else:
            print(f"  {eid:25s}: NO DATA")


if __name__ == "__main__":
    main()
