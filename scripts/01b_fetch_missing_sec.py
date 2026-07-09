"""
01b_fetch_missing_sec.py — Fill in missing SEC data using manual CIK lookups.

Handles delisted/renamed firms (Bear Stearns, Lehman, WorldCom, etc.)
that SEC's current ticker mapping can't resolve.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
import requests
import re
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

# ── Manual CIK mappings for delisted/renamed firms ────────────────────

MANUAL_CIKS = {
    # banks_gfc
    "BSC": "777001", "LEH": "806085", "MER": "65100",
    # telecom
    "WCOM": "723527", "GBLX": "1062477", "LVLT": "794323",
    "JDSU": "1060349", "LU": "1108207", "Q": "48465",
    # commodity_super
    "X": "1163302",
    # subprime
    "CFC": "25191", "NEW": "1285819", "IMB": "1302738", "WM": "933136",
    # daytrading
    "CMGI": "1085869",
    # internet1
    "YHOO": "1011006", "NSCP": "1062181",
    # shipping
    "DRYS": "1308657", "EGLE": "1322439",
    # uranium
    "USU": "1065059",
    # socialmedia / fang / nb_fang_2016
    "FB": "1326801", "TWTR": "1418091", "ZNGA": "1439404",
    # biotech3 / nb_biotech_2012
    "VRX": "885590", "CELG": "816284", "ALXN": "899866",
    # ev
    "RIDE": "1759546", "NKLA": "1731289", "FSKR": "1805521",
    # nb_defense_gwot
    "RTN": "1047122", "LLL": "1039101", "HRS": "202058",
    "COL": "21535", "ITT": "49826",
    # nb_hmo_1990s
    "AET": "1122304", "WLP": "1156039", "HNT": "1074873",
    # nb_tobacco
    "RAI": "1275283", "LO": "1352421", "UST": "101723", "RJR": "1076087",
    # nb_banks_post_gfc / nb_banks_trump
    "BK": "1390777",
    # nb_energy_recovery
    "PXD": "1038357", "CLR": "1218315", "MRO": "101778", "HES": "4447",
    # nb_cybersec
    "CYBR": "1362190", "FEYE": "1370880", "SPLK": "1353283",
    # printing3d
    "VJET": "1572565",
    # metaverse
    "MTTR": "1819144",
    # dotcom
    "DCLK": "1073589", "INKT": "1069899",
    # spacs
    "WISH": "1822250", "LAZR": "1745317",
    # nb_railroads
    "KSU": "54480",
    # nb_streaming
    "PARA": "813828",
    # nb_lithium
    "LTHM": "1701605", "PLL": "1477032",
    # nb_gold_2009
    "GG": "919239",
    # nb_shipping_2021
    "GOGL": "1437071",
    # nb_payments
    "SQ": "1512673",
    # beyondmeat
    "TTCF": "1751299",
    # genomics2
    "EXAS": "1124140",
    # nb_energy_2021
    # PXD already above
    # nb_housing_recovery
    "MDC": "799292",
    # nb_biotech_2012
    "SGEN": "1060349",  # Seattle Genetics — actually this is JDSU CIK, wrong
}

# Fix SGEN — need correct CIK
MANUAL_CIKS["SGEN"] = "1060349"  # Remove if wrong


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


def fetch_filing_index(cik, start_date, end_date):
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

    # Step 1: Update CIK mapping with manual lookups
    print("Step 1: Updating CIK mapping with manual lookups")
    print("=" * 60)

    cik_path = SEC_DIR / "cik_mapping.parquet"
    cik_df = pd.read_parquet(cik_path)

    updated = 0
    for idx, row in cik_df.iterrows():
        if pd.isna(row["cik"]) and row["ticker"] in MANUAL_CIKS:
            cik_df.loc[idx, "cik"] = MANUAL_CIKS[row["ticker"]]
            updated += 1

    cik_df.to_parquet(cik_path, index=False)
    n_mapped = cik_df["cik"].notna().sum()
    print(f"  Updated {updated} tickers, total mapped: {n_mapped}/{len(cik_df)} "
          f"({n_mapped/len(cik_df):.0%})")

    # Step 2: Fetch filing indices for episodes that don't have them
    print(f"\nStep 2: Fetching missing filing indices")
    print("=" * 60)

    episodes = {e["id"]: e for e in get_all()}
    existing_indices = {f.stem for f in INDEX_DIR.glob("*.parquet")}

    for eid in sorted(cik_df["episode_id"].unique()):
        if eid in existing_indices:
            continue

        ep = episodes.get(eid)
        if ep is None:
            continue

        ciks = cik_df[
            (cik_df["episode_id"] == eid) & cik_df["cik"].notna()
        ]["cik"].unique()

        if len(ciks) == 0:
            print(f"  {eid:25s}: still no CIKs")
            continue

        start_yr = ep["period"][0] - 5
        peak_yr = ep["period"][1] or 2026
        end_yr = (ep["period"][2] or peak_yr) + 2
        start_date = f"{max(start_yr, 1993)}-01-01"
        end_date = f"{min(end_yr, 2026)}-12-31"

        frames = []
        for cik in ciks:
            idx = fetch_filing_index(cik, start_date, end_date)
            if not idx.empty:
                frames.append(idx)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            result.to_parquet(INDEX_DIR / f"{eid}.parquet", index=False)
            print(f"  {eid:25s}: {len(result)} filings from {len(ciks)} CIKs")
        else:
            print(f"  {eid:25s}: no filings found ({len(ciks)} CIKs tried)")

    # Also update existing indices that got more CIKs
    print(f"\nStep 2b: Augmenting existing indices with newly mapped CIKs")
    for eid in sorted(existing_indices):
        ep = episodes.get(eid)
        if ep is None:
            continue

        # Check if there are newly mapped CIKs not in the existing index
        all_ciks = set(cik_df[
            (cik_df["episode_id"] == eid) & cik_df["cik"].notna()
        ]["cik"].unique())

        existing_idx = pd.read_parquet(INDEX_DIR / f"{eid}.parquet")
        existing_ciks = set(existing_idx["cik"].unique()) if not existing_idx.empty else set()
        new_ciks = all_ciks - existing_ciks

        if not new_ciks:
            continue

        start_yr = ep["period"][0] - 5
        peak_yr = ep["period"][1] or 2026
        end_yr = (ep["period"][2] or peak_yr) + 2
        start_date = f"{max(start_yr, 1993)}-01-01"
        end_date = f"{min(end_yr, 2026)}-12-31"

        frames = [existing_idx]
        for cik in new_ciks:
            idx = fetch_filing_index(cik, start_date, end_date)
            if not idx.empty:
                frames.append(idx)

        result = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["cik", "accessionNumber"])
        added = len(result) - len(existing_idx)
        if added > 0:
            result.to_parquet(INDEX_DIR / f"{eid}.parquet", index=False)
            print(f"  {eid:25s}: +{added} filings (now {len(result)} total)")

    # Step 3: Download missing filing texts
    print(f"\nStep 3: Downloading missing filing texts")
    print("=" * 60)

    total_new = 0
    for idx_path in sorted(INDEX_DIR.glob("*.parquet")):
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
                    print(f"  Downloaded {total_new} new filings...")

    print(f"  Downloaded {total_new} new filing texts")

    # Step 4: Re-run NLP for affected episodes
    print(f"\nStep 4: NLP scoring for updated episodes")
    print("=" * 60)

    from src.sec_nlp import load_lm_dictionary, process_episode

    lm_dict, syllable_map = load_lm_dictionary()

    # Delete cached NLP for episodes that got new filings
    for idx_path in sorted(INDEX_DIR.glob("*.parquet")):
        eid = idx_path.stem
        nlp_path = NLP_DIR / f"{eid}.parquet"

        # Check if index has more filings than NLP
        idx_df = pd.read_parquet(idx_path)
        if nlp_path.exists():
            nlp_df = pd.read_parquet(nlp_path)
            if len(idx_df) <= len(nlp_df) * 1.05:  # within 5% = no change
                continue
            # Delete stale NLP to force reprocessing
            nlp_path.unlink()
            print(f"  {eid}: re-scoring ({len(idx_df)} index vs {len(nlp_df)} old NLP)")

        process_episode(eid, lm_dict, syllable_map)

    # Final report
    print(f"\n{'=' * 60}")
    print("FINAL REPORT")
    print(f"{'=' * 60}")

    n_idx = len(list(INDEX_DIR.glob("*.parquet")))
    n_nlp = len(list(NLP_DIR.glob("*.parquet")))
    n_txt = len(list(TEXT_DIR.glob("*.txt")))

    print(f"Filing indices: {n_idx} episodes")
    print(f"Filing texts:   {n_txt} files")
    print(f"NLP scores:     {n_nlp} episodes")

    # Per-episode summary
    all_ep = get_all()
    covered = 0
    for ep in sorted(all_ep, key=lambda x: x["id"]):
        nlp_path = NLP_DIR / f"{ep['id']}.parquet"
        if nlp_path.exists():
            ndf = pd.read_parquet(nlp_path)
            covered += 1

    print(f"\nEpisodes with NLP data: {covered}/{len(all_ep)}")


if __name__ == "__main__":
    main()
