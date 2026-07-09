"""
sec_nlp.py — NLP pipeline for SEC filing text analysis.

Uses the full Loughran-McDonald Master Dictionary (86,486 words) with
exact word matching. Computes:
  - All 6 L-M sentiment categories (negative, positive, uncertainty,
    litigious, constraining, modal weak/strong)
  - 6 readability indices (Gunning Fog, Flesch-Kincaid, Flesch Reading Ease,
    Coleman-Liau, ARI, SMOG)
  - Custom bubble dictionary (off-balance-sheet, covenant, risk, hype language)
"""
import re
import math
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR

SEC_DIR = DATA_DIR / "sec"
NLP_DIR = SEC_DIR / "nlp_scores"
TEXT_DIR = SEC_DIR / "filings_text"
INDEX_DIR = SEC_DIR / "filings_index"
NLP_DIR.mkdir(parents=True, exist_ok=True)

LM_DICT_PATH = SEC_DIR / "lm_dictionary.csv"


# ── Load Loughran-McDonald Dictionary ───────────────────────────────────

def load_lm_dictionary():
    """Load the full L-M master dictionary into word sets."""
    if not LM_DICT_PATH.exists():
        raise FileNotFoundError(
            f"L-M dictionary not found at {LM_DICT_PATH}. "
            "Download from https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
        )

    df = pd.read_csv(LM_DICT_PATH)
    df = df.dropna(subset=["Word"])
    df["Word"] = df["Word"].astype(str)

    lm = {
        "negative": set(df[df["Negative"] != 0]["Word"].str.upper()),
        "positive": set(df[df["Positive"] != 0]["Word"].str.upper()),
        "uncertainty": set(df[df["Uncertainty"] != 0]["Word"].str.upper()),
        "litigious": set(df[df["Litigious"] != 0]["Word"].str.upper()),
        "constraining": set(df[df["Constraining"] != 0]["Word"].str.upper()),
        "modal_weak": set(df[df["Modal"] == 1]["Word"].str.upper()),
        "modal_moderate": set(df[df["Modal"] == 2]["Word"].str.upper()),
        "modal_strong": set(df[df["Modal"] == 3]["Word"].str.upper()),
    }

    syllable_map = {}
    for _, row in df.iterrows():
        if pd.notna(row["Word"]) and pd.notna(row.get("Syllables", None)) and row.get("Syllables", 0) > 0:
            syllable_map[str(row["Word"]).upper()] = int(row["Syllables"])

    total = sum(len(v) for v in lm.values())
    print(f"  L-M dictionary: {len(df)} words, {total} categorized, "
          f"{len(syllable_map)} with syllables")

    return lm, syllable_map


# ── Custom Bubble Dictionary ────────────────────────────────────────────

BUBBLE_DICT = {
    "offbalance": [
        "variable interest entity", "variable interest entities", "vie ",
        "special purpose entity", "special purpose entities", "special purpose vehicle",
        "spe ", "spv ", "off-balance", "off balance sheet",
        "unconsolidated", "structured investment vehicle", "conduit",
        "securitization", "asset-backed", "collateralized",
    ],
    "covenant_stress": [
        "covenant", "waiver", "amendment to credit", "debt default",
        "acceleration clause", "refinancing", "maturity date",
        "credit facility", "revolving credit", "debt covenant",
        "loan agreement", "credit agreement", "forbearance",
        "debt-to-equity", "leverage ratio", "interest coverage",
    ],
    "risk_escalation": [
        "material weakness", "going concern", "liquidity risk",
        "impairment", "goodwill impairment", "write-down", "write-off",
        "writedown", "writeoff", "restatement", "restated",
        "significant deficiency", "adverse opinion", "qualified opinion",
        "internal control weakness", "remediation",
    ],
    "growth_narrative": [
        "unprecedented", "transformative", "paradigm", "revolutionary",
        "disruptive", "exponential growth", "game-changing", "game changing",
        "first mover", "massive opportunity", "enormous potential",
        "groundbreaking", "best-in-class", "world-class", "category-defining",
        "inflection point", "hockey stick", "explosive growth",
    ],
}


def _count_phrases(text_lower, phrases):
    return sum(text_lower.count(p.lower()) for p in phrases)


# ── Readability ───────────────────────────────────────────────────────

def _syllable_count_heuristic(word):
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def compute_readability(words, sentences, syllable_map):
    n_words = len(words)
    n_sentences = max(sentences, 1)

    total_syllables = 0
    n_complex = 0
    for w in words:
        syl = syllable_map.get(w.upper(), _syllable_count_heuristic(w))
        total_syllables += syl
        if syl >= 3:
            n_complex += 1

    aws = n_words / n_sentences
    asp = total_syllables / max(n_words, 1)
    pct_c = n_complex / max(n_words, 1)

    fog = 0.4 * (aws + 100 * pct_c)
    fk = 0.39 * aws + 11.8 * asp - 15.59
    fre = 206.835 - 1.015 * aws - 84.6 * asp
    n_chars = sum(len(w) for w in words)
    L = (n_chars / n_words) * 100
    S = (n_sentences / n_words) * 100
    cli = 0.0588 * L - 0.296 * S - 15.8
    ari = 4.71 * (n_chars / n_words) + 0.5 * aws - 21.43
    smog = 3.0 + math.sqrt(n_complex * (30 / n_sentences)) if n_sentences >= 30 else 3.0 + math.sqrt(n_complex)

    return {
        "fog_index": fog, "fk_grade": fk, "flesch_reading_ease": fre,
        "coleman_liau": cli, "ari": ari, "smog": smog,
        "avg_words_per_sentence": aws, "avg_syllables_per_word": asp,
        "pct_complex_words": pct_c, "total_syllables": total_syllables,
    }


# ── Process a Single Filing ──────────────────────────────────────────

def process_filing(text, lm_dict, syllable_map):
    if not text or len(text) < 1000:
        return None

    words = re.findall(r"[a-zA-Z]+", text)
    n_words = len(words)
    if n_words < 200:
        return None

    sentences_raw = re.split(r"[.!?]+", text)
    n_sentences = max(len([s for s in sentences_raw if len(s.strip()) > 10]), 1)

    words_upper = [w.upper() for w in words]
    n_neg = sum(1 for w in words_upper if w in lm_dict["negative"])
    n_pos = sum(1 for w in words_upper if w in lm_dict["positive"])
    n_unc = sum(1 for w in words_upper if w in lm_dict["uncertainty"])
    n_lit = sum(1 for w in words_upper if w in lm_dict["litigious"])
    n_con = sum(1 for w in words_upper if w in lm_dict["constraining"])
    n_mw = sum(1 for w in words_upper if w in lm_dict["modal_weak"])
    n_mm = sum(1 for w in words_upper if w in lm_dict["modal_moderate"])
    n_ms = sum(1 for w in words_upper if w in lm_dict["modal_strong"])

    readability = compute_readability(words, n_sentences, syllable_map)

    text_lower = text.lower()
    offbal = _count_phrases(text_lower, BUBBLE_DICT["offbalance"])
    cov = _count_phrases(text_lower, BUBBLE_DICT["covenant_stress"])
    risk = _count_phrases(text_lower, BUBBLE_DICT["risk_escalation"])
    growth = _count_phrases(text_lower, BUBBLE_DICT["growth_narrative"])

    return {
        "word_count": n_words, "sentence_count": n_sentences, "filing_size": len(text),
        "lm_negative_pct": n_neg / n_words, "lm_positive_pct": n_pos / n_words,
        "lm_uncertainty_pct": n_unc / n_words, "lm_litigious_pct": n_lit / n_words,
        "lm_constraining_pct": n_con / n_words,
        "lm_modal_weak_pct": n_mw / n_words, "lm_modal_moderate_pct": n_mm / n_words,
        "lm_modal_strong_pct": n_ms / n_words,
        "lm_negative_count": n_neg, "lm_positive_count": n_pos,
        "lm_uncertainty_count": n_unc, "lm_litigious_count": n_lit,
        "lm_constraining_count": n_con,
        "lm_net_sentiment": (n_pos - n_neg) / n_words,
        "lm_negativity_ratio": n_neg / max(n_pos, 1),
        **readability,
        "offbalance_per10k": offbal / n_words * 10000,
        "covenant_per10k": cov / n_words * 10000,
        "risk_escalation_per10k": risk / n_words * 10000,
        "growth_narrative_per10k": growth / n_words * 10000,
        "offbalance_count": offbal, "covenant_count": cov,
        "risk_escalation_count": risk, "growth_narrative_count": growth,
    }


# ── Process All Episodes ─────────────────────────────────────────────

def process_episode(episode_id, lm_dict, syllable_map):
    """Process all filings for one episode."""
    out_path = NLP_DIR / f"{episode_id}.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        print(f"  {episode_id:25s}: CACHED ({len(df)} filings)")
        return df

    idx_path = INDEX_DIR / f"{episode_id}.parquet"
    if not idx_path.exists():
        return None

    index = pd.read_parquet(idx_path)
    rows = []

    for _, filing in index.iterrows():
        cik = str(filing["cik"])
        accession = filing["accessionNumber"].replace("-", "")
        text_path = TEXT_DIR / f"{cik}_{accession}.txt"

        if not text_path.exists():
            continue

        text = text_path.read_text(encoding="utf-8", errors="ignore")
        scores = process_filing(text, lm_dict, syllable_map)
        if scores is None:
            continue

        scores["cik"] = cik
        scores["accession"] = filing["accessionNumber"]
        scores["filing_date"] = filing["filingDate"]
        scores["form"] = filing["form"]
        rows.append(scores)

    if not rows:
        print(f"  {episode_id:25s}: no processable filings")
        return None

    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df.to_parquet(out_path, index=False)

    n_ciks = df["cik"].nunique()
    print(f"  {episode_id:25s}: {len(df)} filings, {n_ciks} firms")
    return df


def process_all_episodes():
    """Process NLP for all episodes with filing indices."""
    print("Loading L-M dictionary...")
    lm_dict, syllable_map = load_lm_dictionary()

    episode_ids = sorted(f.stem for f in INDEX_DIR.glob("*.parquet"))
    print(f"Processing {len(episode_ids)} episodes...")

    for eid in episode_ids:
        process_episode(eid, lm_dict, syllable_map)


if __name__ == "__main__":
    process_all_episodes()
