"""
21_embed_minilm.py — MiniLM contextual embeddings for bubble project filings.

Adapted from smallcap_graduation/scripts/06c_embed_minilm.py.

Reads raw SEC filing text from data/sec/filings_text/*.txt, extracts
three key sections (Risk Factors, MD&A, Quantitative Disclosures),
chunks with 25% overlap, encodes with all-MiniLM-L6-v2, and mean-pools
per section.

Optimizations vs v1:
  - Regex HTML stripping instead of BeautifulSoup (200x faster per filing)
  - Cross-filing mega-batching: accumulate chunks across filings,
    encode in large batches (much better GPU/CPU utilization)
  - Skip filings with no extractable sections before encoding
  - Numpy array construction instead of per-element dict building

Output:
  data/sec/embeddings_minilm.parquet
  Shape: (n_filings, 1152 + metadata)  [384d × 3 sections]

Runs on CPU. ~10-15 min for 8,500 filings.
"""
import warnings
warnings.filterwarnings("ignore")

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

SEC_DIR = DATA_DIR / "sec"
FILINGS_TEXT_DIR = SEC_DIR / "filings_text"

MODEL_NAME = "all-MiniLM-L6-v2"
MEGA_BATCH = 512  # chunks to encode at once across filings
CHUNK_WORDS = 256

# Section extraction patterns
SEC_SECTIONS = {
    "risk_factors": r"(?i)item\s+1a\.?\s*[\-\—]?\s*risk\s+factors",
    "mda": r"(?i)item\s+7\.?\s*[\-\—]?\s*management.s\s+discussion",
    "quant_disclosures": r"(?i)item\s+7a\.?\s*[\-\—]?\s*quantitative\s+and\s+qualitative",
}
SECTION_NAMES = list(SEC_SECTIONS.keys())

# Pre-compile regexes
_RE_TAGS = re.compile(r'<[^>]+>')
_RE_MULTI_SPACE = re.compile(r'[ \t]+')
_RE_MULTI_NL = re.compile(r'\n{3,}')
_RE_FWD_LOOKING = re.compile(
    r'(?i)(?:this|the)\s+(?:annual|quarterly)\s+report.*?'
    r'forward[- ]looking\s+statements.*?(?:\n\n|\Z)')
_RE_SECTIONS = {name: re.compile(pat) for name, pat in SEC_SECTIONS.items()}
_RE_NEXT_ITEM = re.compile(r'(?i)\nitem\s+\d')


def clean_filing_text(raw_text):
    """Strip HTML/SGML tags via regex (200x faster than BeautifulSoup)."""
    text = _RE_TAGS.sub(' ', raw_text)
    text = _RE_MULTI_SPACE.sub(' ', text)
    text = _RE_MULTI_NL.sub('\n\n', text)
    text = _RE_FWD_LOOKING.sub('', text, count=1)
    return text


def extract_sections(text):
    """Extract Risk Factors, MD&A, and Quant Disclosures sections."""
    sections = {}
    for section_name, pattern in _RE_SECTIONS.items():
        match = pattern.search(text)
        if match:
            start = match.start()
            next_item = _RE_NEXT_ITEM.search(text, match.end())
            end = next_item.start() if next_item else min(start + 50000, len(text))
            sections[section_name] = text[start:end]
        else:
            sections[section_name] = ""
    return sections


def chunk_text(text):
    """Split text into overlapping word-level chunks."""
    if not text:
        return []
    words = text.split()
    if len(words) < 10:
        return []
    chunks = []
    step = CHUNK_WORDS * 3 // 4  # 25% overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + CHUNK_WORDS]
        if len(chunk_words) >= 10:
            chunks.append(" ".join(chunk_words))
    return chunks


def main():
    print("=" * 60)
    print("21_embed_minilm.py — MiniLM embeddings (optimized)")
    print("=" * 60)

    # Load model
    print(f"\n  Loading {MODEL_NAME}...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    emb_dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded: {emb_dim}d embeddings")

    # Load filing index metadata
    filing_meta = {}
    for idx_path in sorted(SEC_DIR.joinpath("filings_index").glob("*.parquet")):
        episode_id = idx_path.stem
        idx = pd.read_parquet(idx_path)
        for _, row in idx.iterrows():
            cik = str(row["cik"])
            acc = row["accessionNumber"].replace("-", "")
            fname = f"{cik}_{acc}.txt"
            filing_meta[fname] = {
                "cik": cik,
                "accession": row["accessionNumber"],
                "episode_id": episode_id,
                "filing_date": row["filingDate"],
                "form": row.get("form", ""),
            }
    print(f"  Filing index: {len(filing_meta)} entries")

    filing_paths = sorted(FILINGS_TEXT_DIR.glob("*.txt"))
    n_files = len(filing_paths)
    print(f"  Text files: {n_files}")

    # ── Phase 1: Extract and chunk all filings (fast with regex) ────
    print("\n  Phase 1: Extracting sections & chunking...", flush=True)
    t0 = time.time()

    # For each filing, store: metadata + list of (section_name, chunk_text)
    filing_records = []   # metadata dicts
    filing_chunks = []    # list of lists: [(section_name, chunk_str), ...]
    n_sections_found = {s: 0 for s in SECTION_NAMES}

    for i, fp in enumerate(filing_paths):
        fname = fp.name
        meta = filing_meta.get(fname)
        if not meta:
            parts = fname.replace(".txt", "").split("_", 1)
            meta = {"cik": parts[0], "accession": parts[1] if len(parts) > 1 else "",
                    "episode_id": "", "filing_date": "", "form": ""}

        try:
            raw_text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        clean_text = clean_filing_text(raw_text)
        sections = extract_sections(clean_text)

        chunks_for_filing = []
        for sec_name in SECTION_NAMES:
            sec_chunks = chunk_text(sections.get(sec_name, ""))
            if sec_chunks:
                n_sections_found[sec_name] += 1
            for c in sec_chunks:
                chunks_for_filing.append((sec_name, c))

        filing_records.append(meta)
        filing_chunks.append(chunks_for_filing)

    t_extract = time.time() - t0
    total_chunks = sum(len(fc) for fc in filing_chunks)
    n_with_sections = sum(1 for fc in filing_chunks if fc)
    print(f"  Extracted {len(filing_records)} filings in {t_extract:.1f}s")
    print(f"  Total chunks: {total_chunks:,}")
    print(f"  Filings with sections: {n_with_sections} / {len(filing_records)} "
          f"({100 * n_with_sections / max(len(filing_records), 1):.0f}%)")
    for sec_name, count in n_sections_found.items():
        print(f"    {sec_name:20s}: {count:,}")

    # ── Phase 2: Mega-batch encoding ────────────────────────────────
    print(f"\n  Phase 2: Encoding {total_chunks:,} chunks in mega-batches "
          f"of {MEGA_BATCH}...", flush=True)
    t0 = time.time()

    # Flatten all chunks into one list with back-pointers
    all_chunk_texts = []
    chunk_filing_idx = []   # which filing this chunk belongs to
    chunk_section = []      # which section

    for fi, chunks in enumerate(filing_chunks):
        for sec_name, chunk_str in chunks:
            all_chunk_texts.append(chunk_str)
            chunk_filing_idx.append(fi)
            chunk_section.append(sec_name)

    # Encode in mega-batches
    all_embeddings = np.zeros((len(all_chunk_texts), emb_dim), dtype=np.float32)
    n_encoded = 0

    for batch_start in range(0, len(all_chunk_texts), MEGA_BATCH):
        batch_end = min(batch_start + MEGA_BATCH, len(all_chunk_texts))
        batch_texts = all_chunk_texts[batch_start:batch_end]

        batch_embs = model.encode(batch_texts, batch_size=MEGA_BATCH,
                                  show_progress_bar=False)
        all_embeddings[batch_start:batch_end] = batch_embs
        n_encoded += len(batch_texts)

        if n_encoded % (MEGA_BATCH * 10) == 0 or batch_end == len(all_chunk_texts):
            elapsed = time.time() - t0
            rate = n_encoded / elapsed
            remaining = (len(all_chunk_texts) - n_encoded) / max(rate, 1)
            print(f"    [{n_encoded:,}/{len(all_chunk_texts):,}] "
                  f"{rate:.0f} chunks/sec, "
                  f"ETA {remaining/60:.1f} min", flush=True)

    t_encode = time.time() - t0
    print(f"  Encoded {len(all_chunk_texts):,} chunks in {t_encode/60:.1f} min")

    # ── Phase 3: Mean-pool per section per filing ───────────────────
    print("\n  Phase 3: Pooling embeddings per filing...", flush=True)

    chunk_filing_idx = np.array(chunk_filing_idx)
    chunk_section = np.array(chunk_section)

    # Pre-allocate embedding arrays: (n_filings, n_sections, emb_dim)
    n_filings = len(filing_records)
    emb_array = np.zeros((n_filings, len(SECTION_NAMES), emb_dim), dtype=np.float32)

    for si, sec_name in enumerate(SECTION_NAMES):
        sec_mask = chunk_section == sec_name
        if not sec_mask.any():
            continue
        sec_filing_idx = chunk_filing_idx[sec_mask]
        sec_embeddings = all_embeddings[sec_mask]

        # Group by filing index and mean-pool
        for fi in np.unique(sec_filing_idx):
            fi_mask = sec_filing_idx == fi
            emb_array[fi, si] = sec_embeddings[fi_mask].mean(axis=0)

    # ── Phase 4: Build DataFrame ────────────────────────────────────
    print("  Building DataFrame...", flush=True)

    # Metadata columns
    meta_df = pd.DataFrame(filing_records)
    meta_df["n_chunks"] = [len(fc) for fc in filing_chunks]

    # Embedding columns: reshape (n_filings, n_sections, emb_dim) -> flat columns
    emb_flat = emb_array.reshape(n_filings, -1)  # (n_filings, n_sections * emb_dim)
    emb_col_names = []
    for sec_name in SECTION_NAMES:
        for d in range(emb_dim):
            emb_col_names.append(f"emb_{sec_name}_{d}")

    emb_df = pd.DataFrame(emb_flat, columns=emb_col_names)
    df = pd.concat([meta_df, emb_df], axis=1)

    # Save
    out_path = SEC_DIR / "embeddings_minilm.parquet"
    df.to_parquet(out_path, index=False)

    n_with_chunks = (df["n_chunks"] > 0).sum()
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Filings with sections: {n_with_chunks:,} / {len(df):,} "
          f"({100 * n_with_chunks / len(df):.0f}%)")
    print(f"  Episodes covered: {df['episode_id'].nunique()}")

    total_time = time.time() - (t0 - t_encode)
    print(f"\n  Total time: {total_time/60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
