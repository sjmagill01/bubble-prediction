# Volatility Signatures of Speculative Bubbles

**A cross-episode survival analysis of U.S. equity bubbles, 1996–2021.**

Samuel Magill, Erdős Institute, Quant Finance Bootcamp, Summer 2026

---

## What this project does

Can speculative bubbles be identified (and timed?) before they crash? This
repository builds a catalog of **27 U.S. bubble episodes** (dot-com, GFC banks,
homebuilders, shale, SPACs, crypto-adjacent equities, meme stocks, ...) and
**42 near-bubble controls** (sectors that rallied 50%+ but did *not* crash),
constructs **23 monthly metrics** across three information channels, and
evaluates identification and timing with survival models and classifiers under
a strictly pre-peak, leakage-controlled protocol.

**"Bubble" is defined operationally:** a >40% sector drawdown within 24 months
of a run-up peak. Some bubbles bounce back, such as crypto.

### Headline results

| Question | Answer |
|---|---|
| Can we identify which rallying sectors will crash? | **Yes, modestly.** LR with all 23 features: episode AUC **0.844** (permutation *p* = 0.033, 10,000 splits × 500 shuffles). Naïve baseline (vol ratio alone): 0.780. |
| Is equity volatility alone enough? | **No.** Vol-only features are indistinguishable from noise (*p* = 0.131). The SEC-text, credit, and interaction channels carry the significance. |
| Can we time the crash? | **No.** The apparent month-level timing signal (AUC 0.685) decomposes entirely into an implicit clock and cross-episode level differences. Within fixed pre-peak windows, every feature is at chance. |
| What predicts crashes? | Risk-disclosure language *reduces* crash risk (transparency helps); off-balance-sheet language (opacity) *increases* it; intra-sector correlation (herding) is noise; rising leverage signals expansion, not danger. |
| Caveats | The +6pp edge over the naïve baseline is **in-era only**: under expanding walk-forward evaluation the un-fitted baseline wins at every temporal cutoff. The signal is also concentrated in the final pre-peak year, which is anchored to a hindsight-known peak. |

### Two bubble regimes

Episodes split by crash mechanism, proxied by sector leverage L = D/(D+E):

- **Leverage bubbles** (banks, homebuilders, shale): debt-funded, crash through
  balance-sheet stress. The model works here: holding out *fang* drops AUC by 0.13.
- **Mania bubbles** (crypto, SPACs, meme stocks, dotcom): equity-funded, L ≈ 0,
  crash by sentiment reversal. Out-of-distribution for a capital-structure
  framework: holding out *crypto* *improves* AUC by 0.07.

The model is honestly described as a **leveraged-fragility detector**, not a
general bubble detector.

**Start here:** [`notebooks/01_main_results.ipynb`](notebooks/01_main_results.ipynb)
walks through every headline result with figures rendered inline, and
[`notebooks/02_case_studies.ipynb`](notebooks/02_case_studies.ipynb) shows the
multi-channel anatomy of individual episodes. Both run off the data included
in this repo.

---

## The 23 metrics (three channels)

| Channel | Metrics | Source |
|---|---|---|
| **Equity-structural (A–F)** | vol ratio, leverage-adjusted vol gap, intra-sector correlation, vol-of-vol, investment intensity, leverage trajectory | CRSP daily returns, Compustat fundamentals; Merton de-levering σ_A ≈ σ_E(1−L) |
| **SEC textual (G–N)** | sentiment, uncertainty, readability, off-balance-sheet language, risk escalation, growth narrative, filing length trend, negativity | 5,166 10-K/10-Q filings via EDGAR; Loughran–McDonald dictionaries + custom lexicon |
| **Credit-implied vol (O–S)** | bond-CIV (Merton-inverted TRACE spreads); **text-CIV** (novel): ridge from NLP features to spreads (out-of-fold R² = 0.27), Merton-inverted with actual leverage; extends CIV coverage from 59% to 100% of episodes (validates at r = 0.65 vs bond-CIV) | TRACE via WRDS; SEC filings |
| **Interactions (U–X)** | leverage level, fragility vol (A×L), mania vol (A×(1−L)), fragility instability (D×L) | derived |

---

## Repository layout

```
├── config.py                  # central config: paths, feature sets, constants
├── requirements.txt
├── notebooks/                 # executed walkthroughs (figures render on GitHub)
│   ├── 01_main_results.ipynb  #   AUC, permutation test, Cox betas, robustness
│   └── 02_case_studies.ipynb  #   multi-channel case studies + lead-lag
├── src/                       # shared library code
│   ├── catalog.py             #   episode catalog: 27 bubbles, 42 near-bubbles, 3 targets
│   ├── vol_measures.py        #   realized vol, correlation, Merton de-levering
│   ├── merton.py              #   Merton model spread <-> asset vol inversion
│   ├── sec_nlp.py             #   Loughran-McDonald NLP feature extraction
│   ├── text_civ.py            #   text-CIV ridge pipeline
│   └── wrds_utils.py          #   WRDS connection helper
├── scripts/                   # numbered pipeline (run in order)
│   ├── 00_build_firm_lists.py         # episode -> ticker/CIK/gvkey mapping
│   ├── 01_fetch_sec.py (+01b, 01c)    # EDGAR filing download        [no WRDS needed]
│   ├── 02_fetch_equity.py             # CRSP returns + Compustat     [WRDS]
│   ├── 03_fetch_bond_civ.py           # TRACE spreads -> bond-CIV    [WRDS]
│   ├── 04_compute_measures.py         # daily -> monthly vol measures
│   ├── 05_compute_metrics.py          # the 23 metrics per episode
│   ├── 06_align_panel.py              # peak-aligned train/test panels
│   ├── 07_run_model.py                # Cox PH, RSF, LR, RF; holdout CV
│   ├── 07b_timing_test.py             # timing decomposition and null result
│   ├── 07c_run_parallel.py            # 10,000-split parallel holdout
│   ├── 08_score_targets.py            # score ongoing targets (AI, quantum, nuclear)
│   ├── 09_cvar_analysis.py            # CVaR tail analysis + per-episode impact
│   ├── 10_alternative_models.py       # Bayesian LR, HMM, CUSUM
│   ├── 11_permutation_test.py (+11b)  # label-shuffle significance test
│   ├── 12_robustness.py               # walk-forward, per-fold ridge, Compustat lag
│   ├── 13_threshold_sensitivity.py    # 40% drawdown threshold sensitivity
│   ├── 14_paper_figures.py            # core result figures
│   └── 15_multichannel_figures.py     # case-study and lead-lag figures
├── data/
│   ├── firms/                 # episode ticker/identifier lists
│   ├── measures/              # monthly vol/leverage measures per episode
│   ├── metrics/               # the 23 metrics per episode (sector-month)
│   ├── civ/                   # sector-level bond-CIV and text-CIV series
│   ├── panels/                # final aligned train/test/target panels
│   ├── multichannel/          # sector-level inputs for case-study figures
│   └── results/               # all model outputs (see below)
└── figures/                   # the paper's figures (PNG)
```

---

## Data policy (what is and isn't in this repo)

**Included (~30 MB):** all *derived, sector-level* data: the 23 metrics,
aligned panels, sector-median CIV series, and every model output. Everything
needed to reproduce the modeling results (steps 06–15) **without any data
subscriptions**.

**Not included:** raw inputs that are WRDS-licensed (CRSP daily returns,
Compustat fundamentals, TRACE bond records, firm-level spread panels) or bulk
(2 GB of raw EDGAR filing text). These are excluded via `.gitignore` and are
rebuilt by scripts 00–03, which require a [WRDS](https://wrds-www.wharton.upenn.edu/)
account (set the `WRDS_USERNAME` environment variable). EDGAR filings are
public and fetched directly from the SEC (set your own contact email in
`config.py:SEC_USER_AGENT` per SEC fair-access rules).

Consequently:
- **Full reproduction from raw data**: run scripts 00–15 in order (needs WRDS).
- **Reproduction of all results in the paper**: run scripts 07–15 (or any
  subset) against the included panels; no credentials needed.
- One exception: the per-fold ridge check in `12_robustness.py` needs the
  firm-level NLP–bond overlap sample (licensed); it skips gracefully if absent.

---

## Reproducing the results

```bash
pip install -r requirements.txt

# Modeling from the included panels (no WRDS needed)
python scripts/07_run_model.py            # holdout CV: Cox/RSF/LR/RF
python scripts/07b_timing_test.py         # timing null result
python scripts/07c_run_parallel.py        # 10,000-split evaluation
python scripts/11_permutation_test.py --splits 10000 --perms 500
python scripts/12_robustness.py           # walk-forward + lag robustness
python scripts/13_threshold_sensitivity.py
python scripts/14_paper_figures.py        # regenerate figures/
python scripts/15_multichannel_figures.py
```

Key outputs land in `data/results/`:

| File | Contents |
|---|---|
| `holdout_80_20_survival.parquet`, `holdout_80_20_classifiers.parquet` | per-split AUCs and Cox betas across stratified holdout splits |
| `permutation_test_all.csv` / `_vol_only.csv` | observed AUC vs 5M-value null distribution; *p* = 0.033 / 0.131 |
| `walk_forward.parquet`, `robustness/expanding_walk_forward.csv` | temporal generalization (the adverse finding) |
| `episode_impact.parquet` | leave-one-episode-out AUC impact (two-regime evidence) |
| `cox_coefficients_10k.parquet` | coefficient stability across splits |
| `target_*.parquet` | hazard scores for ongoing targets (AI, quantum, nuclear) |

---

## Methodological safeguards

Much of this project's contribution is negative-space: quantifying how easily
this kind of analysis fools itself.

1. **Pre-peak data only.** An earlier design that included post-peak data
   inflated AUC to 0.875 and produced theory-friendly coefficients (rising
   leverage, spiking correlation) that were pure artifacts of post-crash
   liquidation. Removing contamination flipped both signs.
2. **Train-only statistics.** Standardization and imputation are fit on the
   training fold only.
3. **Leverage-stratified holdout.** Unstratified splits swing AUC 0.37–0.97
   depending on which regime lands in the test set.
4. **Permutation testing.** 10,000 splits × 500 label shuffles. Vol-only
   features fail this test; only the multi-channel model passes.
5. **Honest negatives, disclosed.** No out-of-era edge under walk-forward;
   signal concentrated in the late pre-peak window; Bonferroni-adjusted
   *p* ≈ 0.066 across the two feature sets tested.
