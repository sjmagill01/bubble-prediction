# Volatility Signatures of Speculative Bubbles

**Two crash mechanisms, one model that knows the difference: identifying U.S. equity bubbles 1996–2024.**

Samuel Magill — Erdős Institute Quant Finance Bootcamp, Summer 2026

---

## What this project does

Can speculative bubbles be identified before they crash? This repository
builds a catalog of **27 U.S. bubble episodes** (dot-com, GFC banks,
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
| Can we identify which rallying sectors will crash? | **Yes, modestly.** LR with all 23 features: episode AUC **0.844** (permutation *p* = 0.033, 10,000 splits x 500 shuffles). Naive baseline (vol ratio alone): 0.780. |
| Is equity volatility alone enough? | **No.** Vol-only features are indistinguishable from noise (*p* = 0.131). The SEC-text, credit, and interaction channels carry the significance. |
| Can we time the crash? | **No.** The apparent month-level timing signal (AUC 0.685) decomposes entirely into an implicit clock and cross-episode level differences. Within fixed pre-peak windows, every feature is at chance. |
| What predicts crashes? | Risk-disclosure language *reduces* crash risk (transparency helps); off-balance-sheet language (opacity) *increases* it; intra-sector correlation (herding) is noise; rising leverage signals expansion, not danger. |
| Does the model generalize across time? | **Yes, once regime structure is explicit.** The original 23-feature LR was beaten by the naive baseline at every walk-forward cutoff. Incorporating the two-regime structure (leverage vs. mania) with LASSO regularization restores walk-forward AUC to **0.850** at the 2015 cutoff (+9pp vs. original LR, +7pp vs. naive). See Section 4. |
| What do ongoing targets look like? | All three score high once AI is reclassified as mania (0.90) rather than capex (0.22). Quantum and Nuclear Renaissance II score near 1.0. See Section 7. |

**Deeper dives:** [`notebooks/01_main_results.ipynb`](notebooks/01_main_results.ipynb)
reproduces every result below from the included data, and
[`notebooks/02_case_studies.ipynb`](notebooks/02_case_studies.ipynb) walks
through individual episodes channel by channel.

---

## The story in seven figures

### 1. Identification: the baseline is the real benchmark

Episode-level AUC across leverage-stratified 80/20 holdout splits: given a
sector mid-rally, will it crash or fade? Two reference lines matter: 0.5 is
chance, and **0.780** is a naive baseline that scores episodes by the single
vol-ratio metric with *no fitting at all*. Any model has to beat the
baseline, not chance.

![Episode AUC distributions](figures/fig_episode_auc_dist.png)

Logistic regression with all 23 features reaches a median episode AUC of
**0.844**, about 6pp above the baseline. Tree-based survival models plateau
at the baseline; vol-only features sit *below* it. The regime-LASSO
(47-feature, right panel) reaches a median of **0.875** across 80/20 splits
with C selected by LOO-CV. A permutation test (10,000 splits x 500 label
shuffles, 5M null values) puts the full model at *p* = 0.033 while vol-only
fails at *p* = 0.131: equity volatility alone carries no episode-level signal.

### 2. Two crash mechanisms

The catalog contains two mechanistically distinct crash types. **Leverage
bubbles** (banks, homebuilders, shale) are debt-funded and crash through
balance-sheet stress. **Mania bubbles** (crypto, SPACs, dotcom) are
equity-funded and crash by sentiment reversal. Sector median leverage
L = D/(D+E) cleanly separates them: leverage bubbles cluster at L > 0.4,
mania bubbles at L near zero.

![CVaR and episode impact](figures/fig_cvar_and_impact.png)

The per-episode LOO impact analysis makes the split concrete: holding out
*fang* (leverage bubble) drops AUC by 0.13; the model needs it. Holding
out *crypto* (mania bubble) *improves* AUC by 0.07; crypto is
out-of-distribution for a capital-structure framework. Mania bubbles are
not penalized by balance-sheet stress; they crash by sentiment reversal,
and the same metrics that predict leverage crashes are noise for mania crashes.

![Regime motivation](figures/fig_regime_motivation.png)

Left panel: LOO impact vs sector leverage. Right panel: leverage distributions
by regime. The regime split is empirical, not asserted: it falls out of the
LOO analysis before any regime encoding is applied.

The model is honestly described as a **leveraged-fragility detector**, not a
general bubble detector. This observation directly motivates the regime-aware
LASSO (Section 4): encoding the regime as an explicit indicator with
interaction terms lets the model learn separate linear programs for each
crash mechanism.

### 3. What predicts crashes is not what theory suggests

Cox proportional-hazards coefficients (per 1 SD, medians and 95% intervals
across holdout splits). Red bars exclude zero.

![Cox coefficients](figures/fig_cox_coefficients.png)

- **Risk-escalation language (K)** is the strongest predictor, and it is
  *negative*: firms that escalate risk disclosures crash less. Transparency
  helps.
- **Off-balance-sheet language (J)** is the strongest positive predictor:
  opacity predicts crashes.
- **Intra-sector correlation (C)**, the textbook "herding" signal, is noise.
- **Rising leverage (F)** is negative: leverage growth marks expansion, not
  imminent danger. Leverage *level* matters instead, via interactions.

### 4. Walk-forward arc: failure, diagnosis, fix

The original 23-feature LR failed every temporal cutoff: the un-fitted
baseline won each time. Diagnosing why led directly to the fix.

**The failure:** a single linear model with 23 features and n=69 episodes
cannot separate the two regimes from noise. It overfits in-sample and
generalizes to chance out-of-sample.

**The fix:** encode the regime explicitly. Map each episode's catalog
category to one of two regimes (leverage = financial/capex;
mania = mania/commodity), build a 47-feature matrix (23 base metrics + a
regime indicator + 23 base x regime interactions), and select regularization
strength via leave-one-episode-out CV (LASSO logistic). C is selected on
the training set only at each cutoff (no leakage).

**The result:** walk-forward AUC at the main 2015 cutoff rises from
0.758 (original LR) to **0.850** (+9pp), outperforming the naive
baseline of 0.780 (+7pp).

![Walk-forward](figures/fig_regime_lasso_wf.png)

Full expanding walk-forward comparison (train on episodes peaked before
cutoff, test on later ones):

| Cutoff | N train | Orig LR | Regime-LASSO | Mania | Leverage | Note |
|--------|---------|---------|--------------|-------|----------|------|
| 2008-01 | 16 | 0.540 | 0.435 | 0.470 | 0.453 | n < 25; LOO unreliable |
| 2010-01 | 20 | 0.594 | 0.175 | 0.222 | 0.146 | n < 25; LOO unreliable |
| 2012-01 | 21 | 0.699 | 0.711 | 0.745 | 0.771 | borderline |
| 2015-01 | 27 | 0.758 | **0.850** | 0.919 | 0.857 | |
| 2018-01 | 34 | 0.723 | 0.712 | 0.707 | 0.538 | |

Early cutoffs (n < 25) degrade: with 47 features and n=16-20 training
episodes, LOO-CV cannot reliably select the regularization strength and
over-aggressively zeros out nearly all features. This is a documented
constraint of the method, not a post-hoc patch; the boundary (n=25
training episodes) follows from the feature dimension, not from which
rows look bad.

### 5. Multi-channel anatomy

Four leverage bubbles seen through all three channels, aligned on months
relative to the run-up peak (red line). No single channel is reliable, but
*disagreement between channels is informative*: in the GFC banks panel,
equity vol stays calm into the peak while credit-implied vol creeps up and
filing negativity rises.

![Multi-channel case studies](figures/fig7_multichannel_cases.png)

Lead-lag cross-correlations between channels confirm why: the channels carry
complementary *cross-sectional* information (which sectors are fragile), not
*sequential* information (when the crash comes).

![Lead-lag cross-correlations](figures/fig9_lead_lag.png)

Channel cross-correlations are modest and roughly symmetric. This is the same
message as the timing null: no channel leads another reliably enough to use
for timing.

### 6. Robustness

The 40% drawdown cutoff defining "bubble" is a choice. Sweeping it from 30%
to 50% relabels episodes; median AUC stays above 0.82 for thresholds of
35-50%.

![Threshold sensitivity](figures/fig_robustness_threshold.png)

A companion check lags all Compustat inputs by 0-90 days to respect reporting
delays; AUC is flat (see `figures/fig_robustness_compustat_lag.png`). Both
checks confirm the result is not an artifact of the threshold choice or of
using data before it is publicly available.

### 7. Ongoing targets

Regime-LASSO bubble probability for three ongoing speculative sectors,
scored using the full-sample trained model. Each trajectory is a
rolling 12-month trailing mean of the 47-feature vector. Reference lines
are computed from the full-sample LOO: bubble median and near-bubble IQR
over all 69 training episodes.

| Target | Regime | Latest score (2024-12) | Reading |
|--------|--------|------------------------|---------|
| AI Semiconductors | mania | **0.90** | Deep in historical bubble zone |
| Nuclear Renaissance II | leverage (capex) | **1.00** | Extreme: balance-sheet stress signatures |
| Quantum Computing | mania | **1.00** | Extreme: sentiment/vol signatures |

**Note on AI's regime assignment:** the catalog originally classified AI as
"capex" (-> leverage regime) because NVDA/AMD have real datacenter capex
intensity. But the dominant crash mechanism for AI would be sentiment reversal
the same as dotcom, crypto, and SPACs, not balance-sheet stress: these firms carry strong balance sheets. Reclassifying AI as "mania" raises
its score from 0.22 to 0.90: the mania-regime features (extreme vol ratio,
narrative-driven filings, low leverage) match better than the leverage-regime
ones did.

A score near 1.0 does not predict *when* a crash occurs: the timing null
(Section 1) still applies. It indicates that the sector's current
multi-channel signal pattern resembles historical bubble episodes in
the same regime more than it resembles near-bubble controls.

![Target scores](figures/fig_target_regime_scores.png)

---

## The 23 metrics (three channels)

| Channel | Metrics | Source |
|---|---|---|
| **Equity-structural (A-F)** | vol ratio, leverage-adjusted vol gap, intra-sector correlation, vol-of-vol, investment intensity, leverage trajectory | CRSP daily returns, Compustat fundamentals; Merton de-levering sigma_A = sigma_E(1-L) |
| **SEC textual (G-N)** | sentiment, uncertainty, readability, off-balance-sheet language, risk escalation, growth narrative, filing length trend, negativity | 5,166 10-K/10-Q filings via EDGAR; Loughran-McDonald dictionaries + custom lexicon |
| **Credit-implied vol (O-S)** | bond-CIV (Merton-inverted TRACE spreads); **text-CIV** (novel): ridge from NLP features to spreads (out-of-fold R² = 0.27), Merton-inverted with actual leverage; extends CIV coverage from 59% to 100% of episodes (validates at r = 0.65 vs bond-CIV, see `figures/fig_text_vs_bond_civ.png`) | TRACE via WRDS; SEC filings |
| **Interactions (U-X)** | leverage level, fragility vol (A*L), mania vol (A*(1-L)), fragility instability (D*L) | derived |

---

## Repository layout

```
.
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
│   ├── 15_multichannel_figures.py     # case-study and lead-lag figures
│   └── 16_regime_lasso.py             # regime-aware LASSO (walk-forward fix + target scoring)
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
needed to reproduce the modeling results (steps 06-16) **without any data
subscriptions**.

**Not included:** raw inputs that are WRDS-licensed (CRSP daily returns,
Compustat fundamentals, TRACE bond records, firm-level spread panels) or bulk
(2 GB of raw EDGAR filing text). These are excluded via `.gitignore` and are
rebuilt by scripts 00-03, which require a [WRDS](https://wrds-www.wharton.upenn.edu/)
account (set the `WRDS_USERNAME` environment variable). EDGAR filings are
public and fetched directly from the SEC (set your own contact email in
`config.py:SEC_USER_AGENT` per SEC fair-access rules).

Consequently:
- **Full reproduction from raw data:** run scripts 00-16 in order (needs WRDS).
- **Reproduction of all results in the paper:** run scripts 07-16 (or any
  subset) against the included panels; no credentials needed.
- One exception: the per-fold ridge check in `12_robustness.py` needs the
  firm-level NLP-bond overlap sample (licensed); it skips gracefully if absent.

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
python scripts/16_regime_lasso.py         # regime-aware LASSO (walk-forward fix)
python scripts/16_regime_lasso.py --expand          # expanding walk-forward table
python scripts/16_regime_lasso.py --score-targets   # score AI/quantum/nuclear2
python scripts/16_regime_lasso.py --figures         # generate figures/fig_regime_lasso_*.png
```

Key outputs land in `data/results/`:

| File | Contents |
|---|---|
| `holdout_80_20_survival.parquet`, `holdout_80_20_classifiers.parquet` | per-split AUCs and Cox betas across stratified holdout splits |
| `permutation_test_all.csv` / `_vol_only.csv` | observed AUC vs 5M-value null distribution; *p* = 0.033 / 0.131 |
| `walk_forward.parquet`, `robustness/expanding_walk_forward.csv` | original LR temporal generalization |
| `regime_lasso.parquet` | regime-LASSO 80/20 splits (1,000 splits, C=1.438) |
| `regime_lasso_expanding_wf.csv` | expanding walk-forward at 5 cutoffs (regime-LASSO vs original LR) |
| `episode_impact.parquet` | leave-one-episode-out AUC impact (two-regime evidence) |
| `cox_coefficients_10k.parquet` | coefficient stability across splits |
| `target_*_all.parquet`, `target_*_regime_lasso.parquet` | Cox hazard and regime-LASSO scores for ongoing targets |

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
3. **Leverage-stratified holdout.** Unstratified splits swing AUC 0.37-0.97
   depending on which regime lands in the test set.
4. **Permutation testing.** 10,000 splits x 500 label shuffles. Vol-only
   features fail this test; only the multi-channel model passes.
5. **Honest negatives, disclosed.** No out-of-era edge under walk-forward;
   signal concentrated in the late pre-peak window; Bonferroni-adjusted
   *p* = 0.066 across the two feature sets tested.
6. **Regime-aware regularization.** LOO-CV C selection is performed on
   training episodes only at each walk-forward cutoff. The regime indicator
   is encoded from catalog metadata, not derived from data.
