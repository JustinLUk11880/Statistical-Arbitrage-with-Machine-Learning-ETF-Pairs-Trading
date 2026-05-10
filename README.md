# Statistical Arbitrage with Machine Learning — ETF Pairs Trading

An end-to-end research pipeline for systematic pairs trading on ETFs.
Covers universe construction, statistical pair selection, supervised ML signal filtering,
and a rigorous in-sample / out-of-sample backtest with transaction costs.

---

## Table of Contents
1. [Overview](#overview)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Honest Limitations](#honest-limitations)
5. [What I'd Do Differently](#what-id-do-differently)
6. [Project Structure](#project-structure)
7. [Setup](#setup)

---

## Overview

Classical pairs trading enters a long-short position when two historically cointegrated
assets diverge, betting on mean reversion. The challenge is distinguishing genuine
dislocations from structural breaks — situations where the relationship has permanently
changed and will not revert.

This project attacks that problem two ways:

1. **Rigorous pair selection** — Johansen cointegration testing and Hurst exponent
   filtering on in-sample data only, so the selected pairs have statistically verified
   mean-reversion properties before OOS trading begins.

2. **ML signal filtering** — a logistic regression classifier predicts whether the spread
   is likely to revert over the next 5 trading days. Only trades where the model is
   sufficiently confident (sigmoid probability > 0.55) are taken.

The project is honest about its findings: the 2024–2025 OOS period was hostile to
mean-reversion strategies (macro regime shifts, tariff shock), and the naive strategies
lost money. Adding principled risk controls — a stop-loss and a cointegration stability
gate — substantially reduced those losses. The full ML pipeline with walk-forward
retraining produced a positive Sharpe, though with a trade count too small to be
statistically conclusive.

---

## Methodology

### 2.1 Data and Time Period

- **Source:** Yahoo Finance via `yfinance` (daily adjusted close prices)
- **Universe:** 40 ETFs spanning sector, sub-sector, country, commodity, style, and
  bond exposures (see notebook 01 for full list)
- **Full period:** 2020-01-01 to 2025-12-31
- **In-sample (IS):** 2020-01-01 – 2023-12-31 (≈ 754 trading days after rolling-window warmup)
- **Out-of-sample (OOS):** 2024-01-01 – 2025-12-31 (502 trading days)

The IS/OOS split is a hard calendar date. No data after 2023-12-31 influenced any
model fitting, pair selection, or hyperparameter choice.

---

### 2.2 Pair Selection (In-Sample Only)

All 780 candidate pairs from C(40, 2) were evaluated on IS data using a two-stage filter:

**Stage 1 — Johansen cointegration test**

For each pair, log prices are tested for cointegration using the Johansen trace test
(95% significance level). This tests whether a linear combination of the two log-price
series is stationary — a necessary condition for mean reversion. 603 of 780 pairs failed
this test.

The Johansen test also provides the **cointegrating vector** (hedge ratio), which is
used to construct the spread. This vector is estimated on IS data only and frozen for
the entire OOS period.

**Stage 2 — Hurst exponent filter**

For each of the 177 cointegrated pairs, the Hurst exponent is computed on the
cointegrating spread. H < 0.5 indicates mean-reverting dynamics (each move tends to
reverse). All 177 surviving pairs satisfied this criterion.

**Ranking — Ornstein-Uhlenbeck half-life**

The 177 pairs are ranked by the OU half-life of the spread: the expected time for
the spread to revert halfway to its mean after a shock. Shorter half-life = faster
mean reversion = more tradeable at daily frequency. The top 15 pairs are selected.

| Rank | Pair | Johansen excess | Hurst | Half-life (days) |
|------|------|----------------|-------|-----------------|
| 1 | EWJ / HYG | +36.2 | 0.163 | 12.9 |
| 2 | HYG / XLC | +16.0 | 0.232 | 17.0 |
| 3 | XLU / XOP | +2.7 | 0.249 | 19.0 |
| 4 | HYG / MTUM | +12.1 | 0.234 | 19.9 |
| 5 | EWC / IWD | +1.0 | 0.311 | 19.9 |
| 6 | GDXJ / LQD | +3.6 | 0.398 | 20.1 |
| 7 | USO / XLU | +14.3 | 0.253 | 20.2 |
| 8 | EWG / HYG | +17.8 | 0.258 | 20.5 |
| 9 | EWJ / XLC | +9.2 | 0.334 | 20.8 |
| 10 | EWC / XLF | +1.6 | 0.393 | 21.9 |
| 11 | GDXJ / USO | +12.7 | 0.366 | 22.5 |
| 12 | XLP / XLV | +9.3 | 0.354 | 23.0 |
| 13 | XLP / XLU | +3.0 | 0.226 | 23.4 |
| 14 | EWA / XLB | +2.1 | 0.372 | 23.5 |
| 15 | GDX / USO | +9.9 | 0.400 | 24.4 |

---

### 2.3 Strategy Logic

**Spread construction**

For pair (A, B) with Johansen cointegrating vector [w₁, w₂]:

```
spread(t) = w₁ · log(price_A(t)) + w₂ · log(price_B(t))
```

The hedge ratio [w₁, w₂] is the IS Johansen eigenvector, frozen throughout OOS.

**Rolling Z-score**

```
z(t) = (spread(t) - mean(spread, 252d)) / std(spread, 252d)
```

The 252-day backward-looking window is re-evaluated daily. At the start of OOS, this
window pulls from late IS data, which is correct — it provides context for the
early OOS z-scores without any lookahead.

**Baseline Z-score strategy**

- Long spread when z < −2 (spread unusually low, expected to revert up)
- Short spread when z > +2 (spread unusually high, expected to revert down)
- Exit when z crosses 0
- Positions held continuously until exit signal

**ML-filtered strategy**

Identical to the baseline, but entries additionally require:
- Sigmoid (logistic regression) probability > 0.55

The ML model predicts whether |z| will decrease over the next 5 trading days — i.e.,
whether the spread is genuinely mean-reverting. It gates entry only; exits are still
purely z-score based.

**Enhanced strategies (risk controls)**

Two principled additions applied on top of both strategies:

1. **Cointegration stability gate:** Only enter a trade if the rolling 252-day ADF
   p-value of the spread is < 0.10. If the spread is no longer behaving as stationary,
   no new positions are opened for that pair.

2. **Stop-loss:** Exit a trade if |z| moves beyond 3.5 in the wrong direction.
   A long spread entered at z = −2 is stopped out if z falls below −3.5, capping the
   maximum loss per trade.

---

### 2.4 Machine Learning Model

- **Algorithm:** Logistic Regression with StandardScaler (scikit-learn Pipeline)
- **Features (8):** `z_score`, `spread_change`, `spread_vol_20`, `adf_p_252`,
  `hurst_252`, `adf_stationary`, `hurst_mr`, `regime_score`
- **Excluded:** raw spread level (non-stationary, incomparable across pairs)
- **One model per pair** — each pair's dynamics are modelled independently
- **Training:** IS data only (≤ 2023-12-31), hard calendar split
- **Walk-forward variant:** model retrained every 6 months on expanding data
  (2024-H1 uses IS model; 2024-H2 retrains including 2024-H1; etc.)

Mean OOS AUC across 15 pairs: **0.595** (range: 0.539 – 0.681)

---

### 2.5 Backtesting Framework

- **Period reported:** OOS only (2024-01-01 onwards)
- **Portfolio:** Equal-weight across all 15 pairs (mean of daily spread returns)
- **Transaction costs:** 5 bps per leg = 10 bps per round-trip
  Applied when position changes; charged on the same day as the position update.
- **Position logic:** State machine (no forward-fill hacks); positions computed from
  the start of IS data so state carries over correctly into OOS.

---

## Results

### Performance Table — OOS 2024–2025

| Strategy | Costs | Total Return | Sharpe (ann.) | Max Drawdown | Trades | Win Rate |
|----------|-------|-------------|--------------|-------------|--------|---------|
| 1. Baseline Z-score | None | −20.81% | −1.097 | −24.93% | 39 | 61.5% |
| 1. Baseline Z-score | 10 bps RT | −20.97% | −1.107 | −25.03% | 39 | 61.5% |
| 2. Baseline + Risk Controls | None | −5.35% | −0.686 | −11.23% | 22 | 54.5% |
| 2. Baseline + Risk Controls | 10 bps RT | −5.46% | −0.701 | −11.26% | 22 | 54.5% |
| 3. ML fixed model | None | −20.13% | −1.071 | −24.99% | 35 | 57.1% |
| 3. ML fixed model | 10 bps RT | −20.28% | −1.080 | −25.07% | 35 | 57.1% |
| 4. ML walk-forward + RC | None | +1.54% | +0.997 | −0.68% | 7 | 100% |
| 4. ML walk-forward + RC | 10 bps RT | +1.50% | +0.974 | −0.69% | 7 | 100% |

### Discussion

**Risk controls matter more than the ML model.**
The clearest finding is that strategy 2 (risk controls alone, no ML) reduced the loss
from −21% to −5% and halved the drawdown. The stop-loss prevented a few large trades
from dominating the portfolio, and the cointegration gate stopped entering trades after
the IS cointegration relationships had broken down in 2024–2025.

**The fixed ML model barely helped.**
Strategy 3 is nearly identical to strategy 1. Filtering by a model trained once on
2020–2023 and applied frozen for two years provided almost no benefit: the model's
learned signal distribution became stale as the market regime changed.

**Walk-forward ML + risk controls turned positive — but interpret cautiously.**
Strategy 4 shows +1.54% return with a Sharpe near 1.0. However, it executed only
7 trades in 2 years across 15 pairs. A 7-trade sample provides no statistical power
to distinguish skill from luck. The 100% win rate is not a reliable signal.
The most honest reading is: the combination of regime-aware controls (cointegration
gate + VIX filter) and a freshly-retrained model became very selective, avoiding
nearly all the losing trades. Whether that selectivity would persist out of this
OOS window is unknown.

**Transaction costs are not the problem.**
Even strategy 1 with 39 trades incurs only ~16 bps total cost impact on a −20% return.
The issue is signal quality, not friction.

**Why did 2024–2025 hurt mean-reversion strategies?**
The OOS period included two major regime events: the Fed's rate pivot cycle through 2024
and the April 2025 tariff shock (VIX hit 52). Both caused correlated ETF moves that
overwhelmed the mean-reversion signal. The cointegration gate detects this after the
fact — once ADF p-values rise — but cannot anticipate it in advance.

---

## Honest Limitations

**Daily data has limited signal-to-noise for mean reversion.**
ETF pairs trading is fundamentally a microstructure phenomenon. At daily frequency,
positions must be held for 13–25 days (the OU half-lives of the selected pairs),
giving ample time for macro shocks to override the mean-reversion signal. The same
strategy at intraday frequency would have shorter holding periods and a cleaner
signal-to-noise ratio.

**Cointegration is not stable across regimes.**
IS pair selection verified cointegration over 2020–2023. Several of those relationships
weakened in 2024–2025. There is no guarantee that pairs selected today will remain
cointegrated for the next two years. Rolling re-evaluation of pair validity is necessary
in a live system.

**Transaction cost assumption is a heuristic.**
The 5 bps per leg assumption is a reasonable approximation for liquid ETFs but does not
account for market impact, bid-ask spread variability, or timing of execution. Real
costs depend on position size, time of day, and market conditions.

**The ETF universe is curated, not exhaustive.**
The 40-ticker universe was chosen to span economically related but non-identical
exposures. Different universe choices would produce different pair sets. The results
are specific to this universe and this IS period.

**Small OOS trade count limits statistical inference.**
39 baseline trades across 15 pairs over 2 years is insufficient for robust inference
about the strategy's true edge. Reliable Sharpe estimates typically require 100+ trades.
The results should be treated as exploratory findings, not production performance claims.

---

## What I'd Do Differently

**Higher frequency data** is the highest-leverage improvement. At 5-minute or hourly
bars, OU half-lives would be hours rather than weeks, trades would close faster, and
the macro regime risk per trade would be much smaller. The methodology is identical —
only the data changes.

**Rolling pair re-selection** every quarter using an expanding IS window would detect
when cointegration breaks down and retire pairs that are no longer mean-reverting,
replacing them with new candidates from the universe.

**Position sizing proportional to z-score** — rather than binary ±1 positions, scale
size by how extreme the z-score is. A z of 3.0 gets a larger position than z of 2.1.
This naturally reduces exposure to marginal signals.

**Richer ML features** including macro regime variables (VIX level, credit spread,
yield curve slope) would help the model distinguish temporary dislocations from
structural breaks — which is exactly the failure mode observed in 2024–2025.

---

## Project Structure

```
├── data_raw/
│   └── prices.csv                    # Daily ETF close prices (40 tickers)
├── data_processed/
│   ├── selected_pairs.csv            # 15 selected pairs with hedge ratios
│   ├── features_{T1}_{T2}.csv        # Per-pair engineered features
│   ├── dataset_{T1}_{T2}.csv         # Features + labels per pair
│   └── predictions_{T1}_{T2}.csv     # Model probabilities (IS + OOS)
├── notebooks/
│   ├── 01_data_download.ipynb        # Universe download with caching
│   ├── 02_pair_selection.ipynb       # Johansen + Hurst + OU ranking (IS only)
│   ├── 03_feature_engineering.ipynb  # Per-pair z-score, ADF, Hurst features
│   ├── 04_label_creation.ipynb       # Binary labels: will spread revert?
│   ├── 05_model_training.ipynb       # Per-pair LogisticRegression (IS only)
│   └── 06_backtest_and_evaluation.ipynb  # OOS backtest, all strategies
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py        # hurst_exponent, hedge_ratio, etc.
│   ├── backtest.py
│   ├── models.py
│   ├── labeling.py
│   └── utils.py                      # sharpe_ratio, max_drawdown
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06
# Notebook 01 caches prices.csv; subsequent runs skip the download.
```

**Dependencies:** `yfinance`, `pandas`, `numpy`, `matplotlib`, `statsmodels`, `scikit-learn`

---

*This project is for educational and research purposes only and does not constitute financial advice.*
