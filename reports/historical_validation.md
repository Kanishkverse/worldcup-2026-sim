# Historical validation

Real scorelines from four famous group stages, plus the 2010-2022 champion backtest. Each section ends in a pass/fail gate; the same gates are asserted in tests/test_historical_validation.py.

**Overall: PASS** (ingestion PASS, calibration PASS, bayes PASS, form PASS, correction PASS)

## 1. Results ingestion

Ingest the real scorelines, rebuild the table, and compare the computed order to the real final standings. This exercises the tiebreakers too: three of these four groups were settled on goal difference.

| Tournament | Group | Computed order | Matches reality |
|---|---|---|---|
| 2018 | D | Croatia, Argentina, Nigeria, Iceland | PASS |
| 2018 | F | Sweden, Mexico, South Korea, Germany | PASS |
| 2022 | C | Argentina, Poland, Mexico, Saudi Arabia | PASS |
| 2022 | E | Japan, Spain, Germany, Costa Rica | PASS |

## 2. Calibration

Backtest the match model on the four 32-team World Cups and score the champion probabilities. Uniform 1-of-32 guessing scores Brier 0.97 / log loss 3.47; the model must beat both.

| Year | Champion | Model rank | Champion prob | Brier | Log loss |
|---|---|---|---|---|---|
| 2010 | Spain | 2 | 16.9% | 0.786 | 1.776 |
| 2014 | Germany | 3 | 13.8% | 0.851 | 1.981 |
| 2018 | France | 4 | 8.6% | 0.981 | 2.453 |
| 2022 | Argentina | 2 | 23.7% | 0.669 | 1.438 |

- Mean Brier **0.822** (uniform 0.97), mean log loss **1.912** (uniform 3.47), champion mean rank **2.75** of 32. PASS

## 3. Bayesian updates

Feed each group's real results through the state engine and ask whether the posterior ordering ends up closer to the real final table than the pre-tournament Elo was. Closeness is Kendall tau against the actual order (1.0 = identical). The posterior should also be more certain than the prior (smaller rating std).

| Tournament | Group | Prior tau | Posterior tau | Std shrinks |
|---|---|---|---|---|
| 2018 | D | +0.33 | +0.67 | PASS |
| 2018 | F | -0.33 | +0.00 | PASS |
| 2022 | C | +0.67 | +0.67 | PASS |
| 2022 | E | +0.33 | +0.33 | PASS |

- Mean ordering agreement with reality: prior **+0.25** -> posterior **+0.42**. Conditioning on results moves the ranking toward the truth. PASS

## 4. Form dynamics

Track latent form through each real sequence. It must stay bounded, move with over/under-performance, and the famous trajectories must look right: a shock spikes and then regresses, a collapse trends down, a recovery turns back up.

- Boundedness: max |form| observed **1.04** <= clip 2.5. PASS
- Directionality: corr(beating expectation, form change) **+0.92** (> 0.5). PASS
- Saudi Arabia 2022 spikes after beating Argentina (+0.60) then regresses (-0.23). PASS
- Argentina 2022 dips after the Saudi loss (-0.60) then recovers (+0.37). PASS
- Germany 2018 trends down to elimination (-0.79). PASS
- Japan 2022 ends on a high (+0.35). PASS

## 5. Group-to-knockout correction

The same residual-to-correction step the 2026 pipeline runs, here on real groups. Expected goal difference comes from the rating gap; the residual against the real result becomes an Elo correction. The corrected order should sit closer to the real final table than the prior, and the famous shocks should be signed correctly.

| Tournament | Group | Prior tau | Corrected tau |
|---|---|---|---|
| 2018 | D | +0.33 | +0.67 |
| 2018 | F | -0.33 | +0.00 |
| 2022 | C | +0.67 | +0.67 |
| 2022 | E | +0.33 | +0.33 |

- Ordering vs reality: prior **+0.25** -> corrected **+0.42**. PASS
- Saudi Arabia 2022 corrected up (+32 Elo), Germany 2018 corrected down (-92 Elo). PASS
