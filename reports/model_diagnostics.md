# Model diagnostics

Backtest of the new match model on the 32-team World Cups, 2500 simulations per tournament. No in-tournament updates here: this isolates the match engine and bracket, scored against the real champion and semifinalists.

## Champion scoring

| Year | Champion | Model rank | Champion prob | Brier | Log loss |
|---|---|---|---|---|---|
| 2010 | Spain | 2 | 16.4% | 0.7997 | 1.810 |
| 2014 | Germany | 2 | 13.4% | 0.8570 | 2.010 |
| 2018 | France | 4 | 8.9% | 0.9699 | 2.421 |
| 2022 | Argentina | 2 | 24.1% | 0.6721 | 1.422 |

- Mean Brier score: **0.8247** (lower is better; a uniform 1-of-32 guess scores about 0.97).
- Mean log loss: **1.916** (uniform guess scores about 3.47).

## Calibration: reach the semifinal

All teams across all four tournaments, bucketed by predicted probability of reaching the semifinal, against how often it actually happened.

| Predicted bin | Teams | Mean predicted | Actual rate |
|---|---|---|---|
| 0.00-0.10 | 77 | 3.2% | 2.6% |
| 0.10-0.20 | 22 | 15.7% | 18.2% |
| 0.20-0.35 | 17 | 27.3% | 23.5% |
| 0.35-0.50 | 8 | 40.9% | 50.0% |
| 0.50-1.01 | 4 | 54.8% | 50.0% |

A well-calibrated model tracks the diagonal: the predicted and actual columns should be close in every bin. The known soft spot is form blindness. The model rates on pre-tournament Elo, so a side that catches fire (2018 France, 2022 Argentina from their respective ranks) is scored as a mid-pack contender. That is exactly the gap the live engine's Bayesian updates and form layer are built to close once real results arrive.
