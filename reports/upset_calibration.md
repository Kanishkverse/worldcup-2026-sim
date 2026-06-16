# Upset calibration

Pairings sampled from real group draws: 192 (World Cups 2010, 2014, 2018, 2022).

Outcomes are scored from the favourite's side, by pre-tournament Elo.

## Target (historical)

- favourite 49.0% / draw 26.0% / underdog 25.0%

## Model at GAP_SCALE = 1.0 (default)

- favourite 61.7% / draw 21.0% / underdog 17.3%

## SSE-optimal for the bare match model: GAP_SCALE = 0.5

- favourite 51.0% / draw 24.5% / underdog 24.6%

## Shipped default: GAP_SCALE = 0.85

- favourite 58.8% / draw 22.0% / underdog 19.2%

## Sweep

| GAP_SCALE | favourite | draw | underdog | sq. error |
|---|---|---|---|---|
| 0.50 | 51.0% | 24.5% | 24.6% | 0.00064 |
| 0.55 | 52.1% | 23.7% | 24.2% | 0.00157 |
| 0.60 | 53.0% | 23.7% | 23.2% | 0.00247 |
| 0.65 | 54.6% | 23.2% | 22.2% | 0.00464 |
| 0.70 | 55.9% | 22.8% | 21.3% | 0.00712 |
| 0.75 | 56.9% | 23.1% | 20.0% | 0.00954 |
| 0.80 | 57.5% | 23.0% | 19.5% | 0.01125 |
| 0.85 | 58.7% | 22.7% | 18.6% | 0.01474 |
| 0.90 | 59.8% | 22.0% | 18.2% | 0.01785 |
| 0.95 | 60.3% | 21.6% | 18.1% | 0.01938 |
| 1.00 | 61.7% | 21.0% | 17.3% | 0.02437 |
| 1.05 | 62.2% | 21.3% | 16.5% | 0.02689 |
| 1.10 | 62.9% | 21.2% | 15.9% | 0.03012 |
| 1.15 | 64.0% | 20.3% | 15.7% | 0.03423 |
| 1.20 | 64.5% | 20.2% | 15.2% | 0.03703 |

**Verdict.** The bare match model, with strength held fixed, is favourite-heavy: at GAP_SCALE 1.0 it gives the favourite far more wins and far fewer draws than the real tournaments did. Matching history on this statistic alone wants heavy compression of the rating gap. But this test deliberately freezes everything else. The live engine samples each rating from its posterior, bends expected goals by style matchup, and loads legs by venue and travel, all of which add upset variance the bare model has none of. So the engine ships GAP_SCALE = 0.85: it removes the worst of the favourite bias while leaving room for the stochastic layers to do the rest, instead of compressing the gap so hard that real strength differences vanish from the forecast.

Euro 2024 and Copa America 2024 are good next calibration sets; their draws are not bundled in the repo yet, so they are an open data task in CONTRIBUTING.md rather than a number invented here.
