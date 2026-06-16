# Migration notes: static forecast to dynamic state (v3)

Nothing in the original engine was removed. `worldcup_2026_sim.py`,
`elo_predict.py`, `market_blend.py` and the agentic simulator all run exactly
as before. v3 adds a layer on top, in the new `worldcup_sim/` package, plus
one new entry point, `live_forecast.py`.

## What changed conceptually

The old pipeline answered one question once: given pre-tournament strength,
who wins? Team strength was a fixed number, so matchday three played the same
as matchday one and the only way to "update" for a real result was to pin its
scoreline and re-run. The forecast itself never learned anything from it.

v3 makes strength a distribution that moves. Each played match runs a
Normal-Normal Bayesian update on both teams: the mean shifts toward the
performance the result implies, and the variance shrinks because we now know
more. The Monte Carlo samples that posterior, so early upsets widen a team's
cone of outcomes instead of nudging a point estimate. On top of that sit
style matchups, venue and travel load, latent form, and matchday tactical
intent, none of which existed before.

## What you run now

| Before | After |
|---|---|
| `python3 elo_predict.py --runs 20000` (static) | `python3 live_forecast.py --runs 4000` (conditioned on results) |
| pin results via `WC_RESULTS_CSV`, re-run | edit `data/results_2026.csv`, re-run; ratings update themselves |
| no upset calibration | `python3 scripts/calibrate_upsets.py` |
| no probabilistic scoring | `python3 scripts/model_diagnostics.py` |

The old static prediction is still the right tool before a ball is kicked.
Once results start arriving, `live_forecast.py` is the one to use.

## Data you can edit

- `data/results_2026.csv` - the played matches. Columns:
  `group,matchday,home,away,home_goals,away_goals,venue,winner`. The `winner`
  column is only needed for knockout ties decided on penalties. Add a row,
  re-run, done.
- `data/team_styles.csv` - the six style dimensions per team.
- `data/venues.csv` - stadium altitude, climate and coordinates.

## Tuning knobs

- `worldcup_sim/engine.py: GAP_SCALE` - upset rate. `calibrate_upsets.py`
  explains why it ships at 0.85.
- `worldcup_sim/tournament_state.py: PRIOR_STD, OBS_STD` - how hard a single
  result moves a rating. Wider `PRIOR_STD` or smaller `OBS_STD` makes the
  model more reactive to early shocks.
- `worldcup_sim/form.py: FORM_DECAY, FORM_GAIN` - how much momentum is allowed
  before it regresses.

## Compatibility

- Python and dependencies are unchanged except for `pytest` (tests) and
  `Pillow` (diagram QA only), both dev-only.
- Names are shared with the base engine through `worldcup_sim/priors.py`, so
  the Elo priors and the official 2026 bracket are never duplicated.
- The new package imports `worldcup_2026_sim` for those priors and the bracket
  slot tables; keep both importable from the repo root.
