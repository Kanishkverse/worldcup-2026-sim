# worldcup-2026-sim

Four engines for simulating the 2026 FIFA World Cup, and the experiment that
decided which one to trust.

1. **Statistical engine** — a calibrated minute-by-minute Poisson Monte Carlo
   with an optional LLM tactical layer, backtested on the 2010–2022 World Cups.
2. **Agentic engine** — all 48 real squads, 1,248 real players, every manager,
   striker and keeper played by an LLM agent making bounded decisions inside
   the validated core. Runs on a single consumer GPU with any
   OpenAI-compatible endpoint.
3. **Prediction engine** — pure Elo-to-Poisson Monte Carlo blended with
   de-vigged bookmaker odds. 20,000 tournaments in about 8 seconds. This is
   the one that produces the numbers I'd actually bet on before kickoff.
4. **Live engine** — a dynamic tournament-state simulator (`worldcup_sim/`)
   that updates as results arrive. Strength is a Bayesian distribution that
   moves after every played match, on top of style matchups, venue and travel
   load, latent form, and matchday-aware tactics. This is the one to use once
   the tournament is actually running.

The headline finding sits in between: I ran an ablation where every agent
brain was replaced with a coin-flip, and the championship distribution didn't
change (chi-square p = 0.28, 24 vs 25 distinct champions over 100 runs each).
**Agents change the story of a match, not who wins the Cup.** Full writeup in
[`docs/WorldCup2026_Documentation.pdf`](docs/WorldCup2026_Documentation.pdf)
and [`results/ablation_results.md`](results/ablation_results.md).

![Title odds with the agents' brains removed](figures/viz_ablation.png)

## Current prediction (June 12, 2026)

Geometric blend of the Elo Monte Carlo (validated on four previous World
Cups: actual champion ranked 1st/3rd/4th/2nd) with de-vigged outright market
odds:

| Country | Win % | Elo model | Market |
|---|---|---|---|
| Spain | **27.3** | 38.9 | 15.5 |
| Argentina | **14.2** | 21.0 | 7.8 |
| France | **12.8** | 9.0 | 14.8 |
| England | **11.3** | 10.2 | 10.0 |
| Brazil | **7.0** | 4.8 | 8.1 |
| Portugal | **6.5** | 3.6 | 9.5 |

## Live engine: the forecast that moves with the tournament

The numbers above are the right answer the night before the opening game. They
are the wrong answer the moment results start landing, and the group stage of
2026 landed hard: Spain held 0-0 by Cabo Verde, Qatar drawing Switzerland,
Brazil pegged back by Morocco, Germany putting seven past Curaçao. A static
pre-tournament model cannot see any of that. The live engine is built to.

Every played match runs a Normal-Normal Bayesian update on both teams: the
rating mean moves toward what the result implies and the variance shrinks as
evidence accumulates. The Monte Carlo then samples that posterior, so an early
slip widens a team's spread of outcomes rather than just shaving a point
estimate. Style matchups (±15%), venue and travel load (Mexico City altitude,
Gulf-coast heat, cross-continent hops), latent form, and matchday tactical
intent all sit on top.

![Dynamic engine architecture](figures/arch_v3.png)

What conditioning on the first fourteen results does to the title odds, same
engine, with and without the real games fed in:

| Team | Before (pre-tournament) | After (results conditioned) |
|---|---|---|
| Spain | 22.6% | **7.9%** |
| Argentina | 18.4% | **23.8%** |
| England | 10.4% | **14.0%** |
| France | 10.4% | **13.5%** |
| Brazil | 6.0% | **5.1%** |
| Portugal | 5.4% | **6.2%** |

Spain are the story: a goalless opener against a side they were a heavy
favourite over, against a defensive block their possession game runs straight
into, drags their posterior down and cuts their title odds by two thirds.
Argentina, yet to play, inherit the favourite's slot. That is the realism the
static model was missing, and it updates the moment you add a row to a CSV.

The match model is honest about its own limits. `scripts/calibrate_upsets.py`
backtests the favourite-win, draw and underdog rates against the real
2010–2022 group stages, and `scripts/model_diagnostics.py` scores the champion
probabilities with Brier and log loss plus a calibration curve. Reports land
in [`reports/`](reports/).

## Player-level engine: strength built from the squad up

The engines above carry one rating per team. This one builds the rating from
the players who are actually available, so the teamsheet matters. Each of the
1,248 real players has a rating, form, club minutes and a position; the
aggregation leans on the best of them (a star is worth more than his slot in
an average) and maps the result to Elo. Rule a player out and the number
moves.

`scripts/per_game_demo.py` runs a single fixture through the real match
physics, many times, and reports the whole outcome distribution:

```
France vs Mexico  (2500 simulations)
  France win  52.3%   draw  26.8%   Mexico win  19.8%
  expected goals 1.62 - 0.92   over 2.5: 47%
  most likely 1-1   top scorelines: 1-1 13%, 1-0 12%, 2-0 10%, 2-1 10%
```

This is the base engine's minute-by-minute simulation: formation-based XI
selection, the player cohesion graph, accumulating fatigue, sendings-off,
substitutions, the manager's half-time tactical call, and extra time plus
penalties in a knockout. Rule a player out and the squad reshapes around the
absence in the physics, not by hand.

One honest caveat the data forces. In this dataset elite ratings saturate near
the top, so a single star off a deep squad is within Monte Carlo noise at the
match level: France lose Mbappe and their win probability barely moves, because
Dembele and Thuram are right there. The player signal is real but it lives in
the aggregate (Mbappe is about 11 Elo) and compounds over a tournament; take
out France's whole first XI and the per-game swing is unmistakable
(52% to 48%, expected goals 1.61 to 1.52). Depth absorbs single absences,
which is the correct behaviour, not a bug.

### Group results as a calibration fold for the knockouts

`scripts/group_to_ko_correction.py` does what you would actually want once
real group games are in: for every played match it compares the per-game
prediction to the result, turns the gap into a per-team Elo correction, and
carries that into the elimination rounds. Germany putting seven past Curacao
earns a positive correction and their title odds climb; Spain's goalless draw
with Cabo Verde earns a negative one and theirs fall. The mechanism is
validated against real tournaments in
[`reports/historical_validation.md`](reports/historical_validation.md): applied
to four famous group stages it moves the ratings closer to the real final
tables (Kendall tau +0.25 to +0.42) and signs the shocks correctly
(Saudi Arabia 2022 up, Germany 2018 down).

### A single fixture, live, and against the market

`scripts/match_market.py` runs one game and sets it beside the market. With
`--live` it plays the teams at their current tournament rating, not their
pre-tournament one: the played results are replayed onto the player-aggregated
ratings, and the move is applied to the squad as a form shift. So Spain after
their goalless draw with Cabo Verde are a weaker side here than they were at
the draw, and the simulator knows it.

The market line comes from real 1X2 prices when you have them (on the command
line or in `data/match_odds_2026.csv`) and is derived from the rating gap when
you do not. Either way the script prints the model, the de-vigged market, a
geometric blend, and the edge:

```
France vs Mexico  (model)  France 52.0%  draw 26.7%  Mexico 21.3%
  outcome        model   market   blend    edge
  France win     52.0%    58.7%   55.4%    -6.7
  draw           26.7%    24.4%   25.6%    +2.3
  Mexico win     21.3%    17.0%   19.1%    +4.3
```

The market is a layer on top, never an input to the simulation. The physics is
not told what the bookmakers expect.

## Quickstart

```bash
pip install -r requirements.txt

# the live engine: condition the forecast on results as they come in
python3 live_forecast.py --runs 4000              # reads data/results_2026.csv
python3 scripts/calibrate_upsets.py               # upset rate vs 2010-2022
python3 scripts/model_diagnostics.py              # Brier / log loss / calibration
python3 scripts/validate_historical.py            # validate the machinery on real tournaments
python3 -m pytest -q                              # the test suite

# the player-level engine: strength from the squad, real per-game physics
python3 scripts/per_game_demo.py                  # one fixture, with/without a player
python3 scripts/group_to_ko_correction.py         # group results -> knockout correction
python3 scripts/match_market.py France Mexico --live   # one game, live, vs the market

# the fast static one: validate the match model, then predict 2026 pre-kickoff
python3 elo_predict.py --backtest
python3 elo_predict.py --runs 20000
python3 market_blend.py

# the statistical engine (heuristic tactics; add --llm + an OpenAI key for the agent layer)
python3 run_backtest.py
python3 run_2026.py

# the agentic engine (needs an OpenAI-compatible endpoint, e.g. local Ollama)
export WC_PLAYERS_CSV=data/players_2026_real.csv
python3 agentic_wc2026.py --demo "Spain,France"            # one verbose match
python3 agentic_wc2026.py --runs 100 --seed 2026           # full Monte Carlo
python3 agentic_wc2026.py --random-agents --runs 100       # the ablation arm
```

The agentic engine talks to any OpenAI-protocol server via `--base-url` and
`--model` (defaults to a local Ollama with qwen2.5:14b-instruct). GPU notes
for cheap cloud rentals are in [`docs/README_RUNPOD.md`](docs/README_RUNPOD.md).

## How the agentic engine stays honest

My first version let agents drive the match directly. It produced 3.61 goals
per match (real World Cups: 2.64) and 95% of goals from passing moves,
because the model likes passing and my rules rewarded it. The simulation was
measuring the LLM's personality, not football.

The shipped architecture keeps the validated Poisson core as physics and lets
agents steer at exactly three points:

- **Pre-match**: the manager's plan compiles to tactical modifiers clamped to
  roughly ±15%.
- **Every goal**: an attacker-vs-keeper duel that can deny the chance; the
  expected-goals math is compensated for the design denial rate so the
  average stays calibrated.
- **Half time**: real substitutions, with team chemistry recomputed.

Result across ~7,000 simulated matches: 2.69 goals per match, 10.1% denial
rate against a 10% target, zero silent failures (every parse fallback is
counted).

![Goals per match calibration](figures/viz_calib.png)

## Repository layout

```
worldcup_2026_sim.py    core engine: features, players, match model, 48-team bracket
agentic_wc2026.py       agentic engine + --random-agents ablation mode
run_2026.py             statistical Monte Carlo entry point
run_backtest.py         2010-2022 validation for the statistical engine
elo_predict.py          Elo->Poisson prediction engine (+ its own backtest)
market_blend.py         de-vig market odds, geometric blend, final table
live_forecast.py        live engine entry point: condition on real results
worldcup_sim/           dynamic state layer:
  tournament_state.py     Bayesian rating/form/fatigue updates
  styles.py               team style embeddings + matchup modifiers
  venues.py               altitude / heat / travel effects
  match_context.py        matchday tactical states
  form.py                 latent tournament form
  path_difficulty.py      bracket path strength tracking
  standings.py            group standings + 2026 tiebreakers
  engine.py               calibrated Poisson Monte Carlo, all layers wired in
  squads.py               player -> team aggregation, availability, player Elo
  per_game.py             single-fixture Monte Carlo over the real match physics
scripts/                parsers, reports, figures, ablation, calibration,
                          diagnostics, historical validation, per-game demo,
                          group-to-knockout correction
data/                   players, squads, styles, venues, live + historical results
results/                probability tables, backtests, ablation, live forecast
reports/                calibration, diagnostics, historical validation, correction
figures/                all charts incl. the v3 architecture diagram
tests/                  unit + integration tests (pytest)
docs/                   full project documentation (PDF) + GPU setup notes
MIGRATION.md            static-to-dynamic migration notes
```

## Contributing

This repo is set up so you can swap in your own ideas and measure them
against the baselines. The interesting open questions are in
[CONTRIBUTING.md](CONTRIBUTING.md) — the short version:

- **Bring your own LLM.** Everything goes through `--base-url`/`--model`.
  Does a stronger model break the ablation result?
- **Widen the clamps.** Give managers ±35% and recalibrate. Does agent
  intelligence start moving outcomes once it has real leverage?
- **Beat the blend.** The prediction engine is deliberately simple. Form
  inputs, injury data, market time series — anything that improves the
  backtest is welcome.

If you claim your agents change outcomes, include the ablation. That's the
house rule.

## License

MIT
