# worldcup-2026-sim

Three engines for simulating the 2026 FIFA World Cup, and the experiment that
decided which one to trust.

1. **Statistical engine** — a calibrated minute-by-minute Poisson Monte Carlo
   with an optional LLM tactical layer, backtested on the 2010–2022 World Cups.
2. **Agentic engine** — all 48 real squads, 1,248 real players, every manager,
   striker and keeper played by an LLM agent making bounded decisions inside
   the validated core. Runs on a single consumer GPU with any
   OpenAI-compatible endpoint.
3. **Prediction engine** — pure Elo-to-Poisson Monte Carlo blended with
   de-vigged bookmaker odds. 20,000 tournaments in about 8 seconds. This is
   the one that produces the numbers I'd actually bet on.

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

## Quickstart

```bash
pip install -r requirements.txt

# the fast one: validate the match model, then predict 2026
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
scripts/                squad-PDF parser, reports, figures, ablation analysis
data/                   1,248 real players parsed from the FIFA squad lists
results/                probability tables, backtests, ablation statistics
figures/                all charts
docs/                   full project documentation (PDF) + GPU setup notes
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
