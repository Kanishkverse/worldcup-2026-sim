# Contributing

I built this to settle one question (do LLM agents change simulated
tournament outcomes?) and ended up with a testbed that can settle a lot more
of them. PRs are welcome. Issues with experiment results are even more
welcome.

## Ground rules

1. **Calibration is non-negotiable.** Any change to match resolution must
   keep goals per match in the 2.5–2.8 band across at least 500 simulated
   matches (real World Cups 2010–2022 average 2.64). Post the number in your
   PR.
2. **If you claim an effect, bring the ablation.** Run your variant with
   `--random-agents` (same decision points, uniform random choices) and show
   the champion distributions differ. A chi-square p-value and distinct-champion
   count is enough. `scripts/ablation_analysis.py` does the work.
3. **No silent fallbacks.** If your agent layer can fail, count the failures
   and report them, the way `LLMClient` does. A sim that quietly degrades to
   heuristics is worse than one that crashes.
4. **Seeds in, numbers out.** Every run takes `--seed`. Note that LLM
   sampling through most servers is unseeded, so same-seed agentic runs are
   independent samples, not replicas. Pool them; don't diff them.

## Open experiments I'd take a PR for

- **Stronger models vs the ablation.** All shipped agentic results are
  qwen2.5 14B. Does a frontier model produce a champion distribution that
  actually beats coin-flips? Everything routes through `--base-url`/`--model`,
  so this is config, not code.
- **Wider clamps, recalibrated.** `plan_to_mods()` clamps manager influence
  to about ±15%. Raise it to ±35%, re-tune `base_xg` until calibration holds,
  re-run the ablation. This is the single most interesting open question in
  the repo.
- **Stateful agents.** Managers and players currently decide from a one-shot
  prompt. Persistent personas with form, confidence, and cross-match fatigue
  might behave differently. Cross-match fatigue carry-over is a good first
  cut and doesn't need an LLM at all.
- **A better prediction engine.** `elo_predict.py` is deliberately minimal
  (Elo, host bump, calibrated Poisson). Form-weighted ratings, injury inputs,
  or a market time series would all plausibly improve the 2010–2022 backtest.
  The bar: actual champion's mean rank across the four backtest years is
  currently 2.5.
- **Penalty shootout model.** Currently a small Elo tilt around 50/50. Real
  shootout data exists; someone should fit it.
- **More tournaments.** The bracket logic handles the 48-team 2026 format and
  the 32-team 2010–2022 format. Euros and Copa América are mostly bracket
  tables away.

## Dev setup

```bash
pip install -r requirements.txt
export WC_PLAYERS_CSV=data/players_2026_real.csv

# fast sanity checks (no LLM, no network)
python3 agentic_wc2026.py --offline --runs 2 --out-prefix /tmp/check
python3 elo_predict.py --backtest --backtest-runs 200
```

Both should finish in under a minute. If you touch the match model, run the
full backtest before opening the PR.

## Style

Plain Python, type hints where they help, docstrings that say why rather
than what. Keep modules flat and runnable as scripts; this is research code
that people should be able to read top to bottom.
