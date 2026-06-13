# Agentic 2026 World Cup Sim — RunPod 4090 + Ollama (qwen)

Every starter on the pitch is an LLM agent, every team has an LLM manager.
All 48 real FIFA squads (from the official squad-list PDF) play the official
2026 format: 12 groups → R32 bracket → final, 103 matches per tournament.

**v2: the agentic layer runs ON TOP of the original, backtest-validated match
engine** (`MatchSimulator`: minute-by-minute Poisson goal process, player-graph
cohesion, fatigue, red cards, threshold subs, calibrated penalties). The agents
steer it at three bounded points:

| Layer | Who decides | LLM calls |
|---|---|---|
| Pre-match | Both **manager agents** pick mentality / pressing; mapped to the engine's tactical modifiers, clamped to ±15% | 2 / match |
| Every scoring moment | The **attacking player's agent** chooses shoot / pass / dribble and the **goalkeeper agent** chooses hold / rush / narrow — bad choices get the goal **DENIED**, good ones keep it (base xG is compensated, so totals stay at the validated ~2.6–2.8 goals/match) | ~6 / match |
| Half time | Both **manager agents** adjust modifiers + up to 3 real-bench substitutions; the cohesion graph re-runs over the new XI | 2 / match |

Decisions move WHO scores and WHICH moments survive — not the physics of the
match. Real played 2026 results in `results_2026.csv` are pinned, not simulated.

**LLM failures are never silent:** unparseable replies fall back to a default
decision and are counted (reported at the end); 5 consecutive transport
failures abort the run.

---

## 1. Files to upload to the pod

```
agentic_wc2026.py        # this engine
worldcup_2026_sim.py     # bracket/format/feature core (imported)
players_2026_real.csv    # real FIFA squads -> per-player ratings
eloratings.csv           # team Elo snapshot
mv.csv                   # squad market values
results_2026.csv         # real played matches (pinned)
```

(Re-derive the players file anytime with `parse_squads_pdf.py` +
`build_real_players.py` + `SquadLists-English.pdf` + `spi_global_rankings.csv`,
but you don't need those on the pod.)

## 2. Create the pod

* GPU: **1× RTX 4090 (24 GB)** — community cloud is fine, ~$0.34–0.69/hr.
* Template: **RunPod PyTorch 2.x** (or any CUDA image with Python 3.10+).
* Container disk ≥ 25 GB (the model download needs ~10 GB).
* Expose nothing — everything runs on localhost inside the pod.

## 3. Install Ollama + the qwen model (inside the pod terminal)

```bash
curl -fsSL https://ollama.com/install.sh | sh

# serve with parallel slots so player/manager agents don't queue
OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=-1 ollama serve > /workspace/ollama.log 2>&1 &

ollama pull qwen2.5:14b-instruct     # ~9 GB, fits the 4090 with room to spare
# faster alternative:   ollama pull qwen2.5:7b-instruct
# qwen3 also works:     ollama pull qwen3:14b   (thinking blocks are stripped)
```

## 4. Python deps + smoke test

```bash
cd /workspace          # wherever you uploaded the files
pip install openai pandas numpy python-dotenv requests

# pipeline check without any LLM (instant):
python3 agentic_wc2026.py --offline --demo "Brazil,France"

# real agentic demo match, verbose play-by-play (~1 min):
python3 agentic_wc2026.py --demo "Brazil,France"
```

You should see manager plans, per-chance player decisions, GK calls, half-time
substitutions with real names, and an LLM accounting line with a low parse
fallback rate (<5 % is healthy for qwen2.5-14b).

## 5. Full tournament run(s)

```bash
# one full agentic World Cup (103 matches, ~1,000 LLM calls):
python3 agentic_wc2026.py --runs 1 --seed 2026

# outcome probabilities need many runs (each ~35–60 min on 14b):
nohup python3 agentic_wc2026.py --runs 35 --seed 2026 > sim.log 2>&1 &
tail -f sim.log
```

| Setup | LLM calls / tournament | Wall time / tournament (4090) |
|---|---|---|
| qwen2.5:14b, default | ~1,000 | ~35–60 min |
| qwen2.5:14b, `--no-keeper-agents` | ~700 | ~25–40 min |
| qwen2.5:7b-instruct | ~1,000 | ~15–25 min |

Useful flags: `--model`, `--base-url`, `--runs`, `--seed`,
`--no-keeper-agents` (skip GK agent calls), `--no-pin` (re-simulate already
played real matches), `--players` (alternate squad CSV).

## 6. Outputs

| File | Contents |
|---|---|
| `agentic_2026_probs.csv` | per-team % to reach R32/R16/QF/SF/Final/Win across runs |
| `agentic_2026_run<N>_matches.json` | every match: stage, score, pens, and the full event log — goals with scorer + chosen action + GK reaction, HT substitutions |
| `agentic_2026_meta.json` | model, runs, seed, elapsed time, LLM call/fallback accounting |
| `agentic_2026_demo.json` | demo-match summary (when using `--demo`) |

Pull them back with `runpodctl send/receive`, the web file browser, or scp.

## 7. Troubleshooting

* **"Cannot reach … start Ollama first"** — `ollama serve` isn't running or
  the model isn't pulled; check `/workspace/ollama.log` and `ollama list`.
* **High parse-fallback % in the accounting line** — the model is ignoring the
  JSON instruction; use an *instruct* qwen tag (not a base model). qwen3's
  `<think>` blocks are stripped automatically.
* **Slow** — drop to `qwen2.5:7b-instruct`, add `--no-keeper-agents`, and keep
  `OLLAMA_NUM_PARALLEL=4`.
* **Out of VRAM** — you're on a smaller GPU; use the 7b model or a q4 tag.
* **Aborted after 5 consecutive LLM failures** — that's the circuit breaker
  refusing to silently degrade into a non-agentic sim; fix the endpoint and
  rerun (runs are seeded, so a rerun reproduces).
