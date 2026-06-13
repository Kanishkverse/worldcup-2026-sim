"""Assemble report.html — visualized 2026 World Cup prediction report.

Inputs (produced by run_2026.py / run_backtest.py / generate_players_2026.py):
    out_2026_probs.csv   Monte Carlo stage probabilities, 48 teams
    out_2026_meta.json   run provenance (iterations, LLM backend/model)
    out_backtest.csv     2010-2022 validation vs FIFA baseline
    out_demo_match.json  single simulated match with live events
    players_2026.csv     per-player performance dataset

Charts are matplotlib PNGs embedded as base64 — the HTML is fully
self-contained and opens offline.
"""

import base64
import io
import json
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ACCENT = "#1a6e3c"          # pitch green
ACCENT2 = "#c8a217"         # trophy gold


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img(b64: str, alt: str) -> str:
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%">'


def chart_champions(probs: pd.DataFrame, n_iters: int) -> str:
    top = probs.head(15).iloc[::-1]
    p = top["Win Trophy %"] / 100.0
    ci = 196.0 * (p * (1 - p) / n_iters) ** 0.5    # 95% binomial CI in pp
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top["Country"], top["Win Trophy %"],
                   xerr=ci, color=ACCENT, error_kw={"ecolor": "#444", "capsize": 3})
    bars[-1].set_color(ACCENT2)   # favourite
    for b, v, e in zip(bars, top["Win Trophy %"], ci):
        ax.text(b.get_width() + e + 0.3, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9)
    ax.set_title(f"Win the 2026 World Cup — probability (top 15, 95% CI over {n_iters} runs)")
    ax.set_xlabel("% of simulated tournaments won")
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_b64(fig)


def chart_stages(probs: pd.DataFrame) -> str:
    top = probs.head(10)
    stages = ["Reach R16 %", "Reach QF %", "Reach SF %", "Reach Final %", "Win Trophy %"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(top))
    colors = plt.cm.Greens([0.35, 0.5, 0.65, 0.8, 0.95])
    width = 0.16
    for i, (st, c) in enumerate(zip(stages, colors)):
        ax.bar([xx + i * width for xx in x], top[st], width=width,
               label=st.replace(" %", ""), color=c)
    ax.set_xticks([xx + 2 * width for xx in x])
    ax.set_xticklabels(top["Country"], rotation=30, ha="right")
    ax.set_ylabel("%")
    ax.set_title("Stage-by-stage survival — top 10 teams")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_b64(fig)


def chart_backtest(bt: pd.DataFrame) -> str:
    yrs = bt[bt["year"] != "ALL"]
    piv_prob = yrs.pivot(index="year", columns="model", values="champ_prob")
    piv_rank = yrs.pivot(index="year", columns="model", values="champ_rank")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    piv_prob.plot.bar(ax=axes[0], color=[ACCENT2, "#999999"], rot=0)
    axes[0].set_title("P(actual champion) per year\n(higher = better)")
    axes[0].set_ylabel("probability")
    piv_rank.plot.bar(ax=axes[1], color=[ACCENT2, "#999999"], rot=0)
    axes[1].set_title("Rank given to actual champion\n(lower = better)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    return fig_to_b64(fig)


def html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="tbl", justify="left",
                      float_format=lambda v: f"{v:.2f}")


def effective_rating(r) -> float:
    sharp = 0.40 * r["form"] + 0.35 * r["club_performance"] + 0.25 * r["club_minutes"]
    return r["base_rating"] * (0.70 + 0.30 * sharp)


def main() -> None:
    probs = pd.read_csv("results/out_2026_probs.csv")
    meta = json.load(open("out_2026_meta.json"))
    bt = pd.read_csv("out_backtest.csv")
    demo = json.load(open("out_demo_match.json"))
    players = pd.read_csv("players_2026.csv")
    players["effective_rating"] = players.apply(effective_rating, axis=1)

    # -- demo match block ----------------------------------------------------
    goals = {(m, t) for m, t in demo["timeline"]}
    ev_rows = "".join(
        f"<tr class='{'red' if 'RED' in e.upper() else ('goal' if 'GOAL' in e.upper() else '')}'>"
        f"<td>{m}'</td><td>{t}</td><td>{e}</td></tr>"
        for m, t, e in sorted(demo["events"] + [[m, t, "GOAL"] for m, t in goals]))
    demo_html = f"""
    <div class="scoreline">{demo['home']} {demo['home_goals']} – {demo['away_goals']} {demo['away']}</div>
    <p class="muted">One <b>simulated</b> rendition of the real June 11 opener (actual
       result: Mexico 2–0 South Africa — that real score is pinned inside every Monte
       Carlo tournament; this replay just showcases the engine). Halftime tactics by
       {demo['backend']}. Red cards permanently cut the offender's team cohesion;
       fatigued players are auto-substituted and the player-graph layer recomputes
       match rates instantly.</p>
    <table class="tbl"><tr><th>Min</th><th>Team</th><th>Event</th></tr>{ev_rows}</table>
    """

    # -- per-player block: top squads of the 4 favourites ---------------------
    fav4 = probs.head(4)["Country"].tolist()
    blocks = []
    for team in fav4:
        sq = (players[players["team"] == team]
              .sort_values("effective_rating", ascending=False).head(8))
        cols = ["player", "position", "age", "club", "base_rating", "form",
                "club_performance", "club_minutes", "market_value_m", "effective_rating"]
        blocks.append(f"<h4>{team}</h4>" + html_table(sq[cols].round(3)))
    players_html = "\n".join(blocks)

    bt_all = bt[bt["year"] == "ALL"]
    feat_row = bt_all[bt_all["model"] == "features"].iloc[0]
    base_row = bt_all[bt_all["model"] == "fifa_baseline"].iloc[0]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>2026 World Cup — Prediction Report</title>
<style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem auto;
        max-width: 980px; color: #1c2520; padding: 0 1rem; }}
 h1 {{ border-bottom: 4px solid {ACCENT}; padding-bottom: .4rem; }}
 h2 {{ color: {ACCENT}; margin-top: 2.2rem; }}
 .muted {{ color: #5a6862; font-size: .92rem; }}
 .tbl {{ border-collapse: collapse; font-size: .85rem; margin: .6rem 0; }}
 .tbl th {{ background: {ACCENT}; color: #fff; padding: .35rem .6rem; text-align: left; }}
 .tbl td {{ padding: .3rem .6rem; border-bottom: 1px solid #e2e8e4; }}
 .tbl tr.red td {{ background: #fdebeb; color: #a31515; font-weight: 600; }}
 .tbl tr.goal td {{ background: #eef7f0; font-weight: 600; }}
 .scoreline {{ font-size: 1.6rem; font-weight: 700; margin: .8rem 0; }}
 .badge {{ display: inline-block; background: #eef3f0; border-radius: 6px;
          padding: .25rem .7rem; margin: .15rem; font-size: .85rem; }}
</style></head><body>
<h1>🏆 2026 FIFA World Cup — Prediction Report</h1>
<p class="muted">Generated {date.today().isoformat()} · Monte Carlo over the official
48-team draw · graph spatio-temporal match engine (minute-by-minute Poisson, red cards,
fatigue substitutions) · halftime tactical agent.</p>
<p>
 <span class="badge">Iterations: <b>{meta['iterations']}</b></span>
 <span class="badge">LLM tactics: <b>{meta['model']}</b> ({meta['backend']})</span>
 <span class="badge">Features: live Elo (Dec 2025) + Transfermarkt squad values + SPI + FIFA pts</span>
 <span class="badge">Per-player dataset: players_2026.csv (1,104 players)</span>
</p>

<h2>1 · Who wins?</h2>
{img(chart_champions(probs, int(meta['iterations'])), 'champion probabilities')}
<h2>2 · Stage-by-stage survival</h2>
{img(chart_stages(probs), 'stage probabilities')}
<details><summary>Full 48-team probability table</summary>{html_table(probs)}</details>

<h2>3 · Does the model actually work? Backtest 2010–2022</h2>
<p>The same engine replayed four completed World Cups (100 tournaments per model per
year). The multi-feature model is compared with a FIFA-ranking-only baseline against
what really happened.</p>
{img(chart_backtest(bt), 'backtest validation')}
{html_table(bt)}
<p>Across all four tournaments the feature model gives the real champion
<b>{feat_row['champ_prob']:.1%}</b> probability on average (FIFA baseline:
{base_row['champ_prob']:.1%}) and ranks the eventual winner
<b>#{feat_row['champ_rank']:.1f}</b> on average (baseline: #{base_row['champ_rank']:.1f}).
Semifinalist recall: <b>{feat_row['sf_recall@4']:.2f}/4</b> vs {base_row['sf_recall@4']:.2f}/4,
with <b>{feat_row['sf_mass']:.0%}</b> of semifinal probability mass on real semifinalists
(baseline: {base_row['sf_mass']:.0%}).</p>

<h2>4 · Anatomy of one simulated match</h2>
{demo_html}

<h2>5 · Per-player performance — squads of the four favourites</h2>
<p class="muted">Player rows come from <code>players_2026.csv</code> — ability, form,
club performance (derived from the club-SPI table), club minutes and market value per
player. Names are positional placeholders; edit the CSV with real rosters and rerun —
the simulator reads it directly (WC_PLAYERS_CSV).</p>
{players_html}

<h2>6 · Method &amp; provenance</h2>
<ul>
 <li><b>Team strength</b> = 0.45·Elo + 0.30·log(squad value) + 0.15·SPI + 0.10·FIFA pts,
     min-max normalised per field. Elo refreshed from <code>eloratings.csv</code>
     (through Dec 2025, 47/48 teams), squad values from <code>mv.csv</code> (40/48;
     snapshot fills gaps).</li>
 <li><b>Players</b>: 23 per squad; effective rating = base × (0.70 + 0.30·sharpness),
     sharpness = 0.40·form + 0.35·club performance + 0.25·club minutes; injured players
     contribute nothing and trigger bench promotion.</li>
 <li><b>Match engine</b>: 11-node GNN cohesion → minute-by-minute attenuated Poisson;
     stochastic red cards compound a permanent cohesion penalty; fatigue crossings force
     substitutions and the graph is re-run.</li>
 <li><b>Halftime tactics</b>: {meta['model']} via the OpenAI protocol decides
     attack/defense multipliers per team with a JSON contract; deterministic heuristic
     as a circuit-broken fallback. The 2010–2022 backtest ran with the heuristic agent
     (≈100k LLM calls would otherwise be needed).</li>
 <li><b>Live conditioning</b>: matches already played at the real tournament
     (<code>results_2026.csv</code>) are pinned to their actual scoreline in every
     simulated tournament instead of being re-simulated — currently:
     Mexico 2–0 South Africa (June 11 opener). Append new rows as results land and
     rerun <code>run_2026.py</code> + <code>make_report.py</code>.</li>
 <li><b>Caveats</b>: group draws/brackets are the official 2026 structure; player rows
     are statistically derived placeholders, not real rosters; 100 iterations give
     champion probabilities ±2–3 pp.</li>
</ul>
</body></html>"""

    with open("report.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote report.html")


if __name__ == "__main__":
    main()
