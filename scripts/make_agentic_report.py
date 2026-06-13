"""Assemble agentic_report.html — the RunPod/Ollama agentic tournament report.

Inputs (produced by agentic_wc2026.py on the pod, pulled back locally):
    agentic_2026_run1_matches.json  full match log: scores, scorers, decisions
    agentic_2026_probs.csv          stage-reach table (0/100 for a single run)
    agentic_2026_meta.json          provenance (optional — absent if not pulled)

Same self-contained style as make_report.py: matplotlib PNGs embedded base64.
"""

import base64
import io
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ACCENT = "#1a6e3c"          # pitch green
ACCENT2 = "#c8a217"         # trophy gold

MATCHES = "agentic_2026_run1_matches.json"
META = "agentic_2026_meta.json"
GK_SCHEMA = {"hold_line", "rush_out", "narrow_angle"}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img(b64: str, alt: str) -> str:
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%">'


def html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="tbl", justify="left",
                      escape=True)


def score_str(m: dict) -> str:
    s = f"{m['home']} {m['hg']}–{m['ag']} {m['away']}"
    if m["pens"]:
        s += f" (pens {m['pens'][0]}–{m['pens'][1]})"
    return s


def parse_goals(log):
    """[(scorer, team, action, gk_action, stage)] for every goal event."""
    out = []
    for m in log:
        for _, team, txt in m["events"]:
            g = re.match(r"GOAL (.+?) \((\w+), GK (.+)\)$", txt)
            if g:
                out.append((g.group(1), team, g.group(2), g.group(3), m["stage"]))
    return out


def group_tables(log):
    """Recompute final group standings from the group-stage scorelines."""
    pts = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # grp -> team -> [pts, gf, ga]
    for m in (m for m in log if m["stage"] == "GROUP"):
        h, a = pts[m["group"]][m["home"]], pts[m["group"]][m["away"]]
        h[1] += m["hg"]; h[2] += m["ag"]; a[1] += m["ag"]; a[2] += m["hg"]
        if m["hg"] > m["ag"]:
            h[0] += 3
        elif m["hg"] < m["ag"]:
            a[0] += 3
        else:
            h[0] += 1; a[0] += 1
    out = {}
    for grp in sorted(pts):
        ranked = sorted(pts[grp].items(),
                        key=lambda kv: (kv[1][0], kv[1][1] - kv[1][2], kv[1][1]),
                        reverse=True)
        out[grp] = [(t, v[0], v[1], v[2]) for t, v in ranked]
    return out


def chart_scorers(goals) -> str:
    top = Counter(g[0] for g in goals).most_common(12)[::-1]
    names, n = [t[0] for t in top], [t[1] for t in top]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, n, color=ACCENT)
    bars[-1].set_color(ACCENT2)
    for b, v in zip(bars, n):
        ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2, str(v),
                va="center", fontsize=9)
    ax.set_title("Golden Boot race — goals in this agentic tournament")
    ax.set_xlabel("goals")
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_b64(fig)


def chart_goals_per_match(log) -> str:
    totals = Counter(m["hg"] + m["ag"] for m in log)
    xs = sorted(totals)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(xs, [totals[x] for x in xs], color=ACCENT)
    ax.set_xlabel("total goals in match")
    ax.set_ylabel("matches")
    avg = sum(m["hg"] + m["ag"] for m in log) / len(log)
    ax.set_title(f"Goals per match (mean {avg:.2f})")
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_b64(fig)


def chart_decisions(goals) -> str:
    acts = Counter(g[2] for g in goals)
    gks = Counter(g[3] if g[3] in GK_SCHEMA else "free-form" for g in goals)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].pie(acts.values(), labels=acts.keys(), autopct="%d%%",
                colors=[ACCENT, ACCENT2, "#999999"])
    axes[0].set_title("Scorer's chosen action (on goals)")
    axes[1].pie(gks.values(), labels=gks.keys(), autopct="%d%%",
                colors=["#5a8f6e", "#c8a217", "#999999", "#cccccc"])
    axes[1].set_title("Goalkeeper agent's choice (on goals)")
    return fig_to_b64(fig)


def main() -> None:
    log = json.load(open(MATCHES))
    meta = json.load(open(META)) if os.path.exists(META) else {}
    goals = parse_goals(log)
    subs = sum(1 for m in log for ev in m["events"] if ev[2].startswith("SUB"))
    champion_match = next(m for m in log if m["stage"] == "FINAL")
    champ = champion_match["winner"]

    # Champion's road to the title
    road = [m for m in log if champ in (m["home"], m["away"])]
    road_rows = [{"Stage": m["stage"] + (f" (Grp {m['group']})" if m["group"] else ""),
                  "Match": score_str(m),
                  "Goals": "; ".join(
                      re.match(r"GOAL (.+?) \(", e[2]).group(1) + f" {e[0]}'"
                      for e in m["events"]
                      if e[1] == champ and e[2].startswith("GOAL")) or "—"}
                 for m in road]

    ko_html = ""
    for stage, label in [("R32", "Round of 32"), ("R16", "Round of 16"),
                         ("QF", "Quarter-finals"), ("SF", "Semi-finals"),
                         ("FINAL", "Final")]:
        ms = [m for m in log if m["stage"] == stage]
        if not ms:
            continue
        rows = [{"Match": score_str(m), "Winner": m["winner"]} for m in ms]
        ko_html += f"<h3>{label}</h3>{html_table(pd.DataFrame(rows))}"

    grp_html = ""
    for grp, ranked in group_tables(log).items():
        rows = [{"#": i + 1, "Team": t, "Pts": p, "GF": gf, "GA": ga}
                for i, (t, p, gf, ga) in enumerate(ranked)]
        grp_html += (f'<div class="grp"><b>Group {grp}</b>'
                     f"{html_table(pd.DataFrame(rows))}</div>")

    gk_freeform = sorted({g[3] for g in goals} - GK_SCHEMA)
    n_group = sum(1 for m in log if m["stage"] == "GROUP")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>2026 World Cup — Agentic Tournament Report</title>
<style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem auto;
        max-width: 980px; color: #1c2520; padding: 0 1rem; }}
 h1 {{ border-bottom: 4px solid {ACCENT}; padding-bottom: .4rem; }}
 h2 {{ color: {ACCENT}; margin-top: 2.2rem; }}
 h3 {{ margin-bottom: .2rem; }}
 .muted {{ color: #5a6862; font-size: .92rem; }}
 .tbl {{ border-collapse: collapse; font-size: .85rem; margin: .6rem 0; }}
 .tbl th {{ background: {ACCENT}; color: #fff; padding: .35rem .6rem; text-align: left; }}
 .tbl td {{ padding: .3rem .6rem; border-bottom: 1px solid #e2e8e4; }}
 .scoreline {{ font-size: 1.6rem; font-weight: 700; margin: .8rem 0; }}
 .badge {{ display: inline-block; background: #eef3f0; border-radius: 6px;
          padding: .25rem .7rem; margin: .15rem; font-size: .85rem; }}
 .grp {{ display: inline-block; vertical-align: top; margin-right: 1.2rem; }}
</style></head><body>
<h1>🤖 2026 World Cup — Agentic Tournament (RunPod · Ollama)</h1>
<p class="muted">Generated {date.today().isoformat()} · every starter an LLM player-agent
(shoot / pass / dribble, GK hold / rush / narrow) · LLM managers set tactics pre-match and
make real-bench substitutions at half time · real FIFA squads (players_2026_real.csv).</p>
<p>
 <span class="badge">Matches: <b>{len(log)}</b> ({n_group} group + knockout)</span>
 <span class="badge">Goals: <b>{len(goals)}</b></span>
 <span class="badge">HT substitutions: <b>{subs}</b></span>
 <span class="badge">Model: <b>{meta.get('model', 'qwen (Ollama)')}</b></span>
 <span class="badge">{meta.get('llm_report', 'single tournament run')}</span>
</p>

<h2>1 · Champion</h2>
<p class="scoreline">🏆 {champ} &nbsp;<span class="muted" style="font-size:1rem">
won the final: {score_str(champion_match)}</span></p>
{html_table(pd.DataFrame(road_rows))}

<h2>2 · Knockout bracket</h2>
{ko_html}

<h2>3 · Group stage standings</h2>
{grp_html}

<h2>4 · Golden Boot</h2>
{img(chart_scorers(goals), "top scorers")}
{img(chart_goals_per_match(log), "goals per match")}

<h2>5 · What the agents decided</h2>
{img(chart_decisions(goals), "agent decisions")}
<p class="muted">Free-form GK strings (model improvised outside the schema, treated as
neutral by the engine): {len(gk_freeform)} occurrence(s){' — e.g. “' + gk_freeform[0] + '”'
    if gk_freeform else ''}.</p>

<h2>6 · Read this with care</h2>
<ul class="muted">
 <li>This is <b>one</b> tournament, not a probability estimate — the companion
     <code>agentic_2026_probs.csv</code> is 0/100 columns until you aggregate
     multiple <code>--runs</code>.</li>
 <li>Scoring runs hot ({len(goals) / len(log):.2f} goals/match vs ~2.6 at real World Cups)
     and scorers concentrate: the engine's chance-picker squares ratings and the
     player agents overwhelmingly chose <b>pass</b>
     ({Counter(g[2] for g in goals)['pass']}/{len(goals)} of goals), which carries a
     finishing bonus when the better-rated teammate receives it.</li>
 <li>Real played 2026 results were pinned, not simulated.</li>
</ul>
</body></html>"""

    with open("agentic_report.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote agentic_report.html ({champ} champion, {len(goals)} goals, "
          f"{len(log)} matches)")


if __name__ == "__main__":
    main()
