"""Ablation study: is the agentic engine's flat champion field tactics or noise?

Three arms, identical engine/bracket/squads (seed 2026):
    LLM       qwen2.5:14b agents      (100 pooled runs, from the pod batches)
    RANDOM    uniform random decisions, same action space (ablation/random_*)
    DEFAULTS  frozen default decisions, zero variance     (ablation/defaults_*)

Outputs ablation_results.md + viz_ablation.png (Apple design language).
"""

import glob
import json
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial"]
INK, SUB, LINE, CARD = "#1D1D1F", "#86868B", "#D2D2D7", "#F5F5F7"
GOLD, BLUE = "#B8860B", "#0071E3"


def champions_from_runs(pattern):
    champs, goals, matches = [], 0, 0
    for f in sorted(glob.glob(pattern)):
        log = json.load(open(f))
        champs.append(next(m["winner"] for m in log if m["stage"] == "FINAL"))
        goals += sum(m["hg"] + m["ag"] for m in log)
        matches += len(log)
    return Counter(champs), goals / matches


def dist_stats(counts: Counter):
    n = sum(counts.values())
    ps = np.array([c / n for c in counts.values()])
    entropy = -float((ps * np.log2(ps)).sum())
    top = counts.most_common()
    return {
        "runs": n,
        "distinct": len(counts),
        "entropy_bits": round(entropy, 2),
        "top1": f"{top[0][0]} {100*top[0][1]/n:.0f}%",
        "top5_share_pct": round(100 * sum(c for _, c in top[:5]) / n, 1),
    }


def tv_distance(c1: Counter, c2: Counter):
    n1, n2 = sum(c1.values()), sum(c2.values())
    teams = set(c1) | set(c2)
    return 0.5 * sum(abs(c1.get(t, 0) / n1 - c2.get(t, 0) / n2) for t in teams)


def main():
    # --- champion distributions ---------------------------------------------
    llm = Counter(dict(pd.read_csv("results/agentic_champions_pooled.csv")
                       .set_index("team")["titles"]))
    rnd, rnd_gpm = champions_from_runs("ablation/random_run*_matches.json")
    dfl, dfl_gpm = champions_from_runs("ablation/defaults_run*_matches.json")

    arms = {"LLM agents": llm, "Random agents": rnd, "Default decisions": dfl}
    summary = pd.DataFrame({k: dist_stats(v) for k, v in arms.items()}).T
    print(summary.to_string())
    print(f"\ngoals/match  random={rnd_gpm:.2f}  defaults={dfl_gpm:.2f}  llm=2.69")
    print(f"TV distance  LLM vs random   = {tv_distance(llm, rnd):.3f}")
    print(f"TV distance  LLM vs defaults = {tv_distance(llm, dfl):.3f}")
    print(f"TV distance  random vs defaults = {tv_distance(rnd, dfl):.3f}")

    # --- per-team title odds, rank agreement ---------------------------------
    teams = sorted(set(llm) | set(rnd) | set(dfl))
    n_llm, n_rnd, n_dfl = (sum(c.values()) for c in (llm, rnd, dfl))
    odds = pd.DataFrame({
        "team": teams,
        "LLM": [100 * llm.get(t, 0) / n_llm for t in teams],
        "Random": [100 * rnd.get(t, 0) / n_rnd for t in teams],
        "Defaults": [100 * dfl.get(t, 0) / n_dfl for t in teams],
    })
    rho_lr = sps.spearmanr(odds["LLM"], odds["Random"])
    rho_ld = sps.spearmanr(odds["LLM"], odds["Defaults"])
    print(f"\nSpearman rho  LLM vs random   = {rho_lr.statistic:.2f} (p={rho_lr.pvalue:.3f})")
    print(f"Spearman rho  LLM vs defaults = {rho_ld.statistic:.2f} (p={rho_ld.pvalue:.3f})")

    # chi-square: can we reject "random and LLM draw champions from same dist"?
    both = sorted(set(llm) | set(rnd))
    tbl = np.array([[llm.get(t, 0) for t in both], [rnd.get(t, 0) for t in both]])
    chi = sps.chi2_contingency(tbl)
    print(f"chi2 LLM vs random: p={chi.pvalue:.3f}")

    odds.sort_values("LLM", ascending=False).to_csv("ablation_odds.csv", index=False)

    # --- viz ------------------------------------------------------------------
    top10 = odds.sort_values("LLM", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ys = np.arange(len(top10))
    for y, (_, r) in zip(ys, top10.iterrows()):
        lo = min(r["LLM"], r["Random"], r["Defaults"])
        hi = max(r["LLM"], r["Random"], r["Defaults"])
        ax.plot([lo, hi], [y, y], color=LINE, lw=2.0, zorder=1)
        ax.scatter(r["LLM"], y, s=110, color=INK, zorder=3)
        ax.scatter(r["Random"], y, s=110, color=GOLD, zorder=3)
        ax.scatter(r["Defaults"], y, s=110, color=BLUE, zorder=3)
        ax.text(-0.6, y, r["team"], ha="right", va="center", fontsize=11.5, color=INK)
    for col, lab in ((INK, "LLM agents"), (GOLD, "Random agents"),
                     (BLUE, "Default decisions")):
        ax.scatter([], [], s=110, color=col, label=lab)
    ax.legend(frameon=False, fontsize=10.5, loc="lower right")
    ax.set_xlim(-8, max(odds[["LLM", "Random", "Defaults"]].max()) + 3)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels(["0%", "5%", "10%", "15%"], fontsize=10, color=SUB)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color="#F0F0F2", lw=1)
    ax.set_axisbelow(True)
    ax.set_title("Title odds with the agents' brains removed — 100 tournaments per arm",
                 fontsize=13.5, color=INK, pad=18, loc="left", fontweight="bold")
    fig.savefig("viz_ablation.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote viz_ablation.png, ablation_odds.csv")

    # --- markdown summary -------------------------------------------------------
    with open("ablation_results.md", "w") as fh:
        fh.write("# Ablation study — random vs LLM agents\n\n")
        fh.write(summary.to_markdown() + "\n\n")
        fh.write(f"- goals/match: random {rnd_gpm:.2f}, defaults {dfl_gpm:.2f}, "
                 f"LLM 2.69 (real World Cups 2.64)\n")
        fh.write(f"- TV distance LLM↔random {tv_distance(llm, rnd):.3f}, "
                 f"LLM↔defaults {tv_distance(llm, dfl):.3f}, "
                 f"random↔defaults {tv_distance(rnd, dfl):.3f}\n")
        fh.write(f"- Spearman rho (title odds): LLM↔random {rho_lr.statistic:.2f} "
                 f"(p={rho_lr.pvalue:.3f}); LLM↔defaults {rho_ld.statistic:.2f} "
                 f"(p={rho_ld.pvalue:.3f})\n")
        fh.write(f"- chi-square LLM vs random champion tables: p={chi.pvalue:.3f}\n")
    print("wrote ablation_results.md")


if __name__ == "__main__":
    main()
