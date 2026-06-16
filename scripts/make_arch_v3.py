"""Render the v3 dynamic-engine architecture diagram to figures/arch_v3.png."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "arch_v3.png")

INK = "#1d1d1f"
SUB = "#6e6e73"
LINE = "#c7c7cc"
BLUE = "#0a84ff"
GREEN = "#34c759"
AMBER = "#ff9f0a"
CARD = "#f5f5f7"


def box(ax, x, y, w, h, title, lines, edge=LINE, title_color=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=edge, facecolor="white"))
    ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=title_color)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.46 - i * 0.235, ln, ha="center", va="top",
                fontsize=8.2, color=SUB)


def arrow(ax, x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.3, color=color, shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(11.5, 6.6))
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 6.6)
ax.axis("off")

ax.text(0.2, 6.35, "Dynamic tournament-state engine", fontsize=15,
        fontweight="bold", color=INK)
ax.text(0.2, 6.02, "results in, conditioned forecast out", fontsize=10,
        color=SUB)

box(ax, 0.2, 4.2, 2.2, 1.3, "Real results",
    ["data/results_2026.csv", "replayed in matchday", "order"], edge=BLUE)

box(ax, 2.9, 4.2, 2.6, 1.3, "Bayesian state",
    ["Normal-Normal update", "rating mean + shrinking std", "form, fatigue, morale"],
    edge=BLUE, title_color=BLUE)

box(ax, 0.2, 2.2, 2.2, 1.45, "Pre-tournament priors",
    ["Elo per team", "(base engine)", "= posterior mean"], edge=LINE)

# Per-match modifier stack.
box(ax, 6.0, 4.55, 2.5, 0.95, "Style matchup",
    ["possession vs block,", "press, transition  +/-15%"], edge=AMBER)
box(ax, 6.0, 3.45, 2.5, 0.95, "Venue + travel",
    ["altitude, heat, humidity,", "cross-country travel"], edge=AMBER)
box(ax, 6.0, 2.35, 2.5, 0.95, "Matchday intent",
    ["need-win / avoid-loss /", "qualified  by group state"], edge=AMBER)

box(ax, 9.0, 3.3, 2.2, 1.45, "Monte Carlo",
    ["calibrated Poisson", "sample posteriors,", "group + 2026 bracket"],
    edge=GREEN, title_color=GREEN)

box(ax, 9.0, 1.1, 2.2, 1.5, "Outputs",
    ["champion / round odds", "path difficulty", "diagnostics, calibration"],
    edge=INK)

box(ax, 2.9, 2.35, 2.6, 1.2, "Match model",
    ["Elo gap -> goal means", "GAP_SCALE upset tune"], edge=LINE)

arrow(ax, 2.4, 4.85, 2.9, 4.85, BLUE)        # results -> state
arrow(ax, 1.3, 3.65, 1.3, 4.2, BLUE)         # priors -> results
arrow(ax, 4.2, 4.2, 4.2, 3.55, INK)          # state -> match model
arrow(ax, 5.5, 3.2, 6.0, 5.0, INK)           # match model -> style
arrow(ax, 5.5, 3.0, 6.0, 3.9, INK)           # match model -> venue
arrow(ax, 5.5, 2.8, 6.0, 2.8, INK)           # match model -> intent
arrow(ax, 8.5, 5.0, 9.0, 4.45, AMBER)        # style -> MC
arrow(ax, 8.5, 3.9, 9.0, 4.05, AMBER)        # venue -> MC
arrow(ax, 8.5, 2.8, 9.0, 3.65, AMBER)        # intent -> MC
arrow(ax, 10.1, 3.3, 10.1, 2.6, GREEN)       # MC -> outputs

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", os.path.relpath(OUT, ROOT))
