"""Per-match market odds: de-vig, derive, and blend against the simulation.

The repo already de-vigs outright title odds and blends them with the model at
the tournament level. This is the same idea for a single game. Give it the 1X2
prices for a fixture and it strips the bookmaker margin to clean home/draw/away
probabilities; if you do not have prices it derives an implied line from the
rating gap instead. Either way it can be set beside the simulation's own
distribution and blended geometrically, so you can see the model, the market,
and a combined number for the same match.

The market is kept as a layer on top of the simulation, never an input to it.
The physics should not be told the answer the bookmakers expect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from . import engine

Probs = Tuple[float, float, float]  # (home, draw, away)


def american_to_decimal(odds: float) -> float:
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def devig(home: float, draw: float, away: float, fmt: str = "decimal") -> Probs:
    """Strip the bookmaker margin from 1X2 prices to true probabilities."""
    if fmt == "american":
        home, draw, away = (american_to_decimal(home), american_to_decimal(draw),
                            american_to_decimal(away))
    raw = np.array([1.0 / home, 1.0 / draw, 1.0 / away])
    p = raw / raw.sum()
    return float(p[0]), float(p[1]), float(p[2])


def overround(home: float, draw: float, away: float, fmt: str = "decimal") -> float:
    """The bookmaker's margin (how far the raw prices sum past 100%)."""
    if fmt == "american":
        home, draw, away = (american_to_decimal(home), american_to_decimal(draw),
                            american_to_decimal(away))
    return float(1.0 / home + 1.0 / draw + 1.0 / away - 1.0)


def implied_from_ratings(rating_home: float, rating_away: float,
                         kmax: int = 10) -> Probs:
    """Derive a 1X2 line from the rating gap, via the calibrated goal model.

    This is the fallback when no real prices are to hand. It uses the same
    Elo-to-Poisson map the engine simulates with, so the derived market is a
    fair, margin-free reference rather than an outside opinion.
    """
    lam_h, lam_a = engine.lambdas_for((rating_home - rating_away) * engine.GAP_SCALE)
    ks = np.arange(0, kmax + 1)
    ph = np.exp(-lam_h) * lam_h ** ks / np.array([math.factorial(k) for k in ks])
    pa = np.exp(-lam_a) * lam_a ** ks / np.array([math.factorial(k) for k in ks])
    home = draw = away = 0.0
    for i in ks:
        for j in ks:
            p = ph[i] * pa[j]
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
    total = home + draw + away
    return home / total, draw / total, away / total


def blend(model: Probs, market: Probs, w: float = 0.5) -> Probs:
    """Geometric blend of two 1X2 distributions; w is the model's weight."""
    m = np.array(model, dtype=float)
    k = np.array(market, dtype=float)
    g = (m ** w) * (k ** (1.0 - w))
    g = g / g.sum()
    return float(g[0]), float(g[1]), float(g[2])


@dataclass
class MarketView:
    home: str
    away: str
    model: Probs
    market: Probs
    blended: Probs
    market_source: str   # "prices" or "derived"
    overround: float = 0.0

    def _edge(self) -> Probs:
        return tuple(self.model[i] - self.market[i] for i in range(3))  # type: ignore

    def summary(self) -> str:
        labels = (f"{self.home} win", "draw", f"{self.away} win")
        edge = self._edge()
        lines = [f"Model vs market ({self.market_source})"]
        head = f"  {'outcome':<18}{'model':>8}{'market':>8}{'blend':>8}{'edge':>8}"
        lines.append(head)
        for i, lab in enumerate(labels):
            lines.append(f"  {lab:<18}{self.model[i]*100:7.1f}%{self.market[i]*100:7.1f}%"
                         f"{self.blended[i]*100:7.1f}%{edge[i]*100:+7.1f}")
        if self.overround:
            lines.append(f"  (book overround {self.overround*100:.1f}%)")
        return "\n".join(lines)


def view(home: str, away: str, model: Probs, market: Probs,
         source: str, w: float = 0.5, book_over: float = 0.0) -> MarketView:
    return MarketView(home=home, away=away, model=model, market=market,
                      blended=blend(model, market, w), market_source=source,
                      overround=book_over)
