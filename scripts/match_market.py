"""Simulate one fixture and set it against the market.

Runs the per-game physics simulator for two teams, optionally as they stand in
the tournament right now (--live), and puts the result next to the market: the
de-vigged 1X2 prices if you have them, or a line derived from the rating gap if
you do not. It prints the model, the market, a geometric blend, and the edge.

Odds come from, in order: --odds on the command line, a row in
data/match_odds_2026.csv, or the derived line.

    python3 scripts/match_market.py France Mexico
    python3 scripts/match_market.py France Mexico --live --runs 4000
    python3 scripts/match_market.py Spain Germany --odds 2.05,3.45,3.70
    python3 scripts/match_market.py Spain Germany --odds 110,260,250 --fmt american
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldcup_sim import live_state, market, squads
from worldcup_sim.per_game import PerGameSimulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ODDS_CSV = os.path.join(ROOT, "data", "match_odds_2026.csv")


def odds_from_csv(home: str, away: str):
    """Return (home_odds, draw_odds, away_odds, fmt) for a fixture, or None.

    Orientation matters for 1X2 prices, so a row stored the other way round is
    flipped to match the requested home/away.
    """
    if not os.path.exists(ODDS_CSV):
        return None
    with open(ODDS_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            h, a = r["home"].strip(), r["away"].strip()
            fmt = (r.get("fmt") or "decimal").strip()
            if h == home and a == away:
                return float(r["home_odds"]), float(r["draw_odds"]), float(r["away_odds"]), fmt
            if h == away and a == home:
                return float(r["away_odds"]), float(r["draw_odds"]), float(r["home_odds"]), fmt
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--runs", type=int, default=3000)
    ap.add_argument("--live", action="store_true",
                    help="play the teams at their current tournament rating")
    ap.add_argument("--knockout", action="store_true")
    ap.add_argument("--odds", help="home,draw,away prices for this fixture")
    ap.add_argument("--fmt", default="decimal", choices=["decimal", "american"])
    ap.add_argument("--weight", type=float, default=0.5,
                    help="model weight in the blend (0..1)")
    args = ap.parse_args()

    home, away = args.home, args.away

    # Ratings: current if --live, else pre-tournament player aggregate.
    priors = squads.player_elo_priors()
    if args.live:
        eff, _form = live_state.current_state()
        rating_h, rating_a = eff[home], eff[away]
        live_h, live_a = eff[home], eff[away]
        tag = "current (live)"
    else:
        rating_h, rating_a = priors[home], priors[away]
        live_h = live_a = None
        tag = "pre-tournament"

    print(f"{home} vs {away}   ratings: {rating_h:.0f} - {rating_a:.0f}  [{tag}]\n")

    sim = PerGameSimulator(use_manager=True, seed=2026)
    res = sim.fixture(home, away, runs=args.runs, knockout=args.knockout,
                      live_a=live_h, live_b=live_a)
    print(res.summary())
    model = (res.a_win, res.draw, res.b_win)

    # Market line: explicit odds, then CSV, then derive from the rating gap.
    book_over = 0.0
    if args.odds:
        h, d, a = (float(x) for x in args.odds.split(","))
        mkt = market.devig(h, d, a, fmt=args.fmt)
        book_over = market.overround(h, d, a, fmt=args.fmt)
        source = "prices (cli)"
    else:
        row = odds_from_csv(home, away)
        if row:
            h, d, a, fmt = row
            mkt = market.devig(h, d, a, fmt=fmt)
            book_over = market.overround(h, d, a, fmt=fmt)
            source = "prices (csv)"
        else:
            mkt = market.implied_from_ratings(rating_h, rating_a)
            source = "derived"

    v = market.view(home, away, model, mkt, source, w=args.weight,
                    book_over=book_over)
    print()
    print(v.summary())


if __name__ == "__main__":
    main()
