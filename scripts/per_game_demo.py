"""Per-game simulator demo: a single fixture, with and without a key player.

Shows the player-to-team idea in action. Pick a fixture, simulate it through
the real match physics, then rule a player out and simulate again. The squad
reshapes around the absence and the outcome distribution moves with it.

    python3 scripts/per_game_demo.py
    python3 scripts/per_game_demo.py --home France --away Mexico --out-home Mbappe
    python3 scripts/per_game_demo.py --home Spain --away Germany --knockout
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldcup_sim import squads
from worldcup_sim.per_game import PerGameSimulator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="France")
    ap.add_argument("--away", default="Mexico")
    ap.add_argument("--out-home", default="Mbappe",
                    help="substring of a home player to rule out for the B run")
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--knockout", action="store_true")
    args = ap.parse_args()

    sim = PerGameSimulator(use_manager=True, seed=2026)

    print(f"Player-aggregated ratings:")
    print(f"  {args.home}: {squads.player_elo(args.home):.0f}    "
          f"{args.away}: {squads.player_elo(args.away):.0f}\n")

    # Common random numbers so the only difference between the two runs is the
    # player who sits out, not the dice.
    crn = 4242
    full = sim.fixture(args.home, args.away, runs=args.runs,
                       knockout=args.knockout, common_seed=crn)
    print("Full strength")
    print(full.summary())

    if args.out_home:
        impact = squads.star_impact(args.home, [args.out_home])
        print(f"\nRuling out: {', '.join(impact['ruled_out']) or args.out_home} "
              f"({args.home} player-Elo {impact['elo_full']:.0f} -> "
              f"{impact['elo_without']:.0f}, {impact['elo_drop']:+.0f})")
        without = sim.fixture(args.home, args.away, runs=args.runs,
                              knockout=args.knockout, out_a=[args.out_home],
                              common_seed=crn)
        print(without.summary())
        print(f"\n{args.home} win probability: {full.a_win*100:.1f}% -> "
              f"{without.a_win*100:.1f}%  "
              f"({(without.a_win-full.a_win)*100:+.1f} points)")
        print("\nNote: one player off a squad this deep is within Monte Carlo "
              "noise at the match level. The player-level signal is the "
              f"{impact['elo_drop']:+.0f} Elo aggregate above, and it compounds "
              "over a tournament. Take out the whole first XI and the swing is "
              "unmistakable:")
        inv = squads.inventory()
        top11 = [p.name for p in sorted(inv.teams[args.home].squad,
                 key=lambda p: p.effective_rating(), reverse=True)[:11]]
        depleted = sim.fixture(args.home, args.away, runs=args.runs,
                               knockout=args.knockout, out_a=top11, common_seed=crn)
        print(f"  {args.home} first XI out: win {full.a_win*100:.1f}% -> "
              f"{depleted.a_win*100:.1f}%, expected goals "
              f"{full.mean_goals_a:.2f} -> {depleted.mean_goals_a:.2f}")


if __name__ == "__main__":
    main()
