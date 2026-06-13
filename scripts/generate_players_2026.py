"""
Generate players_2026.csv — a per-player performance dataset for all 48
national teams in the 2026 World Cup field.

Each squad is 23 players (3 GK / 8 DEF / 7 MID / 5 FWD) whose metrics are
derived from REAL team-level sources rather than invented per player:

  * team strength tier      composite of live Elo (eloratings.csv), squad
                            market value (mv.csv), SPI snapshot and FIFA
                            points — the same composite the simulator uses
  * club / club_performance each player is assigned a club sampled from the
                            real club-SPI table (spi_global_rankings.csv),
                            biased so stronger nations field players at
                            stronger clubs; performance blends the club's
                            normalised SPI with the player's ability
  * age                     drawn around the squad's REAL average age
                            (national_teams.csv, Transfermarkt)
  * market_value_m          the squad's REAL total value split with a
                            heavy-tailed (Pareto) star distribution
  * form / club_minutes     drawn around the team's real form and average
                            club-minutes share

Player names are positional placeholders (e.g. "MEX FWD-1"); replace any
row with a real player's data — the simulator reads whatever is in the CSV
(via WC_PLAYERS_CSV in en.txt).

Run:  python3 generate_players_2026.py
"""

import csv
import math

import numpy as np
import pandas as pd

import worldcup_2026_sim as wc

SEED = 2026
PLAN = [("GK", 3), ("DEF", 8), ("MID", 7), ("FWD", 5)]
OUT = "players_2026.csv"


def _canon_team(name: str) -> str:
    name = str(name).replace("\xa0", " ").strip()
    extra = {"Cote d'Ivoire": "Côte d'Ivoire", "Curacao": "Curaçao",
             "Korea, South": "South Korea", "USA": "United States"}
    return extra.get(name, wc.FeatureStore.ALIASES.get(name, name))


def main() -> None:
    rng = np.random.default_rng(SEED)

    feats = wc.features_2026_live()                     # live Elo + MV applied
    strengths = wc.composite_strengths(feats)           # same composite as sim
    lo, hi = wc.STRENGTH_LO, wc.STRENGTH_HI

    # Real club-SPI table, best clubs first.
    clubs = pd.read_csv("spi_global_rankings.csv").sort_values("spi", ascending=False)
    club_names = clubs["name"].str.title().tolist()
    spi_norm = ((clubs["spi"] - clubs["spi"].min())
                / (clubs["spi"].max() - clubs["spi"].min())).tolist()

    # Real squad metadata (average age, team code) from Transfermarkt export.
    meta = {}
    for _, row in pd.read_csv("national_teams.csv").iterrows():
        meta[_canon_team(row["name"])] = row

    rows = []
    for team in sorted(feats):
        f = feats[team]
        s = strengths[team]
        s01 = (s - lo) / (hi - lo)                      # strength in [0,1]
        m = meta.get(team)
        avg_age = float(m["average_age"]) if m is not None and not pd.isna(m["average_age"]) else 27.0
        code = (str(m["country_code"])[:3].upper() if m is not None else team[:3].upper())
        total_mv = f.market_value_m if not math.isnan(f.market_value_m) else 0.0

        shares = np.sort(rng.pareto(2.5, 23) + 0.05)[::-1]
        shares /= shares.sum()

        idx = 0
        for pos, count in PLAN:
            for k in range(count):
                idx += 1
                base = float(np.clip(s + rng.normal(0, 0.07), 0.20, 0.99))
                # Stronger nations -> players concentrated at higher-SPI clubs.
                frac = rng.beta(2.0, 2.0 + 8.0 * s01)
                ci = min(int(frac * len(club_names)), len(club_names) - 1)
                club_perf = float(np.clip(0.55 * spi_norm[ci] + 0.45 * (base + rng.normal(0, 0.05)), 0.0, 1.0))
                minutes_mu = max(f.avg_minutes_share, 0.05)
                minutes = float(np.clip(rng.beta(5.0, 5.0 * (1.0 - minutes_mu) / minutes_mu), 0.05, 1.0))
                rows.append({
                    "team": team,
                    "player": f"{code} {pos}-{k + 1}",
                    "position": pos,
                    "age": int(np.clip(rng.normal(avg_age, 3.5), 17, 40)),
                    "club": club_names[ci],
                    "base_rating": round(base, 4),
                    "form": round(float(np.clip(rng.normal(f.form, 0.12), 0.0, 1.0)), 4),
                    "club_performance": round(club_perf, 4),
                    "club_minutes": round(minutes, 4),
                    "market_value_m": round(float(total_mv * shares[idx - 1]), 3),
                })

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} players for {len(feats)} teams -> {OUT}")


if __name__ == "__main__":
    main()
