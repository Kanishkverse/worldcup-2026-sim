"""Convert the real FIFA squads into the simulator's per-player format.

Input : squads_2026_real.csv  (from parse_squads_pdf.py — real names, clubs,
                               age, caps, international goals)
        spi_global_rankings.csv (real club SPI table — club quality)
Output: players_2026_real.csv  (WC_PLAYERS_CSV format, real players)

Per-player metrics are derived from REAL data only:
  club_performance  club SPI when the club appears in the SPI table, else a
                    league-tier prior from the club's country code
  base_rating       team composite strength tier shifted by club quality,
                    experience (caps) and an age curve peaking at 27
  club_minutes      experience proxy: caps percentile inside the squad
                    (regular internationals start for their clubs)
  form              team recent-form scalar (no per-player form feed exists)
  market_value_m    real squad value split by rating rank (heavy tail)

Run:  python3 build_real_players.py
"""

import csv
import math
import re
import unicodedata

import numpy as np
import pandas as pd

import worldcup_2026_sim as wc

SEED = 11
POS_MAP = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}

# League quality prior by club country (used when the club has no SPI row).
LEAGUE_TIER = {
    "ENG": 0.95, "ESP": 0.92, "GER": 0.88, "ITA": 0.87, "FRA": 0.85,
    "NED": 0.74, "POR": 0.74, "BRA": 0.70, "TUR": 0.66, "BEL": 0.68,
    "ARG": 0.64, "MEX": 0.62, "USA": 0.61, "KSA": 0.62, "RUS": 0.60,
    "SCO": 0.60, "GRE": 0.58, "SUI": 0.62, "AUT": 0.60, "DEN": 0.60,
    "CRO": 0.56, "NOR": 0.56, "SWE": 0.56, "JPN": 0.55, "KOR": 0.54,
    "QAT": 0.52, "UAE": 0.50, "EGY": 0.48, "MAR": 0.48, "TUN": 0.46,
    "ALG": 0.45, "RSA": 0.44, "CHN": 0.46, "AUS": 0.50, "COL": 0.50,
    "ECU": 0.48, "URU": 0.52, "PAR": 0.46, "PER": 0.45, "CHI": 0.46,
}
DEFAULT_TIER = 0.42


def norm_club(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    drop = {"fc", "cf", "cd", "ac", "sc", "afc", "club", "de", "the", "fk",
            "if", "bk", "sk", "ca", "cdo", "real" if False else "zzz"}
    toks = [t for t in s.split() if t not in drop and len(t) > 1]
    return " ".join(toks)


def build_spi_lookup() -> dict:
    clubs = pd.read_csv("spi_global_rankings.csv")
    lo, hi = clubs["spi"].min(), clubs["spi"].max()
    lut = {}
    for _, r in clubs.iterrows():
        lut[norm_club(r["name"])] = (r["spi"] - lo) / (hi - lo)
    return lut


def club_quality(club: str, country: str, spi_lut: dict) -> tuple:
    key = norm_club(club)
    if key in spi_lut:
        return spi_lut[key], "spi"
    for k in spi_lut:                              # substring fallback
        if key and (key in k or k in key):
            return spi_lut[k], "spi~"
    return LEAGUE_TIER.get(country, DEFAULT_TIER), "tier"


def age_factor(age: int) -> float:
    """Peak at 27, gentle decline either side."""
    return float(np.clip(1.0 - 0.012 * abs(age - 27) - 0.004 * max(age - 31, 0) ** 2, 0.55, 1.0))


def main() -> None:
    rng = np.random.default_rng(SEED)
    feats = wc.features_2026_live()
    strengths = wc.composite_strengths(feats)
    spi_lut = build_spi_lookup()

    squad = pd.read_csv("squads_2026_real.csv")
    out_rows, spi_hits = [], 0
    for team, grp in squad.groupby("team"):
        if team not in strengths:
            print(f"  ! skipping unknown team {team}")
            continue
        s = strengths[team]
        f = feats[team]
        total_mv = f.market_value_m if not math.isnan(f.market_value_m) else 0.0
        caps_rank = grp["caps"].rank(pct=True)

        recs = []
        for (_, r), cap_pct in zip(grp.iterrows(), caps_rank):
            cq, src = club_quality(str(r["club"]), str(r["club_country"]), spi_lut)
            spi_hits += src.startswith("spi")
            exp = min(math.log1p(r["caps"]) / math.log1p(120), 1.0)
            goal_rate = (r["intl_goals"] / max(r["caps"], 1)) if r["position"] in ("FW", "MF") else 0.0
            base = (s
                    + 0.10 * (cq - 0.55)            # club above/below squad norm
                    + 0.05 * (exp - 0.5)            # international experience
                    + 0.06 * min(goal_rate, 0.8)    # proven scorers
                    ) * age_factor(int(r["age"]))
            base = float(np.clip(base + rng.normal(0, 0.015), 0.20, 0.99))
            minutes = float(np.clip(0.45 + 0.5 * cap_pct + rng.normal(0, 0.04), 0.10, 0.98))
            recs.append({
                "team": team,
                "player": r["name"],
                "position": POS_MAP[r["position"]],
                "age": int(r["age"]),
                "club": r["club"],
                "club_country": r["club_country"],
                "caps": int(r["caps"]),
                "intl_goals": int(r["intl_goals"]),
                "base_rating": round(base, 4),
                "form": round(float(np.clip(rng.normal(f.form, 0.06), 0.0, 1.0)), 4),
                "club_performance": round(float(np.clip(0.7 * cq + 0.3 * exp, 0.0, 1.0)), 4),
                "club_minutes": round(minutes, 4),
            })
        # Real squad value split by rating rank, heavy-tailed (stars first).
        recs.sort(key=lambda x: -x["base_rating"])
        shares = np.sort(rng.pareto(2.5, len(recs)) + 0.05)[::-1]
        shares /= shares.sum()
        for rec, sh in zip(recs, shares):
            rec["market_value_m"] = round(float(total_mv * sh), 3)
        out_rows.extend(recs)

    with open("players_2026_real.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} real players -> players_2026_real.csv "
          f"({spi_hits} club-SPI matches, rest league-tier prior)")


if __name__ == "__main__":
    main()
