"""Final 2026 win probabilities: Elo Monte Carlo blended with the betting market.

The market knows things Elo can't (injuries, form, squad news, public money);
the model knows the bracket math (who actually has to beat whom, pinned real
results). A geometric blend of the two de-vigged distributions keeps both
honest and is well-behaved in the tails.

    final ∝ model^w · market^(1-w)        (w = 0.5 by default)

Outputs final_2026_prediction.csv and prints the table.
"""

import argparse

import numpy as np
import pandas as pd

# ESPN outright winner odds, June 11 2026 (American odds / fractional).
# https://www.espn.com/espn/betting/story/_/id/48386952/
MARKET_AMERICAN = {
    "Spain": 450, "France": 475, "England": 750, "Portugal": 800,
    "Brazil": 950, "Argentina": 1000, "Germany": 1400, "Netherlands": 2000,
    "Norway": 3000, "Belgium": 4000, "Colombia": 5000, "Mexico": 5000,
    "Morocco": 5000, "United States": 6000, "Japan": 6000, "Uruguay": 6500,
    "Switzerland": 6500, "Ecuador": 8000, "Türkiye": 9000, "Croatia": 9000,
    "Senegal": 9000, "Sweden": 12000, "Austria": 15000, "Canada": 20000,
    "South Korea": 20000, "Scotland": 20000, "Côte d'Ivoire": 25000,
    "Paraguay": 30000, "Egypt": 30000, "Ghana": 30000, "Algeria": 35000,
    "Bosnia and Herzegovina": 50000, "Czechia": 50000, "Tunisia": 50000,
    "Australia": 60000, "Iran": 70000, "DR Congo": 100000,
    "Saudi Arabia": 100000, "Panama": 100000, "Cabo Verde": 100000,
    "Qatar": 150000, "Uzbekistan": 150000, "New Zealand": 150000,
    "Iraq": 150000, "South Africa": 150000, "Jordan": 250000,
    "Curaçao": 250000, "Haiti": 250000,
}


def implied_devig(american: dict) -> pd.Series:
    """American odds -> implied probabilities, overround removed."""
    raw = {t: 100.0 / (o + 100.0) for t, o in american.items()}
    s = sum(raw.values())
    return pd.Series({t: p / s for t, p in raw.items()}, name="market")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-weight", type=float, default=0.5,
                    help="w in final ∝ model^w · market^(1-w)")
    args = ap.parse_args()

    model = pd.read_csv("elo_2026_probs.csv").set_index("Country")
    mp = (model["Win Trophy %"] / 100.0).clip(lower=1e-5)
    mkt = implied_devig(MARKET_AMERICAN)

    df = pd.DataFrame({"model": mp}).join(mkt, how="outer").fillna(1e-5)
    w = args.model_weight
    blend = (df["model"] ** w) * (df["market"] ** (1 - w))
    df["final"] = blend / blend.sum()

    out = pd.DataFrame({
        "Win % (final)": (100 * df["final"]).round(2),
        "Elo model %": (100 * df["model"]).round(2),
        "Market %": (100 * df["market"]).round(2),
    }).sort_values("Win % (final)", ascending=False)
    out.index.name = "Country"
    out.to_csv("final_2026_prediction.csv")
    print(f"geometric blend, model weight {w}")
    print(out.head(16).to_string())
    print("saved -> final_2026_prediction.csv")


if __name__ == "__main__":
    main()
