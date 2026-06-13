"""Merge agentic run batches into one combined probability table.

Stage reach is reconstructed from each run's match log (no probs CSV needed),
so batches with different --out-prefix values can be pooled:

    python3 merge_agentic_runs.py agentic_2026 agentic_2026b
    -> agentic_2026_combined_probs.csv  (over every *_run*_matches.json found)
"""

import glob
import json
import sys
from collections import Counter

import pandas as pd

_STAGES = ["GROUP", "R32", "R16", "QF", "SF", "FINAL", "CHAMPION"]
_RANK = {s: i for i, s in enumerate(_STAGES)}


def reach_from_log(path: str) -> dict:
    log = json.load(open(path))
    reach = {}
    for m in log:
        stage = m["stage"]
        for team in (m["home"], m["away"]):
            if team not in reach or _RANK[stage] > _RANK[reach[team]]:
                reach[team] = stage
    final = next(m for m in log if m["stage"] == "FINAL")
    reach[final["winner"]] = "CHAMPION"
    return reach


def main() -> None:
    prefixes = sys.argv[1:] or ["agentic_2026"]
    files = sorted(f for p in prefixes for f in glob.glob(f"{p}_run*_matches.json"))
    if not files:
        sys.exit(f"no *_run*_matches.json found for prefixes: {prefixes}")
    reaches = [reach_from_log(f) for f in files]

    teams = sorted({t for r in reaches for t in r})
    rows = []
    for team in teams:
        counts = Counter(r.get(team, "GROUP") for r in reaches)
        def at_least(stage):
            return round(100.0 * sum(n for s, n in counts.items()
                                     if _RANK[s] >= _RANK[stage]) / len(reaches), 1)
        rows.append({"Country": team,
                     "Reach R32 %": at_least("R32"), "Reach R16 %": at_least("R16"),
                     "Reach QF %": at_least("QF"), "Reach SF %": at_least("SF"),
                     "Reach Final %": at_least("FINAL"),
                     "Win Trophy %": at_least("CHAMPION")})
    rows.sort(key=lambda r: (-r["Win Trophy %"], -r["Reach Final %"], r["Country"]))
    df = pd.DataFrame(rows)
    out = f"{prefixes[0]}_combined_probs.csv"
    df.to_csv(out, index=False)
    print(f"merged {len(files)} runs from {len(prefixes)} prefix(es)")
    print(df.head(12).to_string(index=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
