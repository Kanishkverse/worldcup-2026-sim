"""Team style embeddings and style-on-style matchups (Phase 2).

Two teams of equal rating do not produce the same match depending on how
they play. A patient possession side runs into a different game against a
deep block than against a high press or a fast transition team. This module
loads a six-dimensional style vector per team and turns the interaction
between two styles into a pair of expected-goals multipliers, capped at
+/-15% so style colours the match without overruling strength.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

from .priors import DATA_DIR

STYLES_CSV = os.path.join(DATA_DIR, "team_styles.csv")

DIMENSIONS = (
    "possession", "pressing", "transition_speed",
    "defensive_block", "set_piece_strength", "crossing_frequency",
)

# Hard cap on what style can do to a team's expected goals in one match.
MAX_ADJ = 0.15

_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Turkey": "Türkiye",
}


@dataclass(frozen=True)
class Style:
    possession: float = 0.55
    pressing: float = 0.55
    transition_speed: float = 0.60
    defensive_block: float = 0.58
    set_piece_strength: float = 0.58
    crossing_frequency: float = 0.55


_CACHE: Optional[Dict[str, Style]] = None


def load_styles(path: str = STYLES_CSV) -> Dict[str, Style]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    out: Dict[str, Style] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["team"].strip()] = Style(
                **{d: float(row[d]) for d in DIMENSIONS}
            )
    _CACHE = out
    return out


def style_of(team: str) -> Style:
    styles = load_styles()
    if team in styles:
        return styles[team]
    canon = _ALIASES.get(team)
    if canon and canon in styles:
        return styles[canon]
    return Style()  # neutral default for any team without a row


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _attack_log_adjustment(att: Style, dfn: Style) -> float:
    """Log-space xG adjustment for `att` attacking `dfn`.

    Each term is a small footballing effect; they add up and then get
    squashed and capped by matchup_modifier.
    """
    exposure = 0.5 * dfn.possession + 0.5 * dfn.pressing  # how much space they leave

    block_frustration = -0.18 * att.possession * dfn.defensive_block
    transition_reward = 0.22 * att.transition_speed * (exposure - 0.5)
    press_penalty = -0.16 * dfn.pressing * max(0.0, att.possession - att.transition_speed)
    set_pieces = 0.10 * (att.set_piece_strength - 0.5)
    crossing = (0.06 * att.crossing_frequency * (1.0 - dfn.defensive_block)
                - 0.08 * att.crossing_frequency * dfn.defensive_block)

    return block_frustration + transition_reward + press_penalty + set_pieces + crossing


def matchup_modifier(team_a: str, team_b: str):
    """Return (mult_a, mult_b): xG multipliers for each side, in [0.85, 1.15]."""
    sa, sb = style_of(team_a), style_of(team_b)
    mult_a = math.exp(_attack_log_adjustment(sa, sb))
    mult_b = math.exp(_attack_log_adjustment(sb, sa))
    return (_clip(mult_a, 1 - MAX_ADJ, 1 + MAX_ADJ),
            _clip(mult_b, 1 - MAX_ADJ, 1 + MAX_ADJ))
