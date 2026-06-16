"""Venue and travel effects (Phase 4).

2026 is played across sea-level coastal heat, humid Gulf cities, and the
2,240 m thin air of Mexico City. Those conditions are not cosmetic: altitude
and heat drain legs, and the cross-continent hops between, say, Vancouver and
Miami add their own load. This module reads the stadium table and turns a
(venue, team, travel) tuple into bounded fatigue and expected-goals effects.

Teams from high-altitude footballing nations are not punished by altitude,
which is the single most important realism detail here. Mexico at the Azteca
is at home; a sea-level side arriving from a coastal camp is not.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

from .priors import DATA_DIR

VENUES_CSV = os.path.join(DATA_DIR, "venues.csv")

# Nations whose players live and play at altitude, so the Azteca and
# Guadalajara are home conditions rather than a penalty.
ALTITUDE_ACCLIMATISED = {"Mexico", "Ecuador", "Colombia"}


@dataclass(frozen=True)
class Venue:
    venue: str
    city: str
    country: str
    altitude_m: float
    temp_c: float
    humidity_pct: float
    lat: float
    lon: float


@dataclass(frozen=True)
class VenueEffect:
    fatigue_add: float     # extra match load, 0..~0.4
    stamina_mult: float    # multiplier on stamina, <= 1.0
    xg_mult: float         # small expected-goals multiplier, ~[0.92, 1.03]


_CACHE: Optional[Dict[str, Venue]] = None


def load_venues(path: str = VENUES_CSV) -> Dict[str, Venue]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    out: Dict[str, Venue] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["venue"].strip()] = Venue(
                venue=row["venue"].strip(),
                city=row["city"].strip(),
                country=row["country"].strip(),
                altitude_m=float(row["altitude_m"]),
                temp_c=float(row["temp_c"]),
                humidity_pct=float(row["humidity_pct"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            )
    _CACHE = out
    return out


def get_venue(name: str) -> Optional[Venue]:
    return load_venues().get(name)


def distance_km(venue_a: str, venue_b: str) -> float:
    """Great-circle distance between two venues, for travel load."""
    va, vb = get_venue(venue_a), get_venue(venue_b)
    if va is None or vb is None or va.venue == vb.venue:
        return 0.0
    r = 6371.0
    p1, p2 = math.radians(va.lat), math.radians(vb.lat)
    dphi = math.radians(vb.lat - va.lat)
    dlmb = math.radians(vb.lon - va.lon)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def venue_modifier(venue_name: str, team: str, travel_km: float = 0.0) -> VenueEffect:
    """Combine altitude, heat, humidity and travel into one bounded effect."""
    v = get_venue(venue_name)
    if v is None:
        return VenueEffect(fatigue_add=0.0, stamina_mult=1.0, xg_mult=1.0)

    fatigue_add = 0.0
    xg_mult = 1.0

    # Altitude: only above ~1000 m, and only for sides not used to it.
    if v.altitude_m > 1000.0 and team not in ALTITUDE_ACCLIMATISED:
        alt = _clip((v.altitude_m - 1000.0) / 1500.0, 0.0, 1.0)
        fatigue_add += 0.18 * alt
        xg_mult *= (1.0 - 0.06 * alt)

    # Heat, amplified by humidity. Neutral at/below 26 C.
    heat = _clip((v.temp_c - 26.0) / 12.0, 0.0, 1.0)
    heat_load = heat * (0.7 + 0.3 * v.humidity_pct / 100.0)
    fatigue_add += 0.14 * heat_load
    xg_mult *= (1.0 - 0.04 * heat_load)

    # Travel: capped so one long hop matters but does not dominate.
    t = _clip(travel_km / 4000.0, 0.0, 1.0)
    fatigue_add += 0.10 * t

    fatigue_add = _clip(fatigue_add, 0.0, 0.40)
    stamina_mult = 1.0 - 0.4 * fatigue_add
    xg_mult = _clip(xg_mult, 0.92, 1.03)
    return VenueEffect(fatigue_add=fatigue_add, stamina_mult=stamina_mult, xg_mult=xg_mult)
