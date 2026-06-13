"""
================================================================================
 FIFA WORLD CUP — PRODUCTION-GRADE MONTE CARLO SIMULATION + BACKTEST SYSTEM v2
================================================================================

A modular simulation engine for the 48-team 2026 FIFA World Cup, with a
historical backtest harness covering the 32-team 2010 / 2014 / 2018 / 2022
tournaments.

What changed vs v1
------------------
  * REAL FEATURE SQUADS. Synthetic strength-only squads are replaced by a
    multi-feature team model: national Elo rating, total squad market value
    (Transfermarkt-style, EUR millions), SPI-style rating (FiveThirtyEight
    soccer-spi when reachable), recent form, injury status, and per-player
    club-minutes share. FIFA ranking points are demoted to ONE feature among
    several instead of the sole strength source.
  * BACKTEST HARNESS. Real draws + brackets for WC 2010/2014/2018/2022.
    The full feature model is scored against a FIFA-ranking-only baseline on
    champion hit-rate / champion rank / champion probability / log-loss and
    semifinalist recall.
  * LOCAL LLM BACKEND (RunPod / RTX 4090). The tactical agent now speaks the
    OpenAI chat-completions protocol against a LOCAL vLLM (or Ollama) server
    on the pod loopback. Any OpenAI-compatible endpoint/model also works via
    env vars (WC_LLM_BASE_URL / WC_LLM_MODEL / WC_LLM_API_KEY).
  * LIVE MATCH EVENTS. The minute loop now carries mutable per-team match
    state: stochastic red cards apply a permanent, compounding cohesion drop
    for the remainder of the match; players whose accumulated in-match
    fatigue crosses SUB_FATIGUE_THRESHOLD are automatically substituted by
    the best positional bench option, and the player-graph GNN layer is
    re-run to refresh the match lambdas instantly.
  * PROCESS-SAFE LLM INVOCATION. No client object ever crosses a process
    boundary. Each ProcessPoolExecutor worker lazily constructs its own
    agent (and HTTP socket pool) on first use inside the worker process.

Architecture
------------
    Layer 0  FEATURE DATA (TeamFeatures, FeatureStore, live fetchers + offline
             snapshots, composite strength model)
    Layer 1  INVENTORY (Player / Team / WorldCupInventory, async ingestion)
    Layer 2  GRAPH SPATIO-TEMPORAL MATCH ENGINE (11-node GNN cohesion,
             minute-by-minute Attenuated Poisson, red cards, substitutions)
    Layer 3  LLM TACTICAL AGENT (local vLLM via OpenAI protocol, heuristic
             fallback)
    Layer 4  TOURNAMENT RESOLVERS (48-team 2026 bracket; 32-team historical
             bracket)
    Layer 5  MONTE CARLO WRAPPER + TELEMETRY (process-safe)
    Layer 6  BACKTEST HARNESS (2010-2022 validation vs FIFA baseline)

Run (on the RunPod 4090)
------------------------
    # 1. serve a local reasoning model that fits in 24 GB:
    #    vllm serve Qwen/Qwen3-14B-AWQ --max-model-len 8192 \
    #         --gpu-memory-utilization 0.90 --port 8000
    #    (alternatives: Qwen/Qwen3-8B, deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    #     with AWQ/GPTQ quant; or Ollama: `ollama serve` + WC_LLM_BASE_URL=
    #     http://127.0.0.1:11434/v1 WC_LLM_MODEL=qwen3:14b)
    # 2. run:
    python worldcup_sim_v2.py --iterations 1000 --workers 8
    python worldcup_sim_v2.py --backtest 2010,2014,2018,2022 --iterations 400
    python worldcup_sim_v2.py --showcase --llm     # one verbose LLM match
    python worldcup_sim_v2.py --refresh-features   # pull live Elo/SPI/MV feeds

Dependencies:
    numpy, pandas            (required)
    openai                   (optional — enables the local/remote LLM agent)
    httpx or requests        (optional — live feature/injury ingestion)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd

from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "en.txt"))
# ------------------------------------------------------------------------------
# Optional OpenAI-protocol SDK. On the RunPod 4090 this talks to a LOCAL vLLM
# (or Ollama) server on the loopback — no cloud key required. The system is
# fully functional without it: the TacticalManagerAgent degrades to a
# deterministic local heuristic, keeping the Monte Carlo loop offline and free.
# ------------------------------------------------------------------------------
try:
    from openai import OpenAI  # type: ignore

    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

# Optional async HTTP client for the live data-ingestion layer. Falls back to
# `requests` on a thread pool, then to a pure offline no-op, so the simulator
# never hard-depends on network access.
try:
    import httpx  # type: ignore

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False

try:
    import requests  # type: ignore

    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
    _REQUESTS_AVAILABLE = False

LOGGER = logging.getLogger("wc2026")

# ------------------------------------------------------------------------------
# LLM endpoint configuration (OpenAI chat-completions protocol).
#
# Default target is a vLLM server on the pod loopback. Qwen3-14B-AWQ is the
# default model id because it is a strong open reasoning model that fits a
# single RTX 4090 (24 GB) with headroom for an 8k context. Everything is
# overridable without code edits:
#
#   WC_LLM_BASE_URL   http://127.0.0.1:8000/v1      (vLLM default)
#                     http://127.0.0.1:11434/v1     (Ollama)
#                     https://api.openai.com/v1     (OpenAI cloud — set the
#                                                    model id you are entitled
#                                                    to, e.g. a current
#                                                    gpt-*-mini reasoning tier)
#   WC_LLM_MODEL      served model id
#   WC_LLM_API_KEY    "EMPTY" works for local vLLM; a real key for cloud
# ------------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("WC_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = os.getenv("WC_LLM_MODEL", "Qwen/Qwen3-14B-AWQ")
LLM_API_KEY = os.getenv("WC_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
LLM_TIMEOUT_S = float(os.getenv("WC_LLM_TIMEOUT", "30"))


# ==============================================================================
# LAYER 0 — FEATURE DATA (real multi-source team features)
# ==============================================================================
#
# Strength is no longer "FIFA points and nothing else". Each team carries a
# TeamFeatures row sourced from:
#
#   elo               World Football Elo rating (eloratings.net family)
#   market_value_m    total squad market value, EUR millions (Transfermarkt-
#                     style; log-compressed before normalisation because squad
#                     values are heavy-tailed)
#   spi               SPI-style rating (FiveThirtyEight soccer-spi global
#                     rankings when the live CSV on GitHub is reachable)
#   fifa_points/rank  FIFA ranking — deliberately just ONE feature now
#   form              recent-form scalar in [0,1] (rolling last-10 proxy)
#   avg_minutes_share mean club-minutes share of the squad in [0,1] — feeds
#                     per-player match sharpness
#   injuries_out      number of first-choice players ruled out pre-tournament
#
# The composite model:
#     strength = 0.45*elo_n + 0.30*log_mv_n + 0.15*spi_n + 0.10*fifa_n
# (weights renormalise automatically when a feature column is missing, e.g.
# SPI for the historical 2010 field). The FIFA-only baseline used by the
# backtest sets the weight vector to (0, 0, 0, 1).
#
# Embedded tables below are OFFLINE SNAPSHOTS so the engine always runs —
# values are pre-tournament approximations transcribed from eloratings.net,
# Transfermarkt squad pages and the FIFA ranking archive, and should be
# refreshed through FeatureStore.refresh_live() (or replaced with your own
# CSVs via WC_ELO_URL / WC_MV_CSV / WC_SPI_URL) before any run whose numbers
# you intend to publish.
# ==============================================================================


@dataclass(slots=True)
class TeamFeatures:
    """One team's multi-source pre-tournament feature row."""

    team: str
    elo: float = float("nan")
    market_value_m: float = float("nan")   # EUR millions, total squad
    spi: float = float("nan")
    fifa_points: float = float("nan")      # modern points scale (2026 field)
    fifa_rank: float = float("nan")        # ordinal rank (historical fields)
    form: float = 0.55                     # rolling recent form in [0,1]
    avg_minutes_share: float = 0.70        # mean club minutes share in [0,1]
    injuries_out: int = 0                  # first-choice players ruled out


# Composite weights (renormalised over the features actually present).
FEATURE_WEIGHTS: Dict[str, float] = {
    "elo": 0.45,
    "market_value": 0.30,
    "spi": 0.15,
    "fifa": 0.10,
}
FIFA_ONLY_WEIGHTS: Dict[str, float] = {
    "elo": 0.0, "market_value": 0.0, "spi": 0.0, "fifa": 1.0,
}

STRENGTH_LO, STRENGTH_HI = 0.45, 0.97   # band the composite maps into


def _minmax(vals: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a {team: value} map to [0,1]; NaNs -> field median."""
    clean = {k: v for k, v in vals.items() if not math.isnan(v)}
    if not clean:
        return {k: 0.5 for k in vals}
    lo, hi = min(clean.values()), max(clean.values())
    med = float(np.median(list(clean.values())))
    span = (hi - lo) or 1.0
    return {
        k: ((v if not math.isnan(v) else med) - lo) / span
        for k, v in vals.items()
    }


def composite_strengths(
    features: Dict[str, TeamFeatures],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Collapse the feature rows of a tournament field into [STRENGTH_LO,
    STRENGTH_HI] composite strengths.

    Market value is log-compressed (log1p) before normalisation: a 1.3bn
    England squad should not be 50x a 25m Qatar squad in feature space.
    FIFA enters as points when available (2026 field) else inverse rank
    (historical fields) — and is capped at its configured weight either way.
    """
    w = dict(weights or FEATURE_WEIGHTS)

    elo_n = _minmax({t: f.elo for t, f in features.items()})
    mv_n = _minmax({
        t: (math.log1p(f.market_value_m) if not math.isnan(f.market_value_m)
            else float("nan"))
        for t, f in features.items()
    })
    spi_n = _minmax({t: f.spi for t, f in features.items()})

    # FIFA: prefer points; fall back to inverse rank (rank 1 == best).
    have_points = any(not math.isnan(f.fifa_points) for f in features.values())
    if have_points:
        fifa_n = _minmax({t: f.fifa_points for t, f in features.items()})
    else:
        fifa_n = _minmax({
            t: (-f.fifa_rank if not math.isnan(f.fifa_rank) else float("nan"))
            for t, f in features.items()
        })

    # Drop weights for feature columns that are entirely missing, renormalise.
    col_present = {
        "elo": any(not math.isnan(f.elo) for f in features.values()),
        "market_value": any(not math.isnan(f.market_value_m) for f in features.values()),
        "spi": any(not math.isnan(f.spi) for f in features.values()),
        "fifa": True,
    }
    for k, present in col_present.items():
        if not present:
            w[k] = 0.0
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    out: Dict[str, float] = {}
    for t in features:
        score = (w["elo"] * elo_n[t] + w["market_value"] * mv_n[t]
                 + w["spi"] * spi_n[t] + w["fifa"] * fifa_n[t])
        out[t] = STRENGTH_LO + (STRENGTH_HI - STRENGTH_LO) * float(score)
    return out


# ------------------------------------------------------------------------------
# OFFLINE SNAPSHOT — 2026 field (48 teams).
# team -> (elo, market_value_m, fifa_points, form, avg_minutes_share, inj_out)
# Approximate pre-tournament values; refresh via FeatureStore.refresh_live().
# ------------------------------------------------------------------------------
_FEATURES_2026: Dict[str, Tuple[float, float, float, float, float, int]] = {
    "Mexico":                 (1850,  220, 1681.0, 0.60, 0.74, 0),
    "South Africa":           (1620,   35, 1445.0, 0.55, 0.66, 0),
    "South Korea":            (1780,  190, 1575.0, 0.55, 0.72, 0),
    "Czechia":                (1740,  210, 1500.0, 0.52, 0.70, 0),
    "Canada":                 (1790,  180, 1540.0, 0.58, 0.72, 0),
    "Bosnia and Herzegovina": (1700,  120, 1450.0, 0.52, 0.68, 0),
    "Qatar":                  (1630,   25, 1400.0, 0.50, 0.78, 0),
    "Switzerland":            (1880,  320, 1649.0, 0.60, 0.74, 0),
    "Brazil":                 (2030, 1180, 1761.0, 0.60, 0.76, 1),
    "Morocco":                (1940,  420, 1756.0, 0.66, 0.74, 0),
    "Haiti":                  (1450,   15, 1280.0, 0.48, 0.60, 0),
    "Scotland":               (1740,  160, 1500.0, 0.55, 0.72, 0),
    "United States":          (1790,  260, 1673.0, 0.55, 0.72, 0),
    "Paraguay":               (1780,   90, 1480.0, 0.55, 0.70, 0),
    "Australia":              (1750,   60, 1500.0, 0.55, 0.70, 0),
    "Türkiye":                (1850,  380, 1560.0, 0.60, 0.72, 0),
    "Germany":                (1980,  980, 1730.0, 0.58, 0.76, 1),
    "Curaçao":                (1480,   20, 1310.0, 0.50, 0.62, 0),
    "Côte d'Ivoire":          (1740,  280, 1490.0, 0.55, 0.70, 0),
    "Ecuador":                (1870,  320, 1570.0, 0.58, 0.72, 0),
    "Netherlands":            (2000,  860, 1758.0, 0.60, 0.76, 0),
    "Japan":                  (1880,  350, 1660.0, 0.62, 0.74, 0),
    "Sweden":                 (1760,  300, 1530.0, 0.50, 0.72, 0),
    "Tunisia":                (1700,   70, 1490.0, 0.55, 0.68, 0),
    "Belgium":                (1940,  560, 1735.0, 0.56, 0.74, 0),
    "Egypt":                  (1740,  140, 1510.0, 0.58, 0.72, 0),
    "Iran":                   (1800,   60, 1590.0, 0.55, 0.72, 0),
    "New Zealand":            (1590,   30, 1320.0, 0.52, 0.64, 0),
    "Spain":                  (2210, 1320, 1876.0, 0.70, 0.78, 0),
    "Cabo Verde":             (1560,   35, 1380.0, 0.55, 0.64, 0),
    "Saudi Arabia":           (1660,   30, 1430.0, 0.50, 0.76, 0),
    "Uruguay":                (1930,  470, 1673.0, 0.55, 0.74, 0),
    "France":                 (2070, 1380, 1877.0, 0.62, 0.76, 1),
    "Senegal":                (1820,  360, 1689.0, 0.58, 0.72, 0),
    "Iraq":                   (1640,   25, 1400.0, 0.50, 0.68, 0),
    "Norway":                 (1850,  520, 1530.0, 0.64, 0.76, 0),
    "Argentina":              (2150,  920, 1875.0, 0.65, 0.76, 0),
    "Algeria":                (1760,  210, 1500.0, 0.56, 0.70, 0),
    "Austria":                (1830,  310, 1590.0, 0.58, 0.74, 0),
    "Jordan":                 (1600,   15, 1390.0, 0.52, 0.66, 0),
    "Portugal":               (2010, 1150, 1764.0, 0.62, 0.76, 0),
    "DR Congo":               (1670,  110, 1460.0, 0.55, 0.68, 0),
    "Uzbekistan":             (1680,   45, 1440.0, 0.54, 0.70, 0),
    "Colombia":               (1950,  330, 1693.0, 0.58, 0.72, 0),
    "England":                (2080, 1500, 1826.0, 0.62, 0.76, 0),
    "Croatia":                (1920,  340, 1717.0, 0.54, 0.74, 0),
    "Ghana":                  (1680,  230, 1440.0, 0.54, 0.70, 0),
    "Panama":                 (1660,   30, 1430.0, 0.55, 0.66, 0),
}


def features_2026() -> Dict[str, TeamFeatures]:
    """Snapshot feature rows for the 2026 field."""
    return {
        t: TeamFeatures(team=t, elo=e, market_value_m=mv, fifa_points=fp,
                        form=fo, avg_minutes_share=ms, injuries_out=inj)
        for t, (e, mv, fp, fo, ms, inj) in _FEATURES_2026.items()
    }


# Per-process cache: refresh the snapshot from the configured Elo/SPI/MV
# sources once, then reuse for every Monte Carlo iteration in this process.
_FEATS_2026_LIVE: Optional[Dict[str, TeamFeatures]] = None


def features_2026_live() -> Dict[str, TeamFeatures]:
    """2026 feature rows, refreshed from WC_ELO_URL/WC_SPI_URL/WC_MV_CSV.

    Set WC_LIVE_FEATURES=0 to force pure snapshot values. Sources that are
    unset or unreachable are skipped — snapshots fill the gaps.
    """
    global _FEATS_2026_LIVE
    if _FEATS_2026_LIVE is None:
        feats = features_2026()
        if os.getenv("WC_LIVE_FEATURES", "1") != "0":
            FeatureStore(feats).refresh_live()
        _FEATS_2026_LIVE = feats
    return _FEATS_2026_LIVE


# ------------------------------------------------------------------------------
# FeatureStore — live refresh of the snapshot tables (key-free endpoints).
# ------------------------------------------------------------------------------


class FeatureStore:
    """
    Holds the {team: TeamFeatures} table and refreshes it from open feeds.

      * Elo   — a World Football Elo TSV/CSV (eloratings.net family). The
                site has no official API; point WC_ELO_URL at any mirror or a
                local export with columns containing team name + rating.
      * SPI   — FiveThirtyEight soccer-spi international rankings CSV, still
                hosted on GitHub (name,off,def,spi).
      * MV    — Transfermarkt has no official API and scraping violates its
                ToS; supply a CSV (path or URL) via WC_MV_CSV with columns
                `team,market_value_m`, e.g. exported from the open
                `transfermarkt-datasets` project or a licensed feed.

    Any source that fails is skipped — the snapshot values stand.
    """

    SPI_URL = os.getenv(
        "WC_SPI_URL",
        "",
    )
    ELO_URL = os.getenv("WC_ELO_URL", "")
    MV_CSV = os.getenv("WC_MV_CSV", "")

    # Name bridges between feeds and the draw table.
    ALIASES = {
        "USA": "United States", "Korea Republic": "South Korea",
        "Ivory Coast": "Côte d'Ivoire", "Czech Republic": "Czechia",
        "Turkey": "Türkiye", "Cape Verde Islands": "Cabo Verde",
        "Cape Verde": "Cabo Verde", "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Bosnia": "Bosnia and Herzegovina", "Congo DR": "DR Congo",
        "Democratic Republic of Congo": "DR Congo", "Turkiye": "Türkiye",
    }

    def __init__(self, features: Dict[str, TeamFeatures]):
        self.features = features

    def _canon(self, name: str) -> str:
        # eloratings.net exports join words with NBSP (\xa0), not spaces.
        name = name.replace("\xa0", " ").strip()
        return self.ALIASES.get(name, name)

    def _fetch_text(self, url: str, timeout: float = 20.0) -> Optional[str]:
        # 1. Clear any file:// prefixes or handle standard local paths
        clean_path = url.replace("file://localhost", "").replace("file://", "")
        if os.path.exists(clean_path) and not url.startswith("http"):
            try:
                with open(clean_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as exc:
                LOGGER.warning("Local file read failed for %s: %s", clean_path, exc)
                return None
    
        # 2. Fallback to network request if it is an actual URL
        try:
            if _REQUESTS_AVAILABLE:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                return r.text
            if _HTTPX_AVAILABLE:
                resp = httpx.get(url, timeout=timeout, follow_redirects=True)
                resp.raise_for_status()
                return resp.text
        except Exception as exc:  # pragma: no cover - network guard
            LOGGER.warning("Feature fetch failed for %s: %s", url, exc)
        return None

    def refresh_spi(self) -> int:
        """Pull SPI-style ratings from the soccer-spi CSV. Returns rows applied."""
        if not self.SPI_URL:
            LOGGER.info("WC_SPI_URL not set; keeping snapshot SPI values.")
            return 0
        body = self._fetch_text(self.SPI_URL)
        if not body:
            return 0
        applied = 0
        for row in csv.DictReader(io.StringIO(body)):
            team = self._canon((row.get("name") or "").strip())
            if team in self.features:
                try:
                    self.features[team].spi = float(row["spi"])
                    applied += 1
                except (KeyError, ValueError):
                    continue
        LOGGER.info("SPI refresh: applied %d rows.", applied)
        return applied

    def refresh_elo(self) -> int:
        """Pull Elo ratings from WC_ELO_URL (TSV or CSV; team + rating cols)."""
        if not self.ELO_URL:
            LOGGER.info("WC_ELO_URL not set; keeping snapshot Elo values.")
            return 0
        body = self._fetch_text(self.ELO_URL)
        if not body:
            return 0
        delim = "\t" if "\t" in body.splitlines()[0] else ","
        applied = 0
        for raw in csv.reader(io.StringIO(body), delimiter=delim):
            # Heuristic: find a cell that is a known team and a numeric > 1000.
            team = next((self._canon(c.strip()) for c in raw
                         if self._canon(c.strip()) in self.features), None)
            rating = next((float(c) for c in raw
                           if re.fullmatch(r"\d{4}(\.\d+)?", c.strip())), None)
            if team and rating:
                self.features[team].elo = rating
                applied += 1
        LOGGER.info("Elo refresh: applied %d rows.", applied)
        return applied

    def refresh_market_values(self) -> int:
        """Load `team,market_value_m` rows from WC_MV_CSV (path or URL)."""
        if not self.MV_CSV:
            LOGGER.info("WC_MV_CSV not set; keeping snapshot market values.")
            return 0
        body = (self._fetch_text(self.MV_CSV) if self.MV_CSV.startswith("http")
                else (open(self.MV_CSV, encoding="utf-8").read()
                      if os.path.exists(self.MV_CSV) else None))
        if not body:
            return 0
        applied = 0
        for row in csv.DictReader(io.StringIO(body)):
            team = self._canon((row.get("team") or "").strip())
            if team in self.features:
                try:
                    self.features[team].market_value_m = float(row["market_value_m"])
                    applied += 1
                except (KeyError, ValueError):
                    continue
        LOGGER.info("Market-value refresh: applied %d rows.", applied)
        return applied

    def refresh_live(self) -> Dict[str, int]:
        """Refresh every configured live source; snapshots fill the gaps."""
        return {
            "spi": self.refresh_spi(),
            "elo": self.refresh_elo(),
            "market_value": self.refresh_market_values(),
        }


# ------------------------------------------------------------------------------
# OFFLINE SNAPSHOTS — HISTORICAL FIELDS (backtest ground truth).
#
# For each World Cup: the real group draw, plus pre-tournament features
# per team -> (elo, market_value_m, fifa_rank). Elo from the eloratings.net
# archive, market values from Transfermarkt squad pages at tournament time,
# FIFA rank from the pre-tournament ranking release. Values are snapshot
# approximations for backtesting the *feature stack*; refresh from archives
# for publication-grade numbers. Form / minutes / injuries are held neutral
# historically (not reconstructable without a paid feed), so the backtest
# isolates exactly the question asked: does Elo + market value (+ SPI where
# available) beat FIFA rank alone?
# ------------------------------------------------------------------------------

# year -> {group: [(team, elo, market_value_m, fifa_rank), ...]}
_HISTORICAL_FIELDS: Dict[int, Dict[str, List[Tuple[str, float, float, float]]]] = {
    2010: {
        "A": [("South Africa", 1612, 40, 83), ("Mexico", 1862, 95, 17),
              ("Uruguay", 1869, 130, 16), ("France", 1945, 405, 9)],
        "B": [("Argentina", 1968, 396, 7), ("Nigeria", 1726, 75, 21),
              ("South Korea", 1746, 50, 47), ("Greece", 1726, 85, 13)],
        "C": [("England", 2009, 350, 8), ("United States", 1785, 65, 14),
              ("Algeria", 1614, 45, 30), ("Slovenia", 1648, 40, 25)],
        "D": [("Germany", 1930, 308, 6), ("Australia", 1766, 50, 20),
              ("Serbia", 1817, 150, 15), ("Ghana", 1711, 80, 32)],
        "E": [("Netherlands", 2037, 380, 4), ("Denmark", 1811, 90, 36),
              ("Japan", 1729, 55, 45), ("Cameroon", 1668, 110, 19)],
        "F": [("Italy", 1955, 280, 5), ("Paraguay", 1771, 70, 31),
              ("New Zealand", 1457, 10, 78), ("Slovakia", 1654, 60, 34)],
        "G": [("Brazil", 2082, 372, 1), ("North Korea", 1490, 8, 105),
              ("Côte d'Ivoire", 1740, 180, 27), ("Portugal", 1932, 240, 3)],
        "H": [("Spain", 2085, 565, 2), ("Switzerland", 1748, 90, 24),
              ("Honduras", 1633, 30, 38), ("Chile", 1883, 100, 18)],
    },
    2014: {
        "A": [("Brazil", 2113, 467, 3), ("Croatia", 1830, 200, 18),
              ("Mexico", 1832, 90, 20), ("Cameroon", 1644, 100, 56)],
        "B": [("Spain", 2073, 622, 1), ("Netherlands", 1965, 232, 15),
              ("Chile", 1927, 150, 14), ("Australia", 1693, 30, 62)],
        "C": [("Colombia", 1925, 200, 8), ("Greece", 1796, 90, 12),
              ("Côte d'Ivoire", 1737, 130, 23), ("Japan", 1740, 100, 46)],
        "D": [("Uruguay", 1893, 184, 7), ("Costa Rica", 1726, 30, 28),
              ("England", 1898, 360, 10), ("Italy", 1910, 320, 9)],
        "E": [("Switzerland", 1845, 150, 6), ("Ecuador", 1790, 75, 26),
              ("France", 1936, 400, 17), ("Honduras", 1633, 25, 33)],
        "F": [("Argentina", 1996, 446, 5), ("Bosnia and Herzegovina", 1819, 130, 21),
              ("Iran", 1700, 35, 43), ("Nigeria", 1716, 90, 44)],
        "G": [("Germany", 2042, 562, 2), ("Portugal", 1939, 282, 4),
              ("Ghana", 1701, 110, 37), ("United States", 1781, 70, 13)],
        "H": [("Belgium", 1904, 350, 11), ("Algeria", 1690, 80, 22),
              ("Russia", 1815, 220, 19), ("South Korea", 1717, 80, 57)],
    },
    2018: {
        "A": [("Russia", 1685, 200, 70), ("Saudi Arabia", 1582, 20, 67),
              ("Egypt", 1718, 180, 45), ("Uruguay", 1893, 320, 14)],
        "B": [("Portugal", 1968, 500, 4), ("Spain", 2044, 1000, 10),
              ("Morocco", 1760, 200, 41), ("Iran", 1793, 60, 37)],
        "C": [("France", 1995, 1080, 7), ("Australia", 1714, 50, 36),
              ("Peru", 1906, 80, 11), ("Denmark", 1843, 300, 12)],
        "D": [("Argentina", 1980, 700, 5), ("Iceland", 1787, 80, 22),
              ("Croatia", 1848, 360, 20), ("Nigeria", 1699, 130, 48)],
        "E": [("Brazil", 2131, 951, 2), ("Switzerland", 1879, 250, 6),
              ("Costa Rica", 1745, 50, 23), ("Serbia", 1763, 230, 34)],
        "F": [("Germany", 2077, 883, 1), ("Mexico", 1861, 150, 15),
              ("Sweden", 1796, 150, 24), ("South Korea", 1729, 90, 57)],
        "G": [("Belgium", 1933, 754, 3), ("Panama", 1669, 10, 55),
              ("Tunisia", 1659, 50, 21), ("England", 1948, 874, 13)],
        "H": [("Poland", 1814, 250, 8), ("Senegal", 1747, 280, 27),
              ("Colombia", 1925, 330, 16), ("Japan", 1693, 90, 61)],
    },
    2022: {
        "A": [("Qatar", 1680, 25, 50), ("Ecuador", 1840, 150, 44),
              ("Senegal", 1768, 340, 18), ("Netherlands", 2040, 845, 8)],
        "B": [("England", 1957, 1260, 5), ("Iran", 1797, 65, 20),
              ("United States", 1798, 280, 16), ("Wales", 1791, 170, 19)],
        "C": [("Argentina", 2143, 645, 3), ("Saudi Arabia", 1640, 25, 51),
              ("Mexico", 1843, 180, 13), ("Poland", 1814, 250, 26)],
        "D": [("France", 2005, 1340, 4), ("Australia", 1719, 40, 38),
              ("Denmark", 1971, 350, 10), ("Tunisia", 1687, 60, 30)],
        "E": [("Spain", 2048, 900, 7), ("Costa Rica", 1743, 40, 31),
              ("Germany", 1963, 885, 11), ("Japan", 1798, 180, 24)],
        "F": [("Belgium", 2007, 570, 2), ("Canada", 1776, 200, 41),
              ("Morocco", 1766, 240, 22), ("Croatia", 1922, 360, 12)],
        "G": [("Brazil", 2169, 1140, 1), ("Serbia", 1898, 320, 21),
              ("Switzerland", 1902, 280, 15), ("Cameroon", 1610, 150, 43)],
        "H": [("Portugal", 2006, 936, 9), ("Ghana", 1567, 220, 61),
              ("Uruguay", 1936, 440, 14), ("South Korea", 1786, 130, 28)],
    },
}

# Backtest ground truth: champion + the four semifinalists.
_HISTORICAL_RESULTS: Dict[int, Dict[str, object]] = {
    2010: {"champion": "Spain",
           "semifinalists": {"Spain", "Netherlands", "Germany", "Uruguay"}},
    2014: {"champion": "Germany",
           "semifinalists": {"Germany", "Argentina", "Netherlands", "Brazil"}},
    2018: {"champion": "France",
           "semifinalists": {"France", "Croatia", "Belgium", "England"}},
    2022: {"champion": "Argentina",
           "semifinalists": {"Argentina", "France", "Croatia", "Morocco"}},
}


def features_historical(year: int) -> Dict[str, TeamFeatures]:
    """Feature rows for one historical field (neutral form/minutes/injuries)."""
    field_ = _HISTORICAL_FIELDS[year]
    feats: Dict[str, TeamFeatures] = {}
    for members in field_.values():
        for team, elo, mv, rank in members:
            feats[team] = TeamFeatures(team=team, elo=elo, market_value_m=mv,
                                       fifa_rank=rank)
    return feats


# ==============================================================================
# LAYER 1 — DATA INGESTION & INVENTORY
# ==============================================================================


class Position(str, Enum):
    """Coarse tactical positions used to weight the player graph."""

    GK = "Goalkeeper"
    DEF = "Defender"
    MID = "Midfielder"
    FWD = "Forward"


# Canonical 2026 host venues (16 stadiums across 3 nations). Travel-burden
# indices are derived from each team's host-region anchor relative to these.
HOST_VENUES_2026: Dict[str, Tuple[str, str]] = {
    # venue            -> (city, country)
    "MetLife":         ("New York/New Jersey", "USA"),
    "SoFi":            ("Los Angeles", "USA"),
    "AT&T":            ("Dallas", "USA"),
    "Arrowhead":       ("Kansas City", "USA"),
    "NRG":             ("Houston", "USA"),
    "Mercedes-Benz":   ("Atlanta", "USA"),
    "Levi's":          ("San Francisco Bay", "USA"),
    "Lincoln":         ("Philadelphia", "USA"),
    "Lumen":           ("Seattle", "USA"),
    "GilletteHardRock":("Miami", "USA"),
    "BMO":             ("Toronto", "Canada"),
    "BCPlace":         ("Vancouver", "Canada"),
    "Azteca":          ("Mexico City", "Mexico"),
    "Akron":           ("Guadalajara", "Mexico"),
    "BBVA":            ("Monterrey", "Mexico"),
    "Foro":            ("Boston", "USA"),
}


@dataclass(slots=True)
class Player:
    """
    Individual player matrix.

    Every metric is normalised to [0, 1] so the graph engine can combine them
    without per-feature rescaling:

        form              recent match form (rolling)
        club_performance  season club output (goals/assists/defensive actions)
        club_minutes      share of available club minutes actually played this
                          season (match sharpness; low minutes = rusty)
        fatigue           accumulated load BEFORE kickoff (0 = fresh)
        travel_burden     cumulative travel index for this player at this venue
        base_rating       intrinsic ability (skill ceiling)
        market_value_m    player market value, EUR millions (reporting/teamsheet)
    """

    name: str
    position: Position
    base_rating: float
    form: float
    club_performance: float
    club_minutes: float = 0.70
    fatigue: float = 0.0
    travel_burden: float = 0.0
    market_value_m: float = 0.0
    available: bool = True  # flipped by the ingestion layer on injury reports

    def effective_rating(self) -> float:
        """
        Pre-graph node feature: collapse the matrix into a single scalar.

        Form, club performance and club-minutes sharpness lift the base
        rating; fatigue and travel burden attenuate it. Injured/unavailable
        players contribute nothing.
        """
        if not self.available:
            return 0.0
        sharpness = 0.40 * self.form + 0.35 * self.club_performance + 0.25 * self.club_minutes
        quality = self.base_rating * (0.70 + 0.30 * sharpness)
        attenuation = (1.0 - 0.25 * self.fatigue) * (1.0 - 0.12 * self.travel_burden)
        return float(max(0.0, quality * attenuation))


@dataclass(slots=True)
class Team:
    """A national team: a group placement plus a 23-player squad."""

    name: str
    group: str                       # "A" .. "L"
    squad: List[Player]
    anchor_country: str              # host nation the team is regionally based near
    fifa_points: float = 0.0         # FIFA ranking points (reporting)
    features: Optional[TeamFeatures] = None   # full multi-source feature row
    tactics: Dict[str, float] = field(default_factory=lambda: {"aggression": 1.0, "compactness": 1.0})

    # ---- squad selection ----------------------------------------------------
    def starting_xi(self) -> List[Player]:
        """
        Select a 4-3-3-ish XI from available players, best-by-position.

        Returns exactly 11 players (1 GK, 4 DEF, 3 MID, 3 FWD) padded from the
        bench if a position is short due to injuries.
        """
        avail = [p for p in self.squad if p.available]
        by_pos: Dict[Position, List[Player]] = defaultdict(list)
        for p in avail:
            by_pos[p.position].append(p)
        for pos in by_pos:
            by_pos[pos].sort(key=lambda x: x.effective_rating(), reverse=True)

        target = {Position.GK: 1, Position.DEF: 4, Position.MID: 3, Position.FWD: 3}
        xi: List[Player] = []
        for pos, n in target.items():
            xi.extend(by_pos[pos][:n])

        # Pad from any remaining available players if injuries left us short.
        if len(xi) < 11:
            chosen = set(id(p) for p in xi)
            pool = sorted(
                (p for p in avail if id(p) not in chosen),
                key=lambda x: x.effective_rating(),
                reverse=True,
            )
            xi.extend(pool[: 11 - len(xi)])
        return xi[:11]

    def bench(self) -> List[Player]:
        """Available players not in the current starting XI."""
        xi_ids = {id(p) for p in self.starting_xi()}
        return [p for p in self.squad if p.available and id(p) not in xi_ids]

    # ---- aggregate (pre-graph) ratings --------------------------------------
    def attack_base(self) -> float:
        xi = self.starting_xi()
        atk = [p.effective_rating() for p in xi if p.position in (Position.FWD, Position.MID)]
        return float(np.mean(atk)) if atk else 0.3

    def defense_base(self) -> float:
        xi = self.starting_xi()
        dfd = [p.effective_rating() for p in xi if p.position in (Position.DEF, Position.GK)]
        return float(np.mean(dfd)) if dfd else 0.3


@dataclass
class WorldCupInventory:
    """The authoritative store of teams across groups."""

    teams: Dict[str, Team]  # name -> Team
    n_teams: int = 48
    n_groups: int = 12

    def groups(self) -> Dict[str, List[Team]]:
        out: Dict[str, List[Team]] = defaultdict(list)
        for t in self.teams.values():
            out[t.group].append(t)
        return dict(sorted(out.items()))

    def validate(self) -> None:
        groups = self.groups()
        assert len(self.teams) == self.n_teams, f"Expected {self.n_teams} teams, got {len(self.teams)}"
        assert len(groups) == self.n_groups, f"Expected {self.n_groups} groups, got {len(groups)}"
        for g, members in groups.items():
            assert len(members) == 4, f"Group {g} has {len(members)} teams (expected 4)"


# ---- Async ingestion stub ----------------------------------------------------


def _normalize_position(raw: str) -> Optional[Position]:
    """Map free-text feed positions (e.g. 'Centre-Back', 'CM') onto Position."""
    if not raw:
        return None
    s = raw.lower()
    if "goal" in s or s in ("gk", "g"):
        return Position.GK
    if any(k in s for k in ("back", "def", "cb", "rb", "lb", "wb")):
        return Position.DEF
    if any(k in s for k in ("mid", "cm", "dm", "am", "cdm", "cam")):
        return Position.MID
    if any(k in s for k in ("forward", "striker", "wing", "att", "cf", "st", "fw")):
        return Position.FWD
    return None


class AsyncDataIngestor:
    """
    Asynchronous live data-ingestion layer — **key-free, no API token**.

      * fetch_team_summaries() -> Wikipedia REST `page/summary` (no key) for
        each national team, concurrently. Confirms liveness and enriches team
        metadata (the summary `extract`).
      * fetch_injury_reports() -> configurable open JSON endpoints (or, in
        production, a `playwright`-rendered media page — sketch below),
        normalised to the {team, player, status} shape apply_injury_reports()
        consumes.

    Strength grounding now comes from the multi-source FeatureStore (Elo +
    market value + SPI + FIFA); this layer refreshes metadata/injuries on
    top. The HTTP client resolves httpx -> requests-on-a-thread -> offline
    no-op, and network failure is never fatal.
    """

    # Wikipedia REST summary endpoint — open, no key, returns JSON with `extract`.
    WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def __init__(
        self,
        inventory: WorldCupInventory,
        injury_sources: Optional[Sequence[str]] = None,
        *,
        concurrency: int = 8,
        timeout: float = 15.0,
        max_retries: int = 3,
    ):
        self.inventory = inventory
        self.injury_sources = list(injury_sources or [])
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries

    # -- low-level concurrent GET with retry/backoff --------------------------
    async def _get_json(self, url: str, params: Optional[Dict[str, str]] = None) -> Optional[dict]:
        """
        One resilient GET returning parsed JSON, or None on failure.

        Transient failures (network errors, 5xx/timeout) are retried with
        exponential backoff; a *deterministic* failure (non-JSON body) is NOT
        retried — re-requesting the same URL would yield the same bad body.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                if _HTTPX_AVAILABLE:
                    async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                        resp = await client.get(url, params=params)
                        resp.raise_for_status()
                        body = resp.text
                elif _REQUESTS_AVAILABLE:
                    def _blocking() -> str:                 # run blocking IO off-loop
                        r = requests.get(url, params=params, timeout=self.timeout)
                        r.raise_for_status()
                        return r.text

                    body = await asyncio.to_thread(_blocking)
                else:
                    LOGGER.warning("No HTTP client installed; ingestion is offline.")
                    return None
            except Exception as exc:  # network / status — transient, retry
                backoff = 0.5 * (2 ** (attempt - 1))
                LOGGER.warning("GET %s failed (attempt %d/%d): %s", url, attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff)
                continue
            # Got a body. Parse once — a parse failure is deterministic, no retry.
            try:
                return json.loads(body)
            except ValueError as exc:
                LOGGER.warning("GET %s returned non-JSON body (%s); not retrying.", url, exc)
                return None
        return None

    # -- team metadata ingestion (key-free) -----------------------------------
    async def _fetch_one_summary(self, team_name: str, sem: asyncio.Semaphore) -> Tuple[str, str]:
        """Fetch one team's Wikipedia summary extract."""
        title = quote(f"{team_name} national football team")
        async with sem:
            data = await self._get_json(self.WIKI_SUMMARY + title)
        return team_name, (data or {}).get("extract", "")

    async def fetch_team_summaries(self, team_names: Optional[Sequence[str]] = None) -> Dict[str, str]:
        """Concurrently fetch open Wikipedia summaries for the requested teams."""
        names = list(team_names or self.inventory.teams.keys())
        sem = asyncio.Semaphore(self.concurrency)
        LOGGER.info("Fetching key-free team summaries for %d team(s) from Wikipedia REST...", len(names))
        results = await asyncio.gather(*(self._fetch_one_summary(n, sem) for n in names))
        return {name: extract for name, extract in results}

    # -- injury ingestion ------------------------------------------------------
    async def fetch_injury_reports(self) -> List[Dict[str, str]]:
        """
        Concurrently fetch + normalise media injury reports from open feeds.

        Each source is expected to return JSON of shape
        ``{"reports": [{"team": ..., "player": ..., "status": "out"|"doubt"}]}``.

        Production playwright sketch (JS-rendered pages, still key-free):

            from playwright.async_api import async_playwright
            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                rows = await page.query_selector_all(".injury-row")
                ...parse rows into the same {team, player, status} shape...
        """
        if not self.injury_sources:
            LOGGER.info("No injury sources configured; skipping injury ingestion.")
            return []
        payloads = await asyncio.gather(*(self._get_json(u) for u in self.injury_sources))
        reports: List[Dict[str, str]] = []
        for payload in payloads:
            for rep in (payload or {}).get("reports", []) or []:
                if rep.get("team") and rep.get("player"):
                    reports.append(rep)
        LOGGER.info("Ingested %d injury report(s).", len(reports))
        return reports

    # -- reconciliation back onto the inventory --------------------------------
    def apply_injury_reports(self, reports: List[Dict[str, str]]) -> None:
        """Flip availability / bump fatigue for named players from a feed."""
        for rep in reports:
            team = self.inventory.teams.get(rep.get("team", ""))
            if not team:
                continue
            for p in team.squad:
                if p.name == rep.get("player"):
                    status = rep.get("status", "out")
                    if status == "out":
                        p.available = False
                    elif status == "doubt":
                        p.fatigue = min(1.0, p.fatigue + 0.30)
                    LOGGER.info("Applied report: %s -> %s (%s)", p.name, status, team.name)

    # -- convenience orchestrator ---------------------------------------------
    async def refresh(self, team_names: Optional[Sequence[str]] = None) -> Dict[str, str]:
        """Fetch summaries + injuries concurrently; apply injuries; return summaries."""
        summaries, reports = await asyncio.gather(
            self.fetch_team_summaries(team_names),
            self.fetch_injury_reports(),
        )
        self.apply_injury_reports(reports)
        return summaries


# ==============================================================================
# LAYER 2 — GRAPH SPATIO-TEMPORAL MATCH ENGINE (live-event edition)
# ==============================================================================


@dataclass(slots=True)
class TacticalModifiers:
    """Symmetrical multiplicative modifiers handed down from the macro layer."""

    attack_mult: float = 1.0
    defense_mult: float = 1.0
    rationale: str = "baseline"

    def clamp(self) -> "TacticalModifiers":
        """Keep modifiers in a defensible band so the LLM can't break physics."""
        self.attack_mult = float(np.clip(self.attack_mult, 0.70, 1.40))
        self.defense_mult = float(np.clip(self.defense_mult, 0.70, 1.40))
        return self


class PlayerGraph:
    """
    Per-side spatio-temporal graph and a *simulated* GNN message-passing pass.

    Nodes  : the on-pitch players (feature = effective_rating()). Normally 11,
             but the graph happily shrinks to 10/9 nodes after sendings-off —
             losing a node both removes its quality and weakens the diffusion
             structure, which is exactly the cohesion story we want.
    Edges  : passing lanes + pressing structure, encoded in a weighted
             adjacency matrix A. Same-line links (DEF-DEF, MID-MID) and
             adjacent-line links (DEF-MID, MID-FWD) are stronger; long DEF-FWD
             links are weak. Edge weights are modulated by team aggression
             (pressing) and compactness (defensive shape).

    The "GNN" runs K rounds of normalised neighbour aggregation:

        h^{k+1} = (1 - alpha) * h^k + alpha * D^{-1} A h^k

    which diffuses quality across well-connected nodes — a vectorised proxy
    for how cohesion (not just raw talent) drives chance creation. The
    converged node vector is reduced to attack/defense cohesion scalars.
    """

    def __init__(self, team: Team, xi: Optional[List[Player]] = None):
        self.team = team
        self.xi = list(xi) if xi is not None else team.starting_xi()
        self.h0 = np.array([p.effective_rating() for p in self.xi], dtype=np.float64)
        self.node_pos = [p.position for p in self.xi]
        self.adj = self._build_adjacency()

    def _build_adjacency(self) -> np.ndarray:
        """Construct the weighted, symmetric passing/pressing adjacency matrix."""
        n = len(self.xi)
        line_rank = {Position.GK: 0, Position.DEF: 1, Position.MID: 2, Position.FWD: 3}
        aggression = self.team.tactics.get("aggression", 1.0)
        compactness = self.team.tactics.get("compactness", 1.0)

        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                gap = abs(line_rank[self.node_pos[i]] - line_rank[self.node_pos[j]])
                if gap == 0:
                    w = 1.0 * compactness          # same line: strong shape link
                elif gap == 1:
                    w = 0.8 * aggression           # adjacent lines: passing lanes
                elif gap == 2:
                    w = 0.4
                else:
                    w = 0.15                        # GK<->FWD: weak
                A[i, j] = A[j, i] = w
        return A

    def message_passing(self, rounds: int = 3, alpha: float = 0.5) -> np.ndarray:
        """Run the diffusion and return converged node features."""
        deg = self.adj.sum(axis=1)
        deg[deg == 0] = 1.0
        d_inv = np.diag(1.0 / deg)
        propagate = d_inv @ self.adj            # row-normalised neighbour mean

        h = self.h0.copy()
        for _ in range(rounds):
            h = (1.0 - alpha) * h + alpha * (propagate @ h)
        return h

    def cohesion_ratings(self) -> Tuple[float, float]:
        """
        Reduce the converged graph to (attack_cohesion, defense_cohesion).

        Attack cohesion weights MID/FWD nodes; defense cohesion weights GK/DEF.
        Because diffusion mixes in neighbours, an isolated superstar contributes
        less than a well-connected unit of equal raw talent.
        """
        h = self.message_passing()
        atk_idx = [i for i, p in enumerate(self.node_pos) if p in (Position.MID, Position.FWD)]
        dfd_idx = [i for i, p in enumerate(self.node_pos) if p in (Position.GK, Position.DEF)]
        atk = float(np.mean(h[atk_idx])) if atk_idx else 0.3
        dfd = float(np.mean(h[dfd_idx])) if dfd_idx else 0.3
        return atk, dfd


@dataclass(slots=True)
class MatchResult:
    """Outcome of a single 0–120' + (optional) shootout match."""

    home: str
    away: str
    home_goals: int
    away_goals: int
    went_to_pens: bool = False
    home_pens: int = 0
    away_pens: int = 0
    timeline: List[Tuple[int, str]] = field(default_factory=list)   # (minute, scorer team)
    events: List[Tuple[int, str, str]] = field(default_factory=list)  # (minute, team, event)

    @property
    def winner(self) -> Optional[str]:
        """Knockout winner (None only if used as a group match left as a draw)."""
        if self.home_goals != self.away_goals:
            return self.home if self.home_goals > self.away_goals else self.away
        if self.went_to_pens:
            return self.home if self.home_pens > self.away_pens else self.away
        return None  # genuine group-stage draw


@dataclass(frozen=True)
class EngineConfig:
    """
    Tunable parameters of the goal process (the calibration target surface).

    base_xg            : expected goals / 90' for a perfectly balanced top side.
    fatigue_rate       : per-minute TEAM-level fatigue feeding the lambda model.
    match_load_rate    : per-minute PER-PLAYER positional load driving the
                         substitution threshold monitor (faster than the
                         team-level rate so starters realistically cross the
                         0.80 line around minute 60-80).
    fatigue_penalty    : how strongly accumulated fatigue attenuates attack.
    red_card_p_per_min : per-team per-minute sending-off hazard (~0.001 gives
                         ~0.09 reds/team/match ≈ 0.18/match, the modern WC rate).
    red_attack_drop    : permanent multiplicative attack-cohesion drop per red.
    red_defense_drop   : permanent multiplicative defense-cohesion drop per red.
    sub_fatigue_threshold : in-match fatigue level that triggers an automatic
                         substitution for a starter.
    max_subs           : substitution windows per team (FIFA: 5, +1 in ET —
                         modelled as a flat 5 here).

    Frozen + plain (no slots) so instances pickle cleanly across the process
    pool used by the Monte Carlo engine.
    """

    base_xg: float = 1.40
    fatigue_rate: float = 0.0035
    match_load_rate: float = 0.0080
    fatigue_penalty: float = 0.45
    red_card_p_per_min: float = 0.0010
    red_attack_drop: float = 0.80
    red_defense_drop: float = 0.85
    sub_fatigue_threshold: float = 0.80
    max_subs: int = 5


class MatchTeamState:
    """
    Mutable per-team, per-match state for the live-event minute loop.

    Tracks the on-pitch XI, the bench, per-player in-match fatigue, red cards
    (with their permanent compounding cohesion penalty), and substitutions.
    Cohesion is recomputed through the GNN layer only when the on-pitch
    personnel actually change (red card or substitution), keeping the Monte
    Carlo loop fast.
    """

    def __init__(self, team: Team, config: EngineConfig):
        self.team = team
        self.config = config
        self.xi: List[Player] = team.starting_xi()
        self.bench: List[Player] = [p for p in team.squad
                                    if p.available and id(p) not in {id(q) for q in self.xi}]
        # In-match fatigue per on-pitch player, seeded from pre-match fatigue.
        # Slightly noiseless per-player load factor: defenders/GK churn less.
        self.match_fatigue: Dict[int, float] = {id(p): p.fatigue for p in self.xi}
        # Team-level fatigue for the lambda model (original calibration semantics).
        self._lambda_fatigue: float = float(np.mean([p.fatigue for p in self.xi]))
        self.subs_used = 0
        self.red_cards = 0
        self.red_mult_attack = 1.0
        self.red_mult_defense = 1.0
        self._recompute_cohesion()

    # -- cohesion --------------------------------------------------------------
    def _recompute_cohesion(self) -> None:
        """Re-run the graph neural layer over the CURRENT on-pitch personnel."""
        atk, dfd = PlayerGraph(self.team, xi=self.xi).cohesion_ratings()
        self.attack_cohesion = atk * self.red_mult_attack
        self.defense_cohesion = dfd * self.red_mult_defense

    # -- per-minute bookkeeping --------------------------------------------------
    _LOAD_FACTOR = {Position.GK: 0.35, Position.DEF: 0.90,
                    Position.MID: 1.10, Position.FWD: 1.00}

    def tick_fatigue(self) -> None:
        """Accumulate one minute of load: per-player (subs) + team (lambdas)."""
        rate = self.config.match_load_rate
        for p in self.xi:
            self.match_fatigue[id(p)] = min(
                1.0, self.match_fatigue[id(p)] + rate * self._LOAD_FACTOR[p.position]
            )
        self._lambda_fatigue = min(1.0, self._lambda_fatigue + self.config.fatigue_rate)

    def team_fatigue(self) -> float:
        """Team-level fatigue driving the lambda model (calibration-stable)."""
        return self._lambda_fatigue

    # -- live events ------------------------------------------------------------
    def apply_red_card(self, minute: int, rng: np.random.Generator,
                       result: MatchResult) -> None:
        """
        Sending-off: remove a random outfield player (GKs are spared for
        simplicity) and apply a PERMANENT, COMPOUNDING penalty to both
        cohesion channels for the remainder of the match.
        """
        outfield = [p for p in self.xi if p.position != Position.GK]
        if not outfield:
            return
        victim = outfield[int(rng.integers(len(outfield)))]
        self.xi = [p for p in self.xi if id(p) != id(victim)]
        self.red_cards += 1
        # Compounding: a second red multiplies the penalty again.
        self.red_mult_attack *= self.config.red_attack_drop
        self.red_mult_defense *= self.config.red_defense_drop
        self._recompute_cohesion()
        result.events.append((minute, self.team.name, f"RED CARD {victim.name}"))

    def check_substitutions(self, minute: int, result: MatchResult) -> None:
        """
        Tactical threshold monitor: any starter whose accumulated in-match
        fatigue crosses the threshold is replaced by the highest-rated
        positional bench option, and the GNN layer is re-run so the match
        lambdas refresh instantly.
        """
        if self.subs_used >= self.config.max_subs or not self.bench:
            return
        changed = False
        for p in list(self.xi):
            if self.subs_used >= self.config.max_subs or not self.bench:
                break
            if self.match_fatigue[id(p)] < self.config.sub_fatigue_threshold:
                continue
            same_pos = [b for b in self.bench if b.position == p.position]
            pool = same_pos or self.bench
            sub = max(pool, key=lambda b: b.effective_rating())
            self.bench.remove(sub)
            self.xi[self.xi.index(p)] = sub
            self.match_fatigue[id(sub)] = sub.fatigue   # fresh legs
            # Fresh legs relieve ~1/11 of the team-level fatigue load too.
            self._lambda_fatigue = max(sub.fatigue, self._lambda_fatigue * (10.0 / 11.0))
            self.subs_used += 1
            changed = True
            result.events.append(
                (minute, self.team.name, f"SUB {sub.name} for {p.name}")
            )
        if changed:
            self._recompute_cohesion()


class MatchSimulator:
    """
    Micro-simulation engine: minute-by-minute Attenuated Poisson goal process
    with LIVE MATCH EVENTS.

    For each minute t we compute per-side lambda (expected goals in that
    minute) from:

        * CURRENT graph cohesion attack vs opponent CURRENT cohesion defense
          (recomputed through the GNN whenever personnel change),
        * compound per-player positional fatigue accumulating on the pitch,
        * symmetrical tactical modifiers (possibly LLM-adjusted at halftime),
        * permanent compounding red-card penalties,
        * automatic fatigue-threshold substitutions.

    Goals in the minute are drawn from Poisson(lambda_minute). Knockout ties
    are resolved by 30' extra time (same process) then penalties.
    """

    SUB_CHECK_EVERY = 3   # minutes between substitution-threshold sweeps

    def __init__(self, rng: Optional[np.random.Generator] = None, config: Optional[EngineConfig] = None):
        self.rng = rng or np.random.default_rng()
        self.config = config or EngineConfig()
        self.BASE_XG_PER_MATCH = self.config.base_xg
        self.FATIGUE_RATE_PER_MIN = self.config.fatigue_rate
        self.FATIGUE_ATTACK_PENALTY = self.config.fatigue_penalty

    # -- core lambda model -----------------------------------------------------
    def _minute_lambda(
        self,
        atk_cohesion: float,
        opp_def_cohesion: float,
        fatigue: float,
        mods: TacticalModifiers,
    ) -> float:
        """
        Expected goals for ONE side in ONE minute.

        Attack/defence are mapped through a logistic-style ratio so that a
        stronger attack vs a weaker defence raises xG smoothly without blowing
        up. The caller supplies the side's CURRENT mean in-match fatigue.
        """
        ratio = (atk_cohesion + 0.05) / (opp_def_cohesion + 0.05)
        ratio = float(np.clip(ratio, 0.40, 2.20))

        fatigue_factor = 1.0 - self.FATIGUE_ATTACK_PENALTY * min(1.0, fatigue)

        per_match = (
            self.BASE_XG_PER_MATCH
            * ratio
            * fatigue_factor
            * mods.attack_mult
            / mods.defense_mult   # opponent's defensive posture
        )
        return max(0.0, per_match) / 90.0  # convert to per-minute rate

    # -- a single playable period (live-event loop) ----------------------------
    def _simulate_period(
        self,
        minutes: Sequence[int],
        state_a: MatchTeamState,
        state_b: MatchTeamState,
        mods_a: TacticalModifiers,
        mods_b: TacticalModifiers,
        result: MatchResult,
    ) -> Tuple[int, int]:
        """Play a contiguous block of minutes, mutating states + `result`."""
        ha = 0
        hb = 0
        p_red = self.config.red_card_p_per_min
        for t in minutes:
            # -- live edge-case events first: sendings-off --------------------
            if self.rng.random() < p_red:
                state_a.apply_red_card(t, self.rng, result)
            if self.rng.random() < p_red:
                state_b.apply_red_card(t, self.rng, result)

            # -- goal process on CURRENT cohesion/fatigue ----------------------
            lam_a = self._minute_lambda(state_a.attack_cohesion,
                                        state_b.defense_cohesion,
                                        state_a.team_fatigue(), mods_a)
            lam_b = self._minute_lambda(state_b.attack_cohesion,
                                        state_a.defense_cohesion,
                                        state_b.team_fatigue(), mods_b)
            ga_goals = int(self.rng.poisson(lam_a))
            gb_goals = int(self.rng.poisson(lam_b))
            if ga_goals:
                ha += ga_goals
                result.timeline.append((t, result.home))
            if gb_goals:
                hb += gb_goals
                result.timeline.append((t, result.away))

            # -- per-player positional load + threshold substitutions ----------
            state_a.tick_fatigue()
            state_b.tick_fatigue()
            if t % self.SUB_CHECK_EVERY == 0:
                state_a.check_substitutions(t, result)
                state_b.check_substitutions(t, result)
        return ha, hb

    # -- penalty shootout ------------------------------------------------------
    def _shootout(self, team_a: Team, team_b: Team) -> Tuple[int, int]:
        """
        Best-of-5 then sudden death. Conversion probability is anchored on each
        side's attacking cohesion vs the opposing keeper quality.
        """
        def conv_prob(att: Team, opp: Team) -> float:
            gk = next((p for p in opp.starting_xi() if p.position == Position.GK), None)
            gk_q = gk.effective_rating() if gk else 0.5
            base = 0.78 + 0.10 * (att.attack_base() - gk_q)
            return float(np.clip(base, 0.55, 0.92))

        pa, pb = conv_prob(team_a, team_b), conv_prob(team_b, team_a)
        sa = sb = 0
        for _ in range(5):
            sa += int(self.rng.random() < pa)
            sb += int(self.rng.random() < pb)
        while sa == sb:  # sudden death
            sa += int(self.rng.random() < pa)
            sb += int(self.rng.random() < pb)
        return sa, sb

    # -- public entrypoint -----------------------------------------------------
    def simulate(
        self,
        team_a: Team,
        team_b: Team,
        knockout: bool = False,
        tactical_agent: Optional["TacticalManagerAgent"] = None,
        verbose: bool = False,
    ) -> MatchResult:
        """
        Simulate a full match with live events.

        If `tactical_agent` is supplied, it is consulted at halftime to adjust
        the second-half lambdas (macro -> micro feedback loop).
        """
        cfg = self.config
        state_a = MatchTeamState(team_a, cfg)
        state_b = MatchTeamState(team_b, cfg)

        mods_a = TacticalModifiers()
        mods_b = TacticalModifiers()
        result = MatchResult(home=team_a.name, away=team_b.name, home_goals=0, away_goals=0)

        # --- First half (1'..45') ---
        h1a, h1b = self._simulate_period(range(1, 46), state_a, state_b, mods_a, mods_b, result)

        # --- HALFTIME: macro tactical layer adjusts H2 lambdas ---
        if tactical_agent is not None:
            mods_a = tactical_agent.evaluate(team_a, team_b, h1a, h1b, half=1).clamp()
            mods_b = tactical_agent.evaluate(team_b, team_a, h1b, h1a, half=1).clamp()
            if verbose:
                LOGGER.info("HT %s mods: %s", team_a.name, mods_a)
                LOGGER.info("HT %s mods: %s", team_b.name, mods_b)

        # --- Second half (46'..90'); fatigue/reds/subs carry inside the states ---
        h2a, h2b = self._simulate_period(range(46, 91), state_a, state_b, mods_a, mods_b, result)

        result.home_goals = h1a + h2a
        result.away_goals = h1b + h2b

        # --- Knockout resolution: extra time then penalties ---
        if knockout and result.home_goals == result.away_goals:
            eta, etb = self._simulate_period(range(91, 121), state_a, state_b,
                                             mods_a, mods_b, result)
            result.home_goals += eta
            result.away_goals += etb
            if result.home_goals == result.away_goals:
                result.went_to_pens = True
                result.home_pens, result.away_pens = self._shootout(team_a, team_b)

        if verbose:
            for minute, team, ev in result.events:
                LOGGER.info("  %d' [%s] %s", minute, team, ev)
            LOGGER.info(
                "FT %s %d-%d %s%s",
                team_a.name, result.home_goals, result.away_goals, team_b.name,
                f" (pens {result.home_pens}-{result.away_pens})" if result.went_to_pens else "",
            )
        return result


# ==============================================================================
# LAYER 3 — LLM TACTICAL AGENT (local vLLM on the RunPod 4090)
# ==============================================================================


class VectorNewsStore:
    """
    A lightweight local "vector store" of mock 2026 news items.

    To stay dependency-free, semantic retrieval is approximated with a
    bag-of-words cosine similarity over a token vocabulary — the same
    query/retrieve contract a real vector DB (FAISS, Chroma, pgvector) exposes,
    so it can be swapped without changing the agent.
    """

    def __init__(self, items: List[Dict[str, str]]):
        self.items = items
        self._vocab: Dict[str, int] = {}
        self._matrix = self._vectorize([it["text"] for it in items])

    def _tokens(self, text: str) -> List[str]:
        return [w.strip(".,!?;:").lower() for w in text.split() if w.strip(".,!?;:")]

    def _vectorize(self, texts: List[str]) -> np.ndarray:
        for text in texts:
            for tok in self._tokens(text):
                self._vocab.setdefault(tok, len(self._vocab))
        mat = np.zeros((len(texts), len(self._vocab)), dtype=np.float64)
        for r, text in enumerate(texts):
            for tok in self._tokens(text):
                mat[r, self._vocab[tok]] += 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Return the top_k most semantically similar news items."""
        q = np.zeros(len(self._vocab), dtype=np.float64)
        for tok in self._tokens(text):
            if tok in self._vocab:
                q[self._vocab[tok]] += 1.0
        n = np.linalg.norm(q)
        if n == 0:
            return []
        sims = self._matrix @ (q / n)
        order = np.argsort(sims)[::-1][:top_k]
        return [self.items[i] for i in order if sims[i] > 0]


def build_default_news_store() -> VectorNewsStore:
    """Mock 2026 tournament feed used by the tactical agent."""
    return VectorNewsStore([
        {"team": "France", "text": "Kylian Mbappe hamstring tight in training, managed minutes expected."},
        {"team": "Argentina", "text": "Argentina pivoting to an aggressive low-block counter-attack scheme."},
        {"team": "Brazil", "text": "Brazil full-backs bombing forward, leaving space in transition defensively."},
        {"team": "England", "text": "England midfield press disrupted, conceding chances on the second ball."},
        {"team": "Spain", "text": "Spain dominating possession, opponents sitting deep against the tiki-taka."},
        {"team": "Germany", "text": "Germany high defensive line repeatedly exposed by pace in behind."},
        {"team": "Portugal", "text": "Portugal energised after substitutions, increasing attacking tempo late."},
        {"team": "USA", "text": "USA riding home support with intense first-half pressing energy."},
    ])


class TacticalManagerAgent:
    """
    Macro-context tactical agent.

    At halftime it (1) retrieves relevant news for the team, (2) summarises the
    live match state, and (3) emits multiplicative lambda modifiers for H2.

    Backend selection:
        use_llm=True + `openai` package installed + endpoint reachable
            -> chat-completions call against WC_LLM_BASE_URL. On the RunPod
               4090 this is a LOCAL vLLM/Ollama server on the loopback
               (default http://127.0.0.1:8000/v1, model Qwen3-14B-AWQ); set
               the env vars to target any other OpenAI-compatible endpoint.
        otherwise
            -> deterministic local heuristic (default for Monte Carlo).

    PROCESS SAFETY: the OpenAI client owns an HTTP connection pool, and pools
    must never be shared across fork boundaries (socket FD collisions). The
    client here is therefore constructed LAZILY on first use *inside the
    current process*, and is stripped on pickling (__getstate__/__setstate__),
    so any worker that receives this object rebuilds its own connection pool.
    """

    def __init__(self, news_store: VectorNewsStore, use_llm: bool = False):
        self.news = news_store
        self.use_llm = use_llm and _OPENAI_AVAILABLE
        self._client = None         # lazily built per-process; never pickled
        self._llm_failed = False    # circuit breaker after repeated failures
        self._llm_fail_streak = 0   # consecutive failures before tripping it
        if use_llm and not _OPENAI_AVAILABLE:
            LOGGER.warning("LLM requested but `openai` package missing; using local heuristic.")

    # -- process-safe (de)serialisation ----------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None        # never ship a socket pool across a fork
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client = None

    def _get_client(self):
        """Lazy per-process client construction (fork-safe socket init)."""
        if self._client is None:
            import http.cookiejar

            import httpx

            class _BlockCookies(http.cookiejar.CookiePolicy):
                # api.openai.com (Cloudflare) re-sets cookies on every response;
                # a long-lived client accumulates them until the server rejects
                # requests with 431 Request Header Fields Too Large.
                netscape = True
                rfc2965 = hide_cookie2 = False

                def set_ok(self, cookie, request):
                    return False

                def return_ok(self, cookie, request):
                    return False

            http_client = httpx.Client(
                timeout=LLM_TIMEOUT_S,
                cookies=http.cookiejar.CookieJar(policy=_BlockCookies()),
            )
            self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                                  timeout=LLM_TIMEOUT_S, max_retries=1,
                                  http_client=http_client)
        return self._client

    # -- public ----------------------------------------------------------------
    def evaluate(self, team: Team, opponent: Team, gf: int, ga: int, half: int) -> TacticalModifiers:
        """Return tactical modifiers for `team`'s next period."""
        state = (
            f"Halftime. {team.name} {gf}-{ga} {opponent.name}. "
            f"{team.name} attack_base={team.attack_base():.2f} "
            f"defense_base={team.defense_base():.2f}."
        )
        news = self.news.query(f"{team.name} {opponent.name} tactics injury press")
        if self.use_llm and not self._llm_failed:
            try:
                mods = self._evaluate_llm(team, opponent, state, news)
                self._llm_fail_streak = 0
                return mods
            except Exception as exc:  # pragma: no cover - network/runtime guard
                self._llm_fail_streak += 1
                self._client = None   # rebuild the HTTP pool on the next attempt
                if self._llm_fail_streak >= 3:
                    self._llm_failed = True   # don't hammer a dead endpoint per-minute
                    LOGGER.warning("LLM tactical call failed %d times in a row (%s); "
                                   "disabling LLM for this worker.",
                                   self._llm_fail_streak, exc)
                else:
                    LOGGER.warning("LLM tactical call failed (%s); will retry at the "
                                   "next halftime.", exc)
        return self._evaluate_heuristic(team, gf, ga, news)

    # -- local-LLM backend (OpenAI chat-completions protocol) -------------------
    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Robustly pull the first JSON object out of a completion.

        Reasoning models (Qwen3, R1 distills) may emit <think>...</think>
        blocks or markdown fences before the payload; strip both.
        """
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"```(?:json)?", "", text)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in completion: {text[:120]!r}")
        return json.loads(match.group(0))

    def _evaluate_llm(
        self, team: Team, opponent: Team, state: str, news: List[Dict[str, str]]
    ) -> TacticalModifiers:
        news_block = "\n".join(f"- {n['text']}" for n in news) or "- (no relevant news)"
        prompt = (
            "You are an elite football tactical analyst adjusting a team at halftime.\n"
            f"MATCH STATE:\n{state}\n\n"
            f"RELEVANT NEWS / INTEL:\n{news_block}\n\n"
            f"Decide second-half tactical modifiers for {team.name} versus {opponent.name}. "
            "Chase the game when trailing, protect a lead when ahead, and exploit any "
            "opponent weakness named in the intel.\n\n"
            "Respond with ONLY a JSON object, no prose, of the exact shape:\n"
            '{"attack_mult": <float in [0.70, 1.40]>, '
            '"defense_mult": <float in [0.70, 1.40]>, '
            '"rationale": "<one sentence>"}'
        )
        client = self._get_client()
        kwargs = dict(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        try:
            # vLLM + OpenAI both accept JSON mode; some Ollama builds do not.
            resp = client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs
            )
        except Exception:
            resp = client.chat.completions.create(**kwargs)
        data = self._extract_json(resp.choices[0].message.content or "")
        return TacticalModifiers(
            attack_mult=float(data["attack_mult"]),
            defense_mult=float(data["defense_mult"]),
            rationale=str(data.get("rationale", "llm")),
        ).clamp()

    # -- deterministic local backend (default) ---------------------------------
    def _evaluate_heuristic(
        self, team: Team, gf: int, ga: int, news: List[Dict[str, str]]
    ) -> TacticalModifiers:
        """
        Closed-form analogue of the LLM's reasoning — fast and parallel-safe.

        Game-state logic + simple keyword intel parsing on the retrieved news.
        """
        mods = TacticalModifiers(rationale="heuristic")
        diff = gf - ga
        if diff < 0:            # trailing -> chase
            mods.attack_mult, mods.defense_mult = 1.18, 0.92
            mods.rationale = "trailing: push attack"
        elif diff > 0:          # leading -> manage
            mods.attack_mult, mods.defense_mult = 0.94, 1.12
            mods.rationale = "leading: protect lead"

        # Fold in retrieved intel (the "vector store" feedback).
        for item in news:
            if item.get("team") != team.name:
                continue
            txt = item["text"].lower()
            if any(k in txt for k in ("hamstring", "injury", "tight", "fatigue")):
                mods.attack_mult *= 0.95     # key player managed
                mods.rationale += " | injury caution"
            if any(k in txt for k in ("aggressive", "press", "pressing", "tempo")):
                mods.attack_mult *= 1.05
                mods.rationale += " | aggression"
            if "low-block" in txt or "compact" in txt:
                mods.defense_mult *= 1.06
                mods.rationale += " | compact block"
        return mods.clamp()


# ==============================================================================
# LAYER 4 — TOURNAMENT SCHEDULERS & RESOLVERS
# ==============================================================================


@dataclass(slots=True)
class GroupRecord:
    """Running record for one team within its group."""

    team: str
    group: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def points(self) -> int:
        return 3 * self.won + self.drawn

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def apply(self, scored: int, conceded: int) -> None:
        self.played += 1
        self.gf += scored
        self.ga += conceded
        if scored > conceded:
            self.won += 1
        elif scored == conceded:
            self.drawn += 1
        else:
            self.lost += 1

    # Sort key: points -> goal difference -> goals for (descending each).
    def sort_key(self) -> Tuple[int, int, int]:
        return (self.points, self.gd, self.gf)


# Per-process cache of REAL played 2026 results (WC_RESULTS_CSV). Matches in
# this table are pinned to their actual scoreline instead of being simulated,
# so Monte Carlo probabilities condition on the tournament as it happens.
_RESULTS_CSV_CACHE: Optional[Dict[frozenset, Dict[str, str]]] = None


def _fixed_results() -> Dict[frozenset, Dict[str, str]]:
    """{frozenset({home, away}): row} of real played matches ({} if no CSV)."""
    global _RESULTS_CSV_CACHE
    if _RESULTS_CSV_CACHE is None:
        out: Dict[frozenset, Dict[str, str]] = {}
        path = os.getenv("WC_RESULTS_CSV", "")
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    out[frozenset((row["home"].strip(), row["away"].strip()))] = row
            LOGGER.info("Pinned real 2026 results: %d match(es) from %s", len(out), path)
        _RESULTS_CSV_CACHE = out
    return _RESULTS_CSV_CACHE


class Tournament:
    """
    Full 2026 tournament resolver: group stage -> 3rd-place matrix -> knockouts.

    A fresh `np.random.Generator` is injected per tournament so Monte Carlo
    iterations are independent and reproducible from a seed.
    """

    # Real played results are pinned only for the live 2026 field; the
    # historical backtest (Tournament32) must keep simulating every match.
    PIN_REAL_RESULTS = True

    def __init__(
        self,
        inventory: WorldCupInventory,
        tactical_agent: Optional[TacticalManagerAgent] = None,
        rng: Optional[np.random.Generator] = None,
        config: Optional[EngineConfig] = None,
        record: bool = False,
    ):
        self.inv = inventory
        self.agent = tactical_agent
        self.rng = rng or np.random.default_rng()
        self.sim = MatchSimulator(self.rng, config=config)
        # Telemetry: furthest stage reached per team this tournament.
        self.reached: Dict[str, str] = {}
        # Optional full play-by-play recording (populated for report runs).
        self.record = record
        self.match_log: List[Dict] = []
        self.group_standings: Dict[str, List[GroupRecord]] = {}

    def _log_match(self, stage: str, res: "MatchResult", group: str = "") -> None:
        """Append a structured row to match_log when recording is enabled."""
        if not self.record:
            return
        self.match_log.append({
            "stage": stage, "group": group,
            "home": res.home, "away": res.away,
            "hg": res.home_goals, "ag": res.away_goals,
            "pens": (res.home_pens, res.away_pens) if res.went_to_pens else None,
            "winner": res.winner,
            "events": list(res.events),
        })

    # -- group stage -----------------------------------------------------------
    def _round_robin_pairs(self, n: int = 4) -> List[Tuple[int, int]]:
        """All 6 unordered pairings for a 4-team group (each plays 3)."""
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def _fixed_or_simulate(self, ta: Team, tb: Team, knockout: bool) -> MatchResult:
        """Use the real recorded scoreline when this pairing was already played."""
        row = _fixed_results().get(frozenset((ta.name, tb.name))) if self.PIN_REAL_RESULTS else None
        if row is None:
            return self.sim.simulate(ta, tb, knockout=knockout, tactical_agent=self.agent)
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        if row["home"].strip() != ta.name:
            hg, ag = ag, hg
        res = MatchResult(home=ta.name, away=tb.name, home_goals=hg, away_goals=ag)
        if knockout and hg == ag:
            # Recorded knockout draw decided on penalties: a `winner` column
            # is required to break the tie.
            won = (row.get("winner") or "").strip()
            res.went_to_pens = True
            res.home_pens, res.away_pens = (1, 0) if won == ta.name else (0, 1)
        return res

    def play_group_stage(self) -> Dict[str, List[GroupRecord]]:
        standings: Dict[str, List[GroupRecord]] = {}
        for group, teams in self.inv.groups().items():
            recs = {t.name: GroupRecord(team=t.name, group=group) for t in teams}
            for i, j in self._round_robin_pairs(len(teams)):
                ta, tb = teams[i], teams[j]
                res = self._fixed_or_simulate(ta, tb, knockout=False)
                recs[ta.name].apply(res.home_goals, res.away_goals)
                recs[tb.name].apply(res.away_goals, res.home_goals)
                self._log_match("GROUP", res, group=group)
            ranked = sorted(recs.values(), key=lambda r: r.sort_key(), reverse=True)
            standings[group] = ranked
            for r in ranked:
                self.reached[r.team] = "GROUP"
        self.group_standings = standings
        return standings

    # -- third-place matrix ----------------------------------------------------
    def select_best_thirds(self, standings: Dict[str, List[GroupRecord]]) -> List[GroupRecord]:
        """Rank all 12 third-placed teams; return the top 8 that advance."""
        thirds = [ranked[2] for ranked in standings.values()]
        thirds.sort(key=lambda r: r.sort_key(), reverse=True)
        return thirds[:8]

    # -- official 2026 bracket slot map ---------------------------------------
    # Round of 32 (matches 73-88), verbatim from FIFA's published bracket.
    # Tokens: "1X"/"2X" = winner/runner-up of group X; "T<m>" = the third-place
    # team allocated to match <m> (see THIRD_SLOTS for the allowed groups).
    R32_SLOTS: List[Tuple[int, str, str]] = [
        (73, "2A", "2B"), (74, "1E", "T74"), (75, "1F", "2C"), (76, "1C", "2F"),
        (77, "1I", "T77"), (78, "2E", "2I"), (79, "1A", "T79"), (80, "1L", "T80"),
        (81, "1D", "T81"), (82, "1G", "T82"), (83, "2K", "2L"), (84, "1H", "2J"),
        (85, "1B", "T85"), (86, "1J", "2H"), (87, "1K", "T87"), (88, "2D", "2G"),
    ]
    # Official allowed third-place group sets per third-place slot.
    THIRD_SLOTS: Dict[str, frozenset] = {
        "T74": frozenset("ABCDF"), "T77": frozenset("CDFGH"),
        "T79": frozenset("CEFHI"), "T80": frozenset("EHIJK"),
        "T81": frozenset("BEFIJ"), "T82": frozenset("AEHIJ"),
        "T85": frozenset("EFGIJ"), "T87": frozenset("DEIJL"),
    }
    # Later rounds reference prior-match winners: (match_id, feeder_a, feeder_b).
    R16_SLOTS = [(89, 74, 77), (90, 73, 75), (91, 76, 78), (92, 79, 80),
                 (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)]
    QF_SLOTS = [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)]
    SF_SLOTS = [(101, 97, 98), (102, 99, 100)]
    FINAL_SLOT = (103, 101, 102)

    # -- third-place allocation -----------------------------------------------
    @staticmethod
    def _match_thirds_to_slots(qualified_groups: List[str]) -> Dict[str, str]:
        """
        Allocate the 8 qualifying third-place groups to the 8 third-place slots.

        FIFA publishes a 495-row lookup (one per C(12,8) combination of which
        groups' thirds advance) mapping each combination to slot assignments.
        Rather than transcribe it, we reproduce its *effect*: a perfect
        bipartite matching between the 8 qualifying groups and the 8 slots that
        honours each slot's official allowed-group set (THIRD_SLOTS). Slots are
        processed in fixed order (T74..T87) and candidate groups tried
        alphabetically, so the assignment is deterministic and always
        constraint-valid. Returns {slot_token: group_letter}.
        """
        match: Dict[str, str] = {}        # slot  -> group
        group_to_slot: Dict[str, str] = {}  # group -> slot (inverse)

        def augment(slot: str, seen: set) -> bool:
            for g in sorted(qualified_groups):
                if g in Tournament.THIRD_SLOTS[slot] and g not in seen:
                    seen.add(g)
                    if g not in group_to_slot or augment(group_to_slot[g], seen):
                        match[slot] = g
                        group_to_slot[g] = slot
                        return True
            return False

        for slot in Tournament.THIRD_SLOTS:           # fixed order
            augment(slot, set())

        # Defensive fallback (should not trigger for a valid 8-of-12 input).
        if len(match) < len(Tournament.THIRD_SLOTS):
            remaining = [g for g in qualified_groups if g not in group_to_slot]
            for slot in Tournament.THIRD_SLOTS:
                if slot not in match and remaining:
                    match[slot] = remaining.pop()
        return match

    # -- qualifier resolution --------------------------------------------------
    def _resolve_qualifiers(self, standings: Dict[str, List[GroupRecord]]) -> Dict[str, Team]:
        """Build the slot->Team table for 1A..2L plus the 8 third-place slots."""
        slot_team: Dict[str, Team] = {}
        for group, ranked in standings.items():
            slot_team[f"1{group}"] = self.inv.teams[ranked[0].team]
            slot_team[f"2{group}"] = self.inv.teams[ranked[1].team]

        best_thirds = self.select_best_thirds(standings)            # top 8 records
        third_by_group = {r.group: self.inv.teams[r.team] for r in best_thirds}
        for slot_token, group_letter in self._match_thirds_to_slots(
            list(third_by_group.keys())
        ).items():
            slot_team[slot_token] = third_by_group[group_letter]
        return slot_team

    # -- knockout engine -------------------------------------------------------
    def _play_match(self, ta: Team, tb: Team, stage_name: str) -> Team:
        """Play one knockout tie; stamp both participants' furthest stage."""
        res = self._fixed_or_simulate(ta, tb, knockout=True)
        winner = ta if res.winner == ta.name else tb
        self.reached[ta.name] = stage_name
        self.reached[tb.name] = stage_name
        self._log_match(stage_name, res)
        return winner

    # -- full run --------------------------------------------------------------
    def run(self) -> Dict[str, str]:
        """
        Execute one complete tournament via the official 2026 bracket and return
        {team_name: furthest_stage}. Labels: GROUP, R32, R16, QF, SF, FINAL,
        CHAMPION (the label is the round a team *reached*, i.e. last played).
        """
        standings = self.play_group_stage()
        slot_team = self._resolve_qualifiers(standings)

        winners: Dict[int, Team] = {}  # match_id -> winning Team

        # Round of 32: resolve slot tokens -> teams, then play each tie.
        for mid, sa, sb in self.R32_SLOTS:
            winners[mid] = self._play_match(slot_team[sa], slot_team[sb], "R32")
        for mid, _, _ in self.R32_SLOTS:
            self.reached[winners[mid].name] = "R16"  # survivors advance to R16

        # Subsequent rounds follow the explicit winner-feed maps.
        def play_round(slots, stage_in: str, stage_next: str) -> None:
            for mid, fa, fb in slots:
                winners[mid] = self._play_match(winners[fa], winners[fb], stage_in)
            for mid, _, _ in slots:
                self.reached[winners[mid].name] = stage_next

        play_round(self.R16_SLOTS, "R16", "QF")
        play_round(self.QF_SLOTS, "QF", "SF")
        play_round(self.SF_SLOTS, "SF", "FINAL")

        _, fa, fb = self.FINAL_SLOT
        champion = self._play_match(winners[fa], winners[fb], "FINAL")
        self.reached[champion.name] = "CHAMPION"
        return dict(self.reached)


class Tournament32(Tournament):
    """
    Historical 32-team / 8-group resolver (2010-2022 format) for the backtest.

    PIN_REAL_RESULTS stays off: 2010's Group A contains the same
    Mexico/South Africa pairing as 2026, and pinning would leak the live
    result into the backtest.

    Group stage: 8 groups of 4, top two advance. Knockouts follow the FIFA
    bracket used in this era:

        R16: 1A-2B, 1C-2D, 1B-2A, 1D-2C, 1E-2F, 1G-2H, 1F-2E, 1H-2G
        QF : W49-W50, W53-W54, W51-W52, W55-W56
        SF : W57-W58, W59-W60

    Stage labels reuse the shared vocabulary minus R32:
    GROUP -> R16 -> QF -> SF -> FINAL -> CHAMPION.
    """

    PIN_REAL_RESULTS = False

    R16_SLOTS_32: List[Tuple[int, str, str]] = [
        (49, "1A", "2B"), (50, "1C", "2D"), (51, "1B", "2A"), (52, "1D", "2C"),
        (53, "1E", "2F"), (54, "1G", "2H"), (55, "1F", "2E"), (56, "1H", "2G"),
    ]
    QF_SLOTS_32 = [(57, 49, 50), (58, 53, 54), (59, 51, 52), (60, 55, 56)]
    SF_SLOTS_32 = [(61, 57, 58), (62, 59, 60)]
    FINAL_SLOT_32 = (64, 61, 62)

    def run(self) -> Dict[str, str]:
        standings = self.play_group_stage()

        slot_team: Dict[str, Team] = {}
        for group, ranked in standings.items():
            slot_team[f"1{group}"] = self.inv.teams[ranked[0].team]
            slot_team[f"2{group}"] = self.inv.teams[ranked[1].team]

        winners: Dict[int, Team] = {}
        for mid, sa, sb in self.R16_SLOTS_32:
            winners[mid] = self._play_match(slot_team[sa], slot_team[sb], "R16")
        for mid, _, _ in self.R16_SLOTS_32:
            self.reached[winners[mid].name] = "QF"

        def play_round(slots, stage_in: str, stage_next: str) -> None:
            for mid, fa, fb in slots:
                winners[mid] = self._play_match(winners[fa], winners[fb], stage_in)
            for mid, _, _ in slots:
                self.reached[winners[mid].name] = stage_next

        play_round(self.QF_SLOTS_32, "QF", "SF")
        play_round(self.SF_SLOTS_32, "SF", "FINAL")

        _, fa, fb = self.FINAL_SLOT_32
        champion = self._play_match(winners[fa], winners[fb], "FINAL")
        self.reached[champion.name] = "CHAMPION"
        return dict(self.reached)


# ==============================================================================
# LAYER 5 — MONTE CARLO WRAPPER & TELEMETRY (process-safe)
# ==============================================================================

# Stage ordering for "reached at least X" aggregation.
_STAGE_ORDER = ["GROUP", "R32", "R16", "QF", "SF", "FINAL", "CHAMPION"]
_STAGE_RANK = {s: i for i, s in enumerate(_STAGE_ORDER)}

# ------------------------------------------------------------------------------
# PROCESS-SAFE WORKER STATE. ProcessPoolExecutor forks (or spawns) workers; an
# HTTP/LLM client created in the parent and shared globally would collide on
# socket initialisation across processes. Every connection-owning object is
# therefore constructed lazily INSIDE the worker on first use and cached in
# worker-local module globals. Nothing with a live socket ever crosses the
# process boundary.
# ------------------------------------------------------------------------------
_WORKER_AGENT: Optional[TacticalManagerAgent] = None
_WORKER_AGENT_LLM: Optional[bool] = None


def _get_worker_agent(use_llm: bool) -> TacticalManagerAgent:
    """Lazy per-process tactical agent (and, if LLM, per-process socket pool)."""
    global _WORKER_AGENT, _WORKER_AGENT_LLM
    if _WORKER_AGENT is None or _WORKER_AGENT_LLM != use_llm:
        _WORKER_AGENT = TacticalManagerAgent(build_default_news_store(), use_llm=use_llm)
        _WORKER_AGENT_LLM = use_llm
    return _WORKER_AGENT


def _build_inventory(year: int, feature_mode: str, rng_seed: int) -> WorldCupInventory:
    """Dispatch: year 0 == the 2026 field; otherwise a historical field."""
    if year == 0:
        return build_world_cup_2026(rng_seed=rng_seed, feature_mode=feature_mode)
    return build_historical_world_cup(year, rng_seed=rng_seed, feature_mode=feature_mode)


def _run_one_tournament(
    seed: int,
    config: Optional[EngineConfig] = None,
    year: int = 0,
    feature_mode: str = "full",
    use_llm: bool = False,
) -> Dict[str, str]:
    """
    Worker entrypoint for a single Monte Carlo iteration.

    The inventory is rebuilt inside the worker so each process is fully
    independent (no shared mutable state across the fork) and seeded
    deterministically for reproducibility. `config` (frozen, picklable)
    carries any calibrated goal-process parameters across the boundary, and
    the tactical agent (with any LLM socket pool) is built lazily inside this
    process via _get_worker_agent().
    """
    rng = np.random.default_rng(seed)
    inventory = _build_inventory(year, feature_mode, seed)
    agent = _get_worker_agent(use_llm)
    cls = Tournament if year == 0 else Tournament32
    return cls(inventory, tactical_agent=agent, rng=rng, config=config).run()


class MonteCarloEngine:
    """Multiprocessed Monte Carlo driver + Pandas telemetry aggregation."""

    def __init__(
        self,
        iterations: int = 1000,
        workers: Optional[int] = None,
        base_seed: int = 2026,
        config: Optional[EngineConfig] = None,
        year: int = 0,                 # 0 = the 2026 field
        feature_mode: str = "full",    # "full" | "fifa_only"
        use_llm: bool = False,         # per-worker lazy LLM agent
    ):
        self.iterations = iterations
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self.base_seed = base_seed
        self.config = config
        self.year = year
        self.feature_mode = feature_mode
        self.use_llm = use_llm

    def run_counts(self) -> Dict[str, Dict[str, int]]:
        """Run all iterations; return raw 'reached at least stage' counts."""
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        seeds = [self.base_seed + i for i in range(self.iterations)]

        LOGGER.info("Running %d tournaments (%s, %s) on %d workers...",
                    self.iterations, self.year or 2026, self.feature_mode, self.workers)
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = [
                pool.submit(_run_one_tournament, s, self.config,
                            self.year, self.feature_mode, self.use_llm)
                for s in seeds
            ]
            done = 0
            for fut in as_completed(futures):
                result = fut.result()
                for team, stage in result.items():
                    reached_rank = _STAGE_RANK[stage]
                    for s in _STAGE_ORDER:
                        if reached_rank >= _STAGE_RANK[s]:
                            counts[team][s] += 1
                done += 1
                if done % max(1, self.iterations // 10) == 0:
                    LOGGER.info("  ... %d/%d complete", done, self.iterations)
        return counts

    def run(self) -> pd.DataFrame:
        """Run all iterations and return the aggregated probability table."""
        return self._to_dataframe(self.run_counts())

    def _to_dataframe(self, counts: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """Convert raw counts to a sorted percentage-probability DataFrame."""
        n = float(self.iterations)
        rows = []
        for team, stage_counts in counts.items():
            rows.append({
                "Country": team,
                "Reach R16 %": 100.0 * stage_counts.get("R16", 0) / n,
                "Reach QF %": 100.0 * stage_counts.get("QF", 0) / n,
                "Reach SF %": 100.0 * stage_counts.get("SF", 0) / n,
                "Reach Final %": 100.0 * stage_counts.get("FINAL", 0) / n,
                "Win Trophy %": 100.0 * stage_counts.get("CHAMPION", 0) / n,
            })
        df = pd.DataFrame(rows).sort_values("Win Trophy %", ascending=False).reset_index(drop=True)
        return df.round(2)


# ==============================================================================
# CALIBRATION HARNESS — tune the goal process against historical scorelines
# ==============================================================================


class CalibrationHarness:
    """
    Calibrate EngineConfig against a historical scoring target.

    Average goals per match at recent men's World Cups: 2014 = 2.67,
    2018 = 2.64, 2022 = 2.69 — so the default target is 2.65. Average goals is
    monotonically increasing in `base_xg`, which lets us bisect a single scalar
    to the target instead of running a full optimizer.
    """

    HISTORICAL_AVG_GOALS = 2.65          # men's WC scoring baseline
    HISTORICAL_DRAW_RATE = 0.235         # ~ group-stage draw frequency (diagnostic)

    def __init__(
        self,
        target_avg_goals: float = HISTORICAL_AVG_GOALS,
        sample_matches: int = 1500,
        seed: int = 2026,
    ):
        self.target = target_avg_goals
        self.sample_matches = sample_matches
        self.seed = seed

    def measure(self, config: EngineConfig) -> Dict[str, float]:
        """Simulate `sample_matches` random matchups; return scoring diagnostics."""
        rng = np.random.default_rng(self.seed)         # fixed seed -> low-noise objective
        inv = build_world_cup_2026(rng_seed=self.seed)
        teams = list(inv.teams.values())
        sim = MatchSimulator(rng, config=config)
        total_goals = 0
        draws = 0
        for _ in range(self.sample_matches):
            i, j = rng.choice(len(teams), size=2, replace=False)
            res = sim.simulate(teams[i], teams[j], knockout=False)
            total_goals += res.home_goals + res.away_goals
            draws += int(res.home_goals == res.away_goals)
        n = float(self.sample_matches)
        return {"base_xg": config.base_xg, "avg_goals": total_goals / n, "draw_rate": draws / n}

    def calibrate(
        self,
        lo: float = 0.5,
        hi: float = 3.0,
        tol: float = 0.02,
        max_iter: int = 14,
    ) -> Tuple[EngineConfig, Dict[str, float]]:
        """Bisect `base_xg` in [lo, hi] until simulated avg goals hits target."""
        defaults = EngineConfig()
        best: Tuple[EngineConfig, Dict[str, float]] = (defaults, self.measure(defaults))
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            cfg = EngineConfig(
                base_xg=mid,
                fatigue_rate=defaults.fatigue_rate,
                fatigue_penalty=defaults.fatigue_penalty,
            )
            stats = self.measure(cfg)
            best = (cfg, stats)
            LOGGER.info(
                "calibrate base_xg=%.3f -> avg_goals=%.3f draw_rate=%.3f (target %.2f)",
                mid, stats["avg_goals"], stats["draw_rate"], self.target,
            )
            if abs(stats["avg_goals"] - self.target) <= tol:
                break
            if stats["avg_goals"] < self.target:
                lo = mid
            else:
                hi = mid
        return best


# ==============================================================================
# DATA FACTORIES — feature-grounded squads (2026 field + historical fields)
# ==============================================================================

# group -> [team names] — the real 2026 draw (Washington DC, 5 Dec 2025).
_REAL_GROUPS_2026: Dict[str, List[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Host nation per team -> travel-burden anchor. Co-hosts anchor to themselves;
# everyone else anchors to the USA (where the bulk of venues sit).
_HOST_NATIONS = {"Mexico": "Mexico", "Canada": "Canada", "United States": "USA"}

# Coarse inter-nation travel cost between host regions (drives travel_burden).
_REGION_TRAVEL = {
    ("USA", "USA"): 0.20, ("USA", "Canada"): 0.45, ("USA", "Mexico"): 0.50,
    ("Canada", "Canada"): 0.20, ("Canada", "Mexico"): 0.70,
    ("Mexico", "Mexico"): 0.20,
}


def _travel_burden(anchor: str, host: str) -> float:
    key = tuple(sorted((anchor, host)))  # symmetric lookup
    return _REGION_TRAVEL.get((anchor, host), _REGION_TRAVEL.get(key, 0.4))


# Per-process cache of the optional per-player dataset (WC_PLAYERS_CSV).
_PLAYERS_CSV_CACHE: Optional[Dict[str, List[Dict[str, str]]]] = None


def _players_csv_rows(team_name: str) -> List[Dict[str, str]]:
    """Rows of the per-player performance CSV for one team ([] if absent)."""
    global _PLAYERS_CSV_CACHE
    if _PLAYERS_CSV_CACHE is None:
        rows: Dict[str, List[Dict[str, str]]] = {}
        path = os.getenv("WC_PLAYERS_CSV", "")
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    rows.setdefault(row["team"].strip(), []).append(row)
            LOGGER.info("Per-player CSV: %d teams / %d players from %s",
                        len(rows), sum(len(v) for v in rows.values()), path)
        _PLAYERS_CSV_CACHE = rows
    return _PLAYERS_CSV_CACHE.get(team_name, [])


def _squad_from_csv_rows(
    rows: List[Dict[str, str]],
    anchor: str,
    rng: np.random.Generator,
) -> List[Player]:
    """Build a squad from per-player CSV rows.

    Ability/form/club metrics come from the dataset; fatigue and travel
    burden stay situational (per-iteration random) because they describe
    tournament conditions, not the player.
    """
    burden = _travel_burden(anchor, "USA")
    squad: List[Player] = []
    for row in rows:
        pos = _normalize_position(row.get("position", "")) or Position.MID
        squad.append(Player(
            name=row.get("player", "?").strip(),
            position=pos,
            base_rating=float(row["base_rating"]),
            form=float(row["form"]),
            club_performance=float(row["club_performance"]),
            club_minutes=float(row["club_minutes"]),
            fatigue=float(np.clip(rng.normal(0.15, 0.08), 0.0, 0.6)),
            travel_burden=float(np.clip(burden + rng.normal(0, 0.05), 0.0, 1.0)),
            market_value_m=float(row.get("market_value_m") or 0.0),
        ))
    return squad


def _generate_squad(
    team_name: str,
    strength: float,
    anchor: str,
    rng: np.random.Generator,
    feats: Optional[TeamFeatures] = None,
) -> List[Player]:
    """
    Generate a 23-player squad whose metrics are grounded in the team's REAL
    feature row instead of strength-only noise:

      * base_rating       — composite strength tier + per-player spread
      * form              — centred on the team's REAL recent-form scalar
      * club_performance  — centred on the strength tier
      * club_minutes      — drawn around the team's REAL avg club-minutes share
                            (Beta-shaped: most squads have a played-everything
                            core and a low-minutes tail)
      * market_value_m    — the team's REAL squad value split over players with
                            a realistic heavy tail (top earners dominate)
      * injuries_out      — that many first-choice players flagged unavailable
                            BEFORE the tournament starts

    Squad shape: 3 GK, 8 DEF, 7 MID, 5 FWD.
    """
    csv_rows = _players_csv_rows(team_name)
    if csv_rows:
        squad = _squad_from_csv_rows(csv_rows, anchor, rng)
    else:
        plan = [(Position.GK, 3), (Position.DEF, 8), (Position.MID, 7), (Position.FWD, 5)]
        burden = _travel_burden(anchor, "USA")
        form_mu = feats.form if feats else 0.55
        minutes_mu = feats.avg_minutes_share if feats else 0.70
        total_mv = feats.market_value_m if (feats and not math.isnan(feats.market_value_m)) else 0.0

        # Heavy-tailed per-player market-value shares (sorted: stars first).
        raw_shares = np.sort(rng.pareto(2.5, 23) + 0.05)[::-1]
        shares = raw_shares / raw_shares.sum()

        squad = []
        idx = 0
        for pos, count in plan:
            for _ in range(count):
                idx += 1
                base = float(np.clip(strength + rng.normal(0, 0.07), 0.20, 0.99))
                minutes = float(np.clip(rng.beta(5.0, 5.0 * (1.0 - minutes_mu) / max(minutes_mu, 0.05)), 0.05, 1.0))
                squad.append(Player(
                    name=f"{team_name} {pos.value[:3]}{idx}",
                    position=pos,
                    base_rating=base,
                    form=float(np.clip(rng.normal(form_mu, 0.12), 0.0, 1.0)),
                    club_performance=float(np.clip(strength + rng.normal(0, 0.10), 0.0, 1.0)),
                    club_minutes=minutes,
                    fatigue=float(np.clip(rng.normal(0.15, 0.08), 0.0, 0.6)),
                    travel_burden=float(np.clip(burden + rng.normal(0, 0.05), 0.0, 1.0)),
                    market_value_m=float(total_mv * shares[idx - 1]),
                ))

    # Pre-tournament injury status: rule out the N best-rated outfielders.
    if feats and feats.injuries_out > 0:
        outfield = sorted((p for p in squad if p.position != Position.GK),
                          key=lambda p: p.effective_rating(), reverse=True)
        for p in outfield[: feats.injuries_out]:
            p.available = False
    return squad


def _build_inventory_from(
    groups: Dict[str, List[str]],
    features: Dict[str, TeamFeatures],
    rng_seed: int,
    feature_mode: str,
    n_teams: int,
    n_groups: int,
) -> WorldCupInventory:
    """Shared factory: composite (or FIFA-only) strengths -> feature squads."""
    weights = FIFA_ONLY_WEIGHTS if feature_mode == "fifa_only" else FEATURE_WEIGHTS
    strengths = composite_strengths(features, weights)

    rng = np.random.default_rng(rng_seed)
    teams: Dict[str, Team] = {}
    for group, members in groups.items():
        for name in members:
            feats = features[name]
            anchor = _HOST_NATIONS.get(name, "USA")
            squad = _generate_squad(name, strengths[name], anchor, rng, feats=feats)
            teams[name] = Team(
                name=name, group=group, squad=squad, anchor_country=anchor,
                fifa_points=(feats.fifa_points if not math.isnan(feats.fifa_points)
                             else feats.fifa_rank),
                features=feats,
            )
    inv = WorldCupInventory(teams=teams, n_teams=n_teams, n_groups=n_groups)
    inv.validate()
    return inv


def build_world_cup_2026(
    rng_seed: int = 2026,
    feature_mode: str = "full",
    live_refresh: bool = False,
) -> WorldCupInventory:
    """
    Construct and validate the 48-team 2026 inventory.

    Team identities and group placements are the official draw; strength is
    the multi-source composite (Elo + market value + SPI + FIFA) — or FIFA
    only when feature_mode="fifa_only" (the backtest baseline). Per-player
    matrices are grounded in each team's real form / minutes / injuries /
    squad-value features. live_refresh=True pulls the open feeds first.
    """
    feats = features_2026_live()
    if live_refresh:
        FeatureStore(feats).refresh_live()
    return _build_inventory_from(_REAL_GROUPS_2026, feats, rng_seed,
                                 feature_mode, n_teams=48, n_groups=12)


def build_historical_world_cup(
    year: int,
    rng_seed: int = 7,
    feature_mode: str = "full",
) -> WorldCupInventory:
    """Construct a 32-team historical inventory (2010 / 2014 / 2018 / 2022)."""
    if year not in _HISTORICAL_FIELDS:
        raise ValueError(f"No historical field for {year}; have {sorted(_HISTORICAL_FIELDS)}")
    groups = {g: [m[0] for m in members]
              for g, members in _HISTORICAL_FIELDS[year].items()}
    feats = features_historical(year)
    return _build_inventory_from(groups, feats, rng_seed, feature_mode,
                                 n_teams=32, n_groups=8)


# ==============================================================================
# LAYER 6 — BACKTEST HARNESS (2010-2022 validation vs FIFA-rank baseline)
# ==============================================================================


class BacktestHarness:
    """
    Validate the feature stack on completed World Cups.

    For each year, both models (full composite vs FIFA-rank-only) run the same
    Monte Carlo over the REAL draw and bracket. Scoring, per model:

      champ_prob   probability mass the model put on the actual champion
      champ_rank   the actual champion's rank in the model's title list
                   (1 = the model's favourite won)
      champ_hit@1  1 if the model's single favourite was the champion
      champ_logloss  -ln(champ_prob), the proper score for the title call
      sf_recall@4  of the model's four most-likely semifinalists, how many
                   actually made the semifinals (0..4)
      sf_mass      total SF probability the model put on the real four SFs

    A model that genuinely uses information beats the baseline on logloss and
    sf_recall in aggregate, not necessarily every single year — single
    tournaments are one draw from a fat-tailed distribution.
    """

    def __init__(self, iterations: int = 400, workers: Optional[int] = None,
                 base_seed: int = 7, config: Optional[EngineConfig] = None,
                 use_llm: bool = False):
        self.iterations = iterations
        self.workers = workers
        self.base_seed = base_seed
        self.config = config
        self.use_llm = use_llm

    def _probabilities(self, year: int, feature_mode: str) -> pd.DataFrame:
        eng = MonteCarloEngine(
            iterations=self.iterations, workers=self.workers,
            base_seed=self.base_seed, config=self.config,
            year=year, feature_mode=feature_mode, use_llm=self.use_llm,
        )
        return eng.run()

    def _score(self, df: pd.DataFrame, year: int) -> Dict[str, float]:
        truth = _HISTORICAL_RESULTS[year]
        champion: str = truth["champion"]               # type: ignore[assignment]
        sfs: set = truth["semifinalists"]               # type: ignore[assignment]

        ranked = df.sort_values("Win Trophy %", ascending=False).reset_index(drop=True)
        champ_row = ranked[ranked["Country"] == champion]
        champ_prob = float(champ_row["Win Trophy %"].iloc[0]) / 100.0 if len(champ_row) else 0.0
        champ_rank = int(champ_row.index[0]) + 1 if len(champ_row) else len(ranked)

        sf_ranked = df.sort_values("Reach SF %", ascending=False)
        top4 = set(sf_ranked["Country"].head(4))
        sf_recall = len(top4 & sfs)
        sf_mass = float(df[df["Country"].isin(sfs)]["Reach SF %"].sum()) / 100.0

        eps = 1.0 / (2.0 * self.iterations)             # avoid -ln(0)
        return {
            "champ_prob": champ_prob,
            "champ_rank": champ_rank,
            "champ_hit@1": float(ranked["Country"].iloc[0] == champion),
            "champ_logloss": -math.log(max(champ_prob, eps)),
            "sf_recall@4": float(sf_recall),
            "sf_mass": sf_mass,
        }

    def run(self, years: Sequence[int]) -> pd.DataFrame:
        """Run both models over all requested years; return the score table."""
        rows: List[Dict[str, object]] = []
        for year in years:
            truth = _HISTORICAL_RESULTS[year]
            for mode, label in (("full", "features"), ("fifa_only", "fifa_baseline")):
                LOGGER.info("Backtest %d [%s] ...", year, label)
                df = self._probabilities(year, mode)
                scores = self._score(df, year)
                rows.append({
                    "year": year, "model": label,
                    "actual_champion": truth["champion"],
                    "model_top3": ", ".join(
                        f"{c} ({p:.0f}%)" for c, p in
                        zip(df["Country"].head(3), df["Win Trophy %"].head(3))
                    ),
                    **{k: round(v, 3) for k, v in scores.items()},
                })
        out = pd.DataFrame(rows)

        # Aggregate row per model across years.
        for label in ("features", "fifa_baseline"):
            sub = out[out["model"] == label]
            rows.append({
                "year": "ALL", "model": label, "actual_champion": "—",
                "model_top3": "—",
                "champ_prob": round(sub["champ_prob"].mean(), 3),
                "champ_rank": round(sub["champ_rank"].mean(), 2),
                "champ_hit@1": round(sub["champ_hit@1"].mean(), 3),
                "champ_logloss": round(sub["champ_logloss"].mean(), 3),
                "sf_recall@4": round(sub["sf_recall@4"].mean(), 3),
                "sf_mass": round(sub["sf_mass"].mean(), 3),
            })
        return pd.DataFrame(rows)


# ==============================================================================
# SHOWCASE / DEMOS
# ==============================================================================


def _showcase_match(use_llm: bool = True) -> None:
    """Run a single verbose match with the (optionally live) tactical agent."""
    inv = build_world_cup_2026()
    agent = TacticalManagerAgent(build_default_news_store(), use_llm=use_llm)
    backend = (f"local LLM ({LLM_MODEL} @ {LLM_BASE_URL})"
               if agent.use_llm else "local heuristic")
    LOGGER.info("Tactical agent backend: %s", backend)
    sim = MatchSimulator(np.random.default_rng(7))
    france, argentina = inv.teams["France"], inv.teams["Argentina"]
    res = sim.simulate(france, argentina, knockout=True,
                       tactical_agent=agent, verbose=True)
    if res.events:
        LOGGER.info("Live events:")
        for minute, team, event in res.events:
            LOGGER.info("  %3d'  %-14s %s", minute, team, event)
    else:
        LOGGER.info("Live events: none this match.")


def _refresh_demo(n_teams: int) -> None:
    """Exercise the key-free (no API key) live data-ingestion layer."""
    inv = build_world_cup_2026()
    names = list(inv.teams.keys())[:n_teams]
    ingestor = AsyncDataIngestor(inv)
    LOGGER.info("Key-free ingestion demo for: %s", ", ".join(names))
    summaries = asyncio.run(ingestor.fetch_team_summaries(names))
    for team, extract in summaries.items():
        snippet = (extract[:90] + "...") if extract else "(no data / offline)"
        LOGGER.info("  %-26s -> %s", team, snippet)
    got = sum(1 for e in summaries.values() if e)
    LOGGER.info("Wikipedia REST returned data for %d/%d teams (0 = offline).", got, len(names))


def _refresh_features_demo() -> None:
    """Exercise the open-feed feature refreshers (SPI / Elo / MV) and report deltas."""
    feats = features_2026()
    store = FeatureStore(feats)
    report = store.refresh_live()
    for source, status in report.items():
        LOGGER.info("  %-18s %s", source, status)
    df = pd.DataFrame(
        [{"team": n, "elo": f.elo, "mv_eur_m": f.market_value_m,
          "spi": f.spi, "fifa_pts": f.fifa_points}
         for n, f in sorted(feats.items(), key=lambda kv: -(kv[1].elo or 0))]
    )
    print(df.head(15).to_string(index=False))


# ==============================================================================
# MARKDOWN REPORT WRITER — one realised tournament, every match to the trophy
# ==============================================================================


def _df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no tabulate dep)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _group_table_md(ranked: List[GroupRecord]) -> str:
    """Standings table for one group (V = advance, + = third place)."""
    lines = [
        "| # | Team | Pld | W | D | L | GF | GA | GD | Pts | |",
        "|---|------|-----|---|---|---|----|----|----|-----|---|",
    ]
    for i, r in enumerate(ranked, 1):
        mark = "✅" if i <= 2 else ("➕" if i == 3 else "")
        lines.append(
            f"| {i} | {r.team} | {r.played} | {r.won} | {r.drawn} | {r.lost} | "
            f"{r.gf} | {r.ga} | {r.gd:+d} | {r.points} | {mark} |"
        )
    return "\n".join(lines)


def _fmt_ko(row: Dict) -> str:
    """Format one knockout match-log row as a markdown bullet."""
    score = f"{row['hg']}-{row['ag']}"
    extra = ""
    if row["pens"]:
        extra = f" _(a.e.t.; penalties {row['pens'][0]}-{row['pens'][1]})_"
    line = f"- {row['home']} **{score}** {row['away']}{extra} -> **{row['winner']}**"
    evs = row.get("events") or []
    if evs:
        bits = "; ".join(f"{m}' {t}: {e}" for m, t, e in evs[:6])
        line += f"\n  - _events: {bits}_"
    return line


def write_real_data_report(
    path: str,
    seed: int = 2026,
    config: Optional[EngineConfig] = None,
    mc_iters: int = 0,
    workers: Optional[int] = None,
    use_llm: bool = False,
) -> str:
    """
    Run ONE full tournament on the real 2026 data and write a markdown report
    covering every match from the group stage to the trophy. Optionally appends
    Monte Carlo win-probabilities (mc_iters > 0).
    """
    inv = build_world_cup_2026(rng_seed=seed)
    agent = TacticalManagerAgent(build_default_news_store(), use_llm=use_llm)
    tour = Tournament(inv, tactical_agent=agent, rng=np.random.default_rng(seed),
                      config=config, record=True)
    reached = tour.run()
    champion = next(t for t, s in reached.items() if s == "CHAMPION")

    base_xg = (config or EngineConfig()).base_xg
    out: List[str] = []
    out.append("# 2026 FIFA World Cup — Simulated Tournament Report (v2)")
    out.append("")
    out.append(
        "_Single realised tournament on the official 12-group draw. Team "
        "strength is a multi-source composite (Elo, squad market value, "
        "SPI-style rating, FIFA points). Squads are grounded in real squad "
        "value, club minutes, recent form and injury features. Matches run "
        "through the graph spatio-temporal Attenuated-Poisson engine with "
        "live red cards, fatigue-driven substitutions, and a halftime "
        "tactical agent._")
    out.append("")
    out.append(f"- **Champion:** 🏆 **{champion}**")
    out.append(f"- **RNG seed:** `{seed}`  ·  **Calibrated `base_xg`:** `{base_xg:.3f}`")
    out.append(f"- **Engine:** 11-node GNN cohesion -> minute-by-minute Poisson; "
               f"knockouts resolve via extra time + penalties.")
    out.append("")

    out.append("## Group Stage")
    group_matches = [m for m in tour.match_log if m["stage"] == "GROUP"]
    for group, ranked in tour.group_standings.items():
        out.append(f"\n### Group {group}")
        out.append("")
        out.append(_group_table_md(ranked))
        out.append("\n**Results:**\n")
        for m in (gm for gm in group_matches if gm["group"] == group):
            out.append(f"- {m['home']} {m['hg']}-{m['ag']} {m['away']}")

    all_thirds = sorted(
        (ranked[2] for ranked in tour.group_standings.values()),
        key=lambda r: r.sort_key(), reverse=True,
    )
    out.append("\n## Third-Placed Teams — Ranking (top 8 advance)\n")
    out.append("| Rank | Team | Group | Pts | GD | GF | Advances |")
    out.append("|------|------|-------|-----|----|----|----------|")
    for i, r in enumerate(all_thirds, 1):
        out.append(f"| {i} | {r.team} | {r.group} | {r.points} | {r.gd:+d} | "
                   f"{r.gf} | {'✅' if i <= 8 else '❌'} |")

    out.append("\n## Knockout Stage")
    for stage, title in [("R32", "Round of 32"), ("R16", "Round of 16"),
                         ("QF", "Quarter-finals"), ("SF", "Semi-finals"),
                         ("FINAL", "Final")]:
        rows = [m for m in tour.match_log if m["stage"] == stage]
        out.append(f"\n### {title}\n")
        for m in rows:
            out.append(_fmt_ko(m))
    out.append(f"\n## 🏆 Champion: {champion}\n")

    if mc_iters > 0:
        LOGGER.info("Report: running %d-iteration Monte Carlo appendix...", mc_iters)
        df = MonteCarloEngine(iterations=mc_iters, workers=workers,
                              base_seed=seed, config=config).run()
        out.append("---\n")
        out.append(f"## Appendix — Monte Carlo Probabilities ({mc_iters:,} tournaments)\n")
        out.append(_df_to_md(df))

    text = "\n".join(out) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    LOGGER.info("Wrote report (%d chars) to %s — champion: %s", len(text), path, champion)
    return path


# ==============================================================================
# CLI
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FIFA World Cup Monte Carlo simulator v2 "
                    "(multi-source features, backtests, local-LLM tactics)")
    parser.add_argument("--iterations", type=int, default=1000, help="number of full tournaments")
    parser.add_argument("--workers", type=int, default=None, help="process pool size")
    parser.add_argument("--seed", type=int, default=2026, help="base RNG seed")
    parser.add_argument("--top", type=int, default=20, help="rows to display")
    parser.add_argument("--fifa-only", action="store_true",
                        help="use the FIFA-rank-only baseline strengths instead of "
                             "the full Elo+MV+SPI+FIFA composite")
    parser.add_argument("--llm", action="store_true",
                        help="enable the local vLLM/Ollama tactical agent "
                             "(WC_LLM_BASE_URL / WC_LLM_MODEL) inside workers")
    parser.add_argument("--backtest", metavar="YEARS", default="",
                        help="comma-separated completed World Cups to validate on, "
                             "e.g. 2010,2014,2018,2022 — runs full-feature vs "
                             "FIFA-only models and scores both against history; "
                             "--iterations sets the per-model MC size")
    parser.add_argument("--showcase", action="store_true",
                        help="run one verbose LLM-assisted match (red cards / subs visible)")
    parser.add_argument("--calibrate", action="store_true",
                        help="tune base_xg to the historical scoring target, then run")
    parser.add_argument("--refresh", type=int, default=0, metavar="N",
                        help="demo the key-free Wikipedia ingestor for N teams, then exit")
    parser.add_argument("--refresh-features", action="store_true",
                        help="pull the open SPI/Elo/MV feeds into the 2026 feature "
                             "table and print the result, then exit")
    parser.add_argument("--report", metavar="PATH", default="",
                        help="write a full real-data match-by-match report to PATH, "
                             "then exit; --iterations sets the MC appendix size")
    parser.add_argument("--log", default="INFO", help="log level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.showcase:
        _showcase_match(use_llm=True)
        return

    if args.refresh:
        _refresh_demo(args.refresh)
        return

    if args.refresh_features:
        _refresh_features_demo()
        return

    if args.backtest:
        years = [int(y.strip()) for y in args.backtest.split(",") if y.strip()]
        harness = BacktestHarness(iterations=args.iterations, workers=args.workers,
                                  base_seed=args.seed, use_llm=args.llm)
        df = harness.run(years)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 160)
        print("\n" + "=" * 100)
        print(f"  BACKTEST — full feature stack vs FIFA-rank baseline "
              f"({args.iterations:,} tournaments per model per year)")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        print("  Read the ALL rows: lower champ_logloss and higher sf_recall@4 / "
              "sf_mass = better model.")
        return

    if args.report:
        cfg, _ = CalibrationHarness().calibrate()
        write_real_data_report(args.report, seed=args.seed, config=cfg,
                               mc_iters=args.iterations, workers=args.workers,
                               use_llm=args.llm)
        print(f"\nReport written to {args.report}")
        return

    config: Optional[EngineConfig] = None
    if args.calibrate:
        config, stats = CalibrationHarness().calibrate()
        print("\n" + "-" * 78)
        print(f"  CALIBRATED EngineConfig: base_xg={config.base_xg:.3f} "
              f"(avg_goals={stats['avg_goals']:.3f}, draw_rate={stats['draw_rate']:.3f})")
        print("-" * 78)

    engine = MonteCarloEngine(
        iterations=args.iterations, workers=args.workers, base_seed=args.seed,
        config=config, year=0,
        feature_mode="fifa_only" if args.fifa_only else "full",
        use_llm=args.llm,
    )
    df = engine.run()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    mode = "FIFA-only baseline" if args.fifa_only else "full composite features"
    print("\n" + "=" * 78)
    print(f"  2026 FIFA WORLD CUP — MONTE CARLO TELEMETRY  "
          f"({args.iterations:,} iterations, {mode})")
    print("=" * 78)
    print(df.head(args.top).to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
