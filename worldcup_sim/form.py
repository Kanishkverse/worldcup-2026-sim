"""Latent tournament form (Phase 5).

Form is a small, mean-reverting signal that sits on top of the Bayesian
rating. A team that keeps beating expectation drifts positive; a team that
keeps underperforming drifts negative. The decay term pulls it back toward
zero every match so a couple of good results cannot snowball into a runaway
rating, which is the failure mode you see in naive momentum models.
"""

from __future__ import annotations


# A unit of form is worth this many Elo points when it feeds the rating.
FORM_TO_ELO = 22.0

# new_form = DECAY * form + GAIN * signal. With a sustained max signal the
# fixed point is GAIN/(1-DECAY) * signal, which we keep near 1.0 so form
# cannot dominate the prior. CLIP is the hard ceiling on top of that.
FORM_DECAY = 0.55
FORM_GAIN = 0.45
FORM_CLIP = 2.5


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def performance_signal(expected_gd: float, actual_gd: float,
                       quality: float = 0.0, scale: float = 1.6) -> float:
    """How much a result beat (or missed) the model's expectation.

    expected_gd is the pre-match expected goal difference for this team,
    actual_gd is what happened, and quality is an optional xG-based nudge
    in roughly [-1, 1] so a team that was the better side in a draw still
    gets a little credit. Output is standardised into about [-2, 2].
    """
    raw = (actual_gd - expected_gd) / scale + 0.4 * quality
    return _clip(raw, -2.0, 2.0)


def update_form(form: float, signal: float) -> float:
    """One mean-reverting form step."""
    new = FORM_DECAY * form + FORM_GAIN * signal
    return _clip(new, -FORM_CLIP, FORM_CLIP)


def form_rating_bonus(form: float) -> float:
    """Convert current form into an additive rating bonus in Elo points."""
    return FORM_TO_ELO * form
