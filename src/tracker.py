"""The tracker: daily mood score -> personal baseline -> sustained-decline flag.

Nothing here is trained. The detector is a one-sided CUSUM on the person's
standardised drop below their OWN typical mood, which is the textbook tool for
"has the level shifted and stayed shifted?" -- exactly the question that
separates a sustained decline from a bad week.

Each person is flagged at most once (a flag means a human check-in), so the
unit of evaluation is the person, and the threshold is a quantile of
*per-person* maxima among people who never declined -- not a pooled day-level
quantile, which describes whoever contributed the extreme days (project #12).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import valence


def score_matrix(people: pd.DataFrame, sents: pd.DataFrame, pool_probs: np.ndarray,
                 days: int) -> np.ndarray:
    """(people x days) mean sentence valence; NaN on days with no entry."""
    v = valence(pool_probs)[sents.pool_idx.to_numpy()]
    df = pd.DataFrame(dict(p=sents.person_id.to_numpy(), d=sents.day.to_numpy(), v=v))
    g = df.groupby(["p", "d"]).v.mean()
    M = np.full((len(people), days), np.nan)
    M[g.index.get_level_values(0), g.index.get_level_values(1)] = g.to_numpy()
    return M


def latent_matrix(daily: pd.DataFrame, n_people: int, days: int, written_only=False) -> np.ndarray:
    M = daily.latent.to_numpy().reshape(n_people, days).copy()
    if written_only:
        W = daily.wrote.to_numpy().reshape(n_people, days).astype(bool)
        M[~W] = np.nan
    return M


def personal_baseline(M: np.ndarray, baseline_days: int, pop_sd: float | None = None, n0=5.0):
    B = M[:, :baseline_days]
    n = np.sum(~np.isnan(B), 1).astype(float)
    with np.errstate(all="ignore"):
        mu = np.nanmean(B, 1)
        sd = np.nanstd(B, 1, ddof=1)
    if pop_sd is None:
        pop_sd = float(np.nanmedian(sd))
    mu = np.where(np.isnan(mu), np.nanmean(mu), mu)
    sd = np.where(np.isnan(sd), pop_sd, sd)
    # shrink a noisy personal sd toward the cohort's when the baseline is thin
    sd = np.sqrt((np.maximum(n - 1, 0) * sd ** 2 + n0 * pop_sd ** 2) / (np.maximum(n - 1, 0) + n0))
    return mu, sd, n, pop_sd


def cusum(M: np.ndarray, mu: np.ndarray, sd: np.ndarray, start: int, k: float = 0.5) -> np.ndarray:
    """One-sided CUSUM of (mu - x)/sd. Days without an entry carry the statistic."""
    P, D = M.shape
    S = np.zeros((P, D))
    s = np.zeros(P)
    for t in range(start, D):
        x = M[:, t]
        obs = ~np.isnan(x)
        z = np.where(obs, (mu - np.nan_to_num(x)) / sd, 0.0)
        s = np.where(obs, np.maximum(0.0, s + z - k), s)
        S[:, t] = s
    return S


def rolling_mean(M: np.ndarray, window: int = 14, min_obs: int = 3) -> np.ndarray:
    P, D = M.shape
    obs = ~np.isnan(M)
    X = np.nan_to_num(M)
    cs = np.cumsum(np.c_[np.zeros(P), X], 1)
    cn = np.cumsum(np.c_[np.zeros(P), obs], 1)
    out = np.full((P, D), np.nan)
    for t in range(D):
        lo = max(0, t + 1 - window)
        n = cn[:, t + 1] - cn[:, lo]
        with np.errstate(all="ignore"):
            out[:, t] = np.where(n >= min_obs, (cs[:, t + 1] - cs[:, lo]) / n, np.nan)
    return out


def personal_rolling(M, mu, sd, start, window=14):
    R = (mu[:, None] - rolling_mean(M, window)) / sd[:, None]
    R[:, :start] = np.nan
    return np.nan_to_num(R, nan=-np.inf)


def population_rolling(M, start, window=14):
    """Absolute rule: a low recent average, with no reference to the person."""
    R = -rolling_mean(M, window)
    R[:, :start] = np.nan
    return np.nan_to_num(R, nan=-np.inf)


def fit_threshold(stat: np.ndarray, stable: np.ndarray, budget: float, start: int) -> float:
    """Smallest threshold at which at most `budget` of never-declining people are ever flagged."""
    mx = stat[stable, start:].max(1)
    return float(np.quantile(mx, 1.0 - budget, method="higher"))


def first_flag(stat: np.ndarray, h: float, start: int) -> np.ndarray:
    above = stat[:, start:] > h
    hit = above.any(1)
    return np.where(hit, start + above.argmax(1), -1)


def run(M: np.ndarray, cfg, rule: str = "cusum", k: float = 0.5, pop_sd: float | None = None,
        window: int = 14):
    """Return the per-day statistic matrix plus the fitted personal baseline."""
    start = cfg.baseline_days
    mu, sd, n, pop_sd = personal_baseline(M, start, pop_sd)
    if rule == "cusum":
        S = cusum(M, mu, sd, start, k)
    elif rule == "personal_rolling":
        S = personal_rolling(M, mu, sd, start, window)
    elif rule == "population_rolling":
        S = population_rolling(M, start, window)
    else:
        raise ValueError(rule)
    return S, dict(mu=mu, sd=sd, n_baseline=n, pop_sd=pop_sd)
