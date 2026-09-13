"""Metrics: sentence classification, and person-level detection of a sustained decline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import CLASSES, NEG_IDX, POS_IDX, valence

GRACE_DAYS = 42   # a flag counts if it lands between onset and end of ramp + 6 weeks


# ------------------------------------------------------------------ AUC helpers
def roc_auc(y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


# ------------------------------------------------------------ classification
def classification(y_true, probs) -> dict:
    y_true = np.asarray(y_true, int)
    probs = np.asarray(probs, float)
    pred = probs.argmax(1)
    cm = confusion_matrix(y_true, pred, labels=range(len(CLASSES)))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    polar = np.isin(y_true, NEG_IDX + POS_IDX)
    yv = np.isin(y_true[polar], POS_IDX).astype(int)
    return dict(n=int(len(y_true)), accuracy=float(accuracy_score(y_true, pred)),
                macro_f1=float(f1_score(y_true, pred, average="macro", labels=range(len(CLASSES)),
                                        zero_division=0)),
                valence_auc=roc_auc(yv, valence(probs)[polar]),
                valence_sign_acc=float(np.mean((valence(probs)[polar] > 0) == (yv == 1))),
                recall={c: float(r) for c, r in zip(CLASSES, recall)},
                confusion=cm.tolist())


def within_person_r(M: np.ndarray, L: np.ndarray) -> float:
    """Pooled within-person correlation between an observed daily score and true mood,
    over the days that have an observation. Person means are removed from both."""
    ok = ~np.isnan(M)
    Md = np.where(ok, M, 0.0)
    n = np.maximum(ok.sum(1, keepdims=True), 1)
    Md = np.where(ok, M - Md.sum(1, keepdims=True) / n, np.nan)
    Lm = np.where(ok, L, 0.0).sum(1, keepdims=True) / n
    Ld = L - Lm
    return float(np.corrcoef(Md[ok], Ld[ok])[0, 1])


# ------------------------------------------------------------------ detection
def score_detection(people: pd.DataFrame, flag_day: np.ndarray, days: int, start: int,
                    grace: int = GRACE_DAYS) -> dict:
    flag_day = np.asarray(flag_day)
    drift = people.drift.to_numpy() == 1
    onset = people.onset.to_numpy()
    ramp = people.ramp.to_numpy()
    flagged = flag_day >= start
    stable_flagged = flagged & ~drift
    early = drift & flagged & (flag_day < onset)
    window_end = np.minimum(days - 1, onset + ramp + grace)
    detected = drift & flagged & (flag_day >= onset) & (flag_day <= window_end)
    late = drift & flagged & (flag_day > window_end)
    lat = (flag_day - onset)[detected]
    n_d, n_s = int(drift.sum()), int((~drift).sum())
    return dict(
        n_drift=n_d, n_stable=n_s,
        n_detected=int(detected.sum()), detection_rate=float(detected.sum() / max(1, n_d)),
        n_stable_flagged=int(stable_flagged.sum()),
        stable_flag_rate=float(stable_flagged.sum() / max(1, n_s)),
        early_flag_rate=float(early.sum() / max(1, n_d)),
        late_flag_rate=float(late.sum() / max(1, n_d)),
        median_latency=float(np.median(lat)) if len(lat) else float("nan"),
        latency_p25=float(np.percentile(lat, 25)) if len(lat) else float("nan"),
        latency_p75=float(np.percentile(lat, 75)) if len(lat) else float("nan"),
        in_study_precision=float(detected.sum() / max(1, flagged.sum())),
    )


def ppv_at_prevalence(se: float, fpr: float, prevalence: float) -> float:
    """Precision re-expressed at a real-world prevalence. In-study prevalence (40%)
    is set for statistical power and would flatter any precision figure."""
    den = prevalence * se + (1 - prevalence) * fpr
    return float(prevalence * se / den) if den > 0 else float("nan")


def detected_mask(people, flag_day, days, grace=GRACE_DAYS):
    onset = people.onset.to_numpy()
    end = np.minimum(days - 1, onset + people.ramp.to_numpy() + grace)
    return (people.drift.to_numpy() == 1) & (flag_day >= onset) & (flag_day <= end)


def bootstrap_ci(people, flag_day, days, start, n_boot=1000, seed=0, key="detection_rate"):
    rng = np.random.default_rng(seed)
    P = len(people)
    vals = []
    pp = people.reset_index(drop=True)
    for _ in range(n_boot):
        ix = rng.integers(0, P, P)
        sub = pp.iloc[ix].reset_index(drop=True)
        vals.append(score_detection(sub, np.asarray(flag_day)[ix], days, start)[key])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def what_stable_flags_hit(people, daily, flag_day, days, after=7) -> dict:
    """For never-declining people who were flagged: was a bad week behind it?"""
    P = len(people)
    dip = daily.dip.to_numpy().reshape(P, days).astype(bool)
    stable = people.drift.to_numpy() == 0
    hits, n = 0, 0
    for i in np.flatnonzero(stable & (flag_day >= 0)):
        d = flag_day[i]
        n += 1
        if dip[i, max(0, d - after):d + 1].any():
            hits += 1
    # base rate: share of monitored stable person-days within `after` days of a dip
    near = np.zeros_like(dip)
    for lag in range(after + 1):
        near[:, lag:] |= dip[:, :days - lag] if lag else dip
    return dict(n_flagged=n, share_after_bad_week=float(hits / n) if n else float("nan"),
                base_rate_near_bad_week=float(near[stable].mean()))


def by_group(people, flag_day, days, start, col, bins, labels) -> pd.DataFrame:
    d = people[people.drift == 1].copy()
    d["_flag"] = np.asarray(flag_day)[d.index]
    d["_grp"] = pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for g, sub in d.groupby("_grp", observed=False):
        if len(sub) == 0:
            continue
        r = score_detection(sub.reset_index(drop=True), sub._flag.to_numpy(), days, start)
        rows.append(dict(group=str(g), n=len(sub), detection_rate=r["detection_rate"],
                         median_latency=r["median_latency"]))
    return pd.DataFrame(rows)
