"""Synthetic journalers written in real sentences.

Every *sequence* here is synthetic; every *sentence* is real public text.

A journaler has a latent daily mood. It is their own typical level, plus
day-to-day AR(1) wobble, plus occasional short "bad weeks" (transient dips that
recover), plus -- for some people -- a slow, sustained decline that starts on an
onset day, ramps over weeks and does not recover. The sustained decline is the
condition the tracker is meant to notice; the bad week is the hard negative.

On a given day the person may or may not write. If they write, each sentence's
emotion class is drawn from a distribution that depends on mood, and a real
sentence carrying that gold label is taken from a pool.

STRUCTURAL ANTI-SHORTCUT RULE (the series' recurring failure is a generator
that writes the label into the data): ``render_entries`` never receives the
drift flag or any drift parameter. Whether someone writes, how much, and what
they write is a function of their latent mood alone. A test swaps the flag and
requires byte-identical diaries.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import CLASSES
from .lexicon import PRIOR

# how strongly each emotion's log-odds moves per unit of latent mood
BETA = np.array([-0.9, 0.9, 0.7, -0.6, -0.7, 0.0])   # sadness joy love anger fear surprise


@dataclass(frozen=True)
class JournalConfig:
    n_people: int = 500
    days: int = 180
    baseline_days: int = 35
    drift_frac: float = 0.40
    onset_min: int = 60
    onset_max: int = 130
    ramp_min: int = 14
    ramp_max: int = 42
    depth_min: float = 0.6
    depth_max: float = 1.4
    baseline_sd: float = 0.6         # between-person spread of typical mood (ICC ~0.42)
    ar_phi: float = 0.4              # day-to-day inertia, in the range daily-diary studies report
    ar_sd: float = 0.64              # stationary sd = 0.64 / sqrt(1 - 0.16) = 0.70
    dip_rate: float = 1.5            # expected bad weeks per 180 days
    dip_len_min: int = 4
    dip_len_max: int = 10
    dip_depth_min: float = 0.6       # a bad week is exactly as deep as a decline;
    dip_depth_max: float = 1.4       # only its duration tells them apart
    text_coupling: float = 1.0       # multiplies BETA: how much mood leaks into word choice
    adherence_mean: float = 0.60
    adherence_conc: float = 6.0
    engagement_halflife: float = 240.0
    mood_dropout: float = 0.8        # log-odds of writing change per unit mood below own typical
    extra_sentences: float = 3.0     # sentences per entry = 1 + Poisson(this), ~4 per entry
    expressiveness_sd: float = 0.25
    seed: int = 16

    def as_dict(self):
        return asdict(self)


def _rng(seed: int, pid: int, purpose: str) -> np.random.Generator:
    """Stable per-person, per-purpose stream. NOT builtin hash() -- CPython salts it."""
    h = hashlib.blake2b(f"{seed}|{pid}|{purpose}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


def person_latent(cfg: JournalConfig, pid: int, drift: bool):
    """Latent mood path. Wobble and dips come from streams that ignore `drift`,
    so a person with and without a decline is the same person on every other day."""
    D = cfg.days
    r0 = _rng(cfg.seed, pid, "trait")
    mu = r0.normal(0.0, cfg.baseline_sd)
    gain = float(np.exp(r0.normal(0.0, cfg.expressiveness_sd)))
    adherence = float(r0.beta(cfg.adherence_mean * cfg.adherence_conc,
                              (1 - cfg.adherence_mean) * cfg.adherence_conc))

    r1 = _rng(cfg.seed, pid, "wobble")
    eps = r1.normal(0.0, cfg.ar_sd, D)
    ar = np.zeros(D)
    ar[0] = eps[0] / np.sqrt(1 - cfg.ar_phi ** 2)
    for t in range(1, D):
        ar[t] = cfg.ar_phi * ar[t - 1] + eps[t]

    r2 = _rng(cfg.seed, pid, "dips")
    dip = np.zeros(D)
    dip_mask = np.zeros(D, bool)
    for _ in range(r2.poisson(cfg.dip_rate * D / 180.0)):
        start = int(r2.integers(0, D))
        L = int(r2.integers(cfg.dip_len_min, cfg.dip_len_max + 1))
        depth = r2.uniform(cfg.dip_depth_min, cfg.dip_depth_max)
        end = min(D, start + L)
        # a bad week comes on over a couple of days and lifts over a couple more
        shape = np.minimum(1.0, np.minimum(np.arange(1, end - start + 1) / 2.0,
                                           np.arange(end - start, 0, -1) / 2.0))
        dip[start:end] = np.maximum(dip[start:end], depth * shape)
        dip_mask[start:end] = True

    r3 = _rng(cfg.seed, pid, "drift")
    onset = int(r3.integers(cfg.onset_min, cfg.onset_max + 1))
    ramp = int(r3.integers(cfg.ramp_min, cfg.ramp_max + 1))
    depth = float(r3.uniform(cfg.depth_min, cfg.depth_max))
    decline = np.zeros(D)
    if drift:
        t = np.arange(D)
        decline = depth * np.clip((t - onset) / ramp, 0.0, 1.0)

    latent = mu + ar - dip - decline
    meta = dict(person_id=pid, drift=int(drift), onset=onset if drift else -1,
                ramp=ramp if drift else -1, depth=depth if drift else 0.0,
                mu=float(mu), gain=gain, adherence=adherence)
    return latent, dip_mask, meta


def class_probs(mood: np.ndarray, gain: float) -> np.ndarray:
    logits = np.log(PRIOR)[None, :] + gain * np.outer(mood, BETA)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def render_entries(cfg: JournalConfig, pid: int, latent: np.ndarray, mu: float, gain: float,
                   adherence: float, pool_labels: np.ndarray):
    """Diary text for one person. Takes mood, never the drift flag.

    Three independent streams: whether/how much they write, which emotion each
    sentence carries, and which real sentence is picked. Swapping the sentence
    pool (e.g. tweets -> Reddit) therefore changes ONLY the text: same people,
    same days, same emotions."""
    D = cfg.days
    rw = _rng(cfg.seed, pid, "write")
    rc = _rng(cfg.seed, pid, "classes")
    rp = _rng(cfg.seed, pid, "pick")
    t = np.arange(D)
    engage = 0.5 ** (t / cfg.engagement_halflife)
    base = np.log(adherence / (1 - adherence))
    dev = latent - mu
    p_write = 1 / (1 + np.exp(-(base + np.log(engage) + cfg.mood_dropout * dev)))
    wrote = rw.random(D) < p_write
    n_sent = np.where(wrote, 1 + rw.poisson(cfg.extra_sentences, D), 0)

    probs = class_probs(latent, gain * cfg.text_coupling)
    cum = probs.cumsum(1)
    u = rc.random((D, int(n_sent.max()) if n_sent.max() > 0 else 1))
    by_class = [np.flatnonzero(pool_labels == c) for c in range(len(CLASSES))]
    perm = [rp.permutation(ix) for ix in by_class]
    cursor = [0] * len(CLASSES)
    rows = []
    for d in np.flatnonzero(wrote):
        for j in range(int(n_sent[d])):
            c = int(min(len(CLASSES) - 1, np.searchsorted(cum[d], u[d, j], side="right")))
            if len(perm[c]) == 0:
                raise ValueError(f"sentence pool has no '{CLASSES[c]}' sentences")
            # within a person, a sentence is not reused until the class pool is exhausted
            if cursor[c] >= len(perm[c]):
                perm[c] = rp.permutation(by_class[c])
                cursor[c] = 0
            rows.append((pid, int(d), int(perm[c][cursor[c]]), c))
            cursor[c] += 1
    return wrote.astype(np.int8), n_sent.astype(np.int16), p_write, rows


def build_cohort(cfg: JournalConfig, pool_labels: np.ndarray):
    pool_labels = np.asarray(pool_labels, int)
    rp = np.random.default_rng(cfg.seed)
    drift_flags = rp.random(cfg.n_people) < cfg.drift_frac
    people, daily, sents = [], [], []
    for pid in range(cfg.n_people):
        latent, dip_mask, meta = person_latent(cfg, pid, bool(drift_flags[pid]))
        wrote, n_sent, p_write, rows = render_entries(
            cfg, pid, latent, meta["mu"], meta["gain"], meta["adherence"], pool_labels)
        meta["n_entries"] = int(wrote.sum())
        meta["n_dips"] = int(np.sum(np.diff(np.r_[0, dip_mask.astype(int)]) == 1))
        people.append(meta)
        drifting = np.zeros(cfg.days, np.int8)
        if meta["drift"]:
            drifting[meta["onset"]:] = 1
        daily.append(pd.DataFrame(dict(person_id=pid, day=np.arange(cfg.days), latent=latent,
                                       dev=latent - meta["mu"], dip=dip_mask.astype(np.int8),
                                       drifting=drifting, wrote=wrote, n_sent=n_sent,
                                       p_write=p_write)))
        sents.extend(rows)
    people = pd.DataFrame(people)
    daily = pd.concat(daily, ignore_index=True)
    sents = pd.DataFrame(sents, columns=["person_id", "day", "pool_idx", "true_class"])
    return people, daily, sents
