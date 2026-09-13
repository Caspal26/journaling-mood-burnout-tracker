"""Detector properties, with negative controls."""

import numpy as np
import pandas as pd

from src import tracker as T
from src.evaluate import score_detection, within_person_r
from src.journal import JournalConfig, build_cohort
from src.data import CLASSES, onehot

CFG = JournalConfig(n_people=4, days=120, baseline_days=30)


def _flat(P=4, D=120, level=0.2, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return level + noise * rng.standard_normal((P, D))


def test_cusum_stays_at_zero_on_an_unchanging_person():
    M = _flat()
    mu, sd, _, _ = T.personal_baseline(M, 30, pop_sd=0.3)
    S = T.cusum(M, mu, sd, 30, k=0.5)
    assert np.all(S == 0)


def test_cusum_finds_a_sustained_drop_and_bigger_drops_are_found_sooner():
    M = _flat(P=3, noise=0.2, seed=1)
    for i, drop in enumerate((0.0, 0.3, 0.8)):
        M[i, 60:] -= drop
    mu, sd, _, _ = T.personal_baseline(M, 30)
    S = T.cusum(M, mu, sd, 30)
    h = 5.0
    ff = T.first_flag(S, h, 30)
    assert ff[0] == -1 or ff[0] < 60     # no drop: any flag is noise, never "after onset"
    assert ff[1] >= 60 and ff[2] >= 60
    assert ff[2] < ff[1]


def test_missing_days_carry_the_statistic_rather_than_resetting_it():
    M = _flat(P=1, D=60)
    M[0, 30:40] = -1.0
    M[0, 40:50] = np.nan
    mu, sd, _, _ = T.personal_baseline(M, 30, pop_sd=0.3)
    S = T.cusum(M, mu, sd, 30)
    assert np.all(S[0, 40:50] == S[0, 39]) and S[0, 39] > 0


def test_threshold_respects_the_budget_in_sample():
    rng = np.random.default_rng(2)
    S = np.cumsum(np.abs(rng.standard_normal((200, 90))), 1)
    stable = np.ones(200, bool)
    for budget in (0.05, 0.1, 0.3):
        h = T.fit_threshold(S, stable, budget, 10)
        assert (S[:, 10:].max(1) > h).mean() <= budget


def test_first_flag_is_minus_one_when_nothing_crosses():
    S = np.zeros((3, 50))
    assert (T.first_flag(S, 1.0, 10) == -1).all()


def test_personal_cusum_ignores_a_persons_level_but_the_population_rule_does_not():
    """The property the register-shift result depends on."""
    rng = np.random.default_rng(4)
    M = 0.3 * rng.standard_normal((2, 120))
    M[1] = M[0] - 0.8                            # identical person, gloomier register
    mu, sd, _, _ = T.personal_baseline(M, 30, pop_sd=0.3)
    S = T.cusum(M, mu, sd, 30)
    assert np.allclose(S[0], S[1])
    P = T.population_rolling(M, 30)
    assert (P[1, 30:] > P[0, 30:]).all()


def test_shrinkage_pulls_a_thin_baseline_toward_the_cohort():
    M = np.full((2, 40), np.nan)
    M[0, :3] = [0.0, 0.01, 0.02]                  # 3 near-identical entries: raw sd ~0.01
    M[1, :30] = np.random.default_rng(0).normal(0, 0.01, 30)
    _, sd, _, _ = T.personal_baseline(M, 30, pop_sd=0.5)
    assert sd[0] > sd[1] > 0.0
    assert sd[0] > 0.3


def test_within_person_r_perfect_and_shuffled():
    rng = np.random.default_rng(1)
    L = rng.standard_normal((50, 100)) + rng.standard_normal((50, 1)) * 3
    assert within_person_r(L.copy(), L) > 0.999
    Ms = L.copy()
    for i in range(len(Ms)):
        Ms[i] = Ms[i][rng.permutation(100)]
    assert abs(within_person_r(Ms, L)) < 0.05


def _detect(M_cal, ppl_cal, M_ev, ppl_ev, cfg):
    S_c, b = T.run(M_cal, cfg)
    h = T.fit_threshold(S_c, ppl_cal.drift.to_numpy() == 0, 0.1, cfg.baseline_days)
    S_e, _ = T.run(M_ev, cfg, pop_sd=b["pop_sd"])
    return score_detection(ppl_ev, T.first_flag(S_e, h, cfg.baseline_days), cfg.days,
                           cfg.baseline_days)


def test_negative_controls_order_correctly_perfect_mood_beats_gold_text_beats_random():
    pool = np.repeat(np.arange(len(CLASSES)), 300)
    cc, ce = JournalConfig(n_people=400, seed=31), JournalConfig(n_people=400, seed=32)
    kc, ke = build_cohort(cc, pool), build_cohort(ce, pool)
    lat = [T.latent_matrix(k[1], 400, cc.days) for k in (kc, ke)]
    gold = [T.score_matrix(k[0], k[2], onehot(pool), cc.days) for k in (kc, ke)]
    rng = np.random.default_rng(0)
    rnd = rng.dirichlet(np.ones(len(CLASSES)), len(pool))
    rand = [T.score_matrix(k[0], k[2], rnd, cc.days) for k in (kc, ke)]
    d_lat = _detect(lat[0], kc[0], lat[1], ke[0], ce)["detection_rate"]
    d_gold = _detect(gold[0], kc[0], gold[1], ke[0], ce)["detection_rate"]
    d_rand = _detect(rand[0], kc[0], rand[1], ke[0], ce)["detection_rate"]
    assert d_lat > d_gold > d_rand
    assert d_rand < 0.15


def test_shuffling_the_days_destroys_a_perfect_mood_signal():
    pool = np.repeat(np.arange(len(CLASSES)), 300)
    cc, ce = JournalConfig(n_people=300, seed=41), JournalConfig(n_people=300, seed=42)
    kc, ke = build_cohort(cc, pool), build_cohort(ce, pool)
    Mc = T.latent_matrix(kc[1], 300, cc.days)
    Me = T.latent_matrix(ke[1], 300, ce.days)
    rng = np.random.default_rng(1)
    Msh = Me.copy()
    B = ce.baseline_days
    for i in range(len(Msh)):
        Msh[i, B:] = Msh[i, B:][rng.permutation(ce.days - B)]
    real = _detect(Mc, kc[0], Me, ke[0], ce)["detection_rate"]
    shuf = _detect(Mc, kc[0], Msh, ke[0], ce)["detection_rate"]
    assert real > 0.6 and shuf < real / 2
