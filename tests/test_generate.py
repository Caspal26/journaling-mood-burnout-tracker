"""The generator: determinism, and the structural guarantee that the label is not in the text."""

import inspect
from dataclasses import replace

import numpy as np
import pytest

from src.data import CLASSES, NEG_IDX
from src.evaluate import roc_auc
from src.journal import JournalConfig, build_cohort, class_probs, person_latent, render_entries

# a synthetic pool with plenty of every class, so these tests need no download
POOL = np.repeat(np.arange(len(CLASSES)), 400)


@pytest.fixture(scope="module")
def cohort():
    return build_cohort(JournalConfig(n_people=160, seed=3), POOL)


def test_render_entries_cannot_see_the_drift_flag():
    params = inspect.signature(render_entries).parameters
    assert not any("drift" in p or "onset" in p or "depth" in p or "ramp" in p for p in params)


def test_swapping_the_flag_changes_nothing_but_the_decline():
    cfg = JournalConfig(n_people=1, seed=11)
    lat_d, dip_d, meta_d = person_latent(cfg, 0, True)
    lat_s, dip_s, meta_s = person_latent(cfg, 0, False)
    onset = meta_d["onset"]
    assert np.array_equal(lat_d[:onset + 1], lat_s[:onset + 1])     # same person before onset
    assert np.array_equal(dip_d, dip_s)                              # same bad weeks
    assert (lat_d[onset + 1:] < lat_s[onset + 1:]).all()             # and only the decline differs
    # the SAME mood path renders byte-identical diaries whatever the label says
    a = render_entries(cfg, 0, lat_s, meta_s["mu"], meta_s["gain"], meta_s["adherence"], POOL)
    b = render_entries(cfg, 0, lat_s.copy(), meta_d["mu"], meta_d["gain"], meta_d["adherence"], POOL)
    assert np.array_equal(a[0], b[0]) and a[3] == b[3]


def test_cohort_is_deterministic():
    cfg = JournalConfig(n_people=30, seed=5)
    a, b = build_cohort(cfg, POOL), build_cohort(cfg, POOL)
    assert a[2].equals(b[2]) and a[1].equals(b[1])


def test_seed_does_not_use_salted_builtin_hash():
    src = inspect.getsource(__import__("src.journal", fromlist=["_rng"]))
    assert "blake2b" in src and "hash((" not in src


def test_swapping_the_sentence_pool_changes_only_the_text():
    cfg = JournalConfig(n_people=40, seed=8)
    other = np.repeat(np.arange(len(CLASSES)), 150)[::-1].copy()
    a, b = build_cohort(cfg, POOL), build_cohort(cfg, other)
    assert np.array_equal(a[1].wrote, b[1].wrote)
    assert np.array_equal(a[2].true_class, b[2].true_class)
    assert np.allclose(a[1].latent, b[1].latent)


def test_lower_mood_means_more_negative_emotion():
    p = class_probs(np.array([-2.0, 0.0, 2.0]), 1.0)
    neg = p[:, NEG_IDX].sum(1)
    assert neg[0] > neg[1] > neg[2]


def test_sweeping_writing_behaviour_does_not_change_anyones_mood():
    cfg = JournalConfig(n_people=40, seed=9)
    a = build_cohort(cfg, POOL)
    b = build_cohort(replace(cfg, mood_dropout=0.0), POOL)
    assert np.allclose(a[1].latent, b[1].latent)
    assert not np.array_equal(a[1].wrote, b[1].wrote)


def test_with_mood_independent_writing_entry_count_cannot_reveal_a_decline():
    ppl, _, _ = build_cohort(JournalConfig(n_people=400, seed=21, mood_dropout=0.0), POOL)
    auc = roc_auc(ppl.drift, -ppl.n_entries)
    assert 0.40 < auc < 0.60


def test_people_who_decline_write_less_when_writing_follows_mood(cohort):
    ppl, daily, _ = cohort
    d = daily.merge(ppl[["person_id", "onset", "ramp", "drift"]], on="person_id")
    after = d[(d.drift == 1) & (d.day >= d.onset + d.ramp)].wrote.mean()
    stable = d[(d.drift == 0) & (d.day >= 35)].wrote.mean()
    assert after < stable


def test_bad_weeks_happen_to_everyone_equally(cohort):
    ppl = cohort[0]
    a = ppl[ppl.drift == 1].n_dips.mean()
    b = ppl[ppl.drift == 0].n_dips.mean()
    assert abs(a - b) < 0.5


def test_a_bad_week_is_as_deep_as_a_decline():
    cfg = JournalConfig()
    assert cfg.dip_depth_min == cfg.depth_min and cfg.dip_depth_max == cfg.depth_max


def test_onset_leaves_a_clean_monitoring_start():
    cfg = JournalConfig()
    assert cfg.onset_min > cfg.baseline_days
