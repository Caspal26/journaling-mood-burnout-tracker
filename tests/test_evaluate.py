"""Scoring rules, pinned on hand-built fixtures."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import evaluate as EV


def _people():
    # two declining people (onset 60, ramp 20) and two who never decline
    return pd.DataFrame(dict(person_id=[0, 1, 2, 3], drift=[1, 1, 0, 0],
                             onset=[60, 60, -1, -1], ramp=[20, 20, -1, -1]))


def test_a_flag_before_onset_is_a_false_flag_not_a_detection():
    r = EV.score_detection(_people(), np.array([50, 70, -1, -1]), 180, 35)
    assert r["n_detected"] == 1 and r["early_flag_rate"] == 0.5
    assert r["median_latency"] == 10


def test_a_flag_after_the_grace_window_does_not_count():
    r = EV.score_detection(_people(), np.array([60 + 20 + EV.GRACE_DAYS + 1, 60, -1, -1]), 180, 35)
    assert r["n_detected"] == 1 and r["late_flag_rate"] == 0.5


def test_stable_people_flagged_count_against_the_budget():
    r = EV.score_detection(_people(), np.array([-1, -1, 100, -1]), 180, 35)
    assert r["stable_flag_rate"] == 0.5 and r["detection_rate"] == 0.0


def test_flags_during_the_baseline_window_are_ignored():
    r = EV.score_detection(_people(), np.array([-1, -1, 10, -1]), 180, 35)
    assert r["stable_flag_rate"] == 0.0


def test_roc_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.integers(0, 5, 300).astype(float)
    assert abs(EV.roc_auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_ppv_reduces_to_empirical_precision_at_the_sample_prevalence():
    se, fpr, pi = 0.6, 0.1, 0.4
    n = 10000
    tp, fp = se * pi * n, fpr * (1 - pi) * n
    assert abs(EV.ppv_at_prevalence(se, fpr, pi) - tp / (tp + fp)) < 1e-12
    assert EV.ppv_at_prevalence(se, fpr, 0.05) < EV.ppv_at_prevalence(se, fpr, 0.4)


def test_classification_metrics_on_a_perfect_and_a_constant_classifier():
    y = np.array([0, 1, 2, 3, 4, 5] * 20)
    perfect = np.eye(6)[y]
    r = EV.classification(y, perfect)
    assert r["accuracy"] == 1.0 and r["valence_auc"] == 1.0
    const = np.tile(np.eye(6)[1], (len(y), 1))
    r = EV.classification(y, const)
    assert abs(r["valence_auc"] - 0.5) < 1e-9
