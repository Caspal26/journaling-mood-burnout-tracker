"""Corpora, pools and the no-learning baseline."""

import numpy as np
import pandas as pd
import pytest

from src import classifiers as C
from src import data as DATA
from src import lexicon as LEX


def test_valence_is_positive_minus_negative_and_ignores_surprise():
    P = np.eye(6)
    v = DATA.valence(P)
    assert list(v) == [-1, 1, 1, -1, -1, 0]


def test_dedupe_drops_normalised_duplicates_only():
    train = pd.DataFrame(dict(text=["I feel  Happy", "i am sad"], label=[1, 0]))
    pool = pd.DataFrame(dict(text=["i feel happy", "i feel great"], label=[1, 1]))
    kept, dropped = DATA.dedupe_against(pool, train)
    assert dropped == 1 and kept.text.tolist() == ["i feel great"]


def test_goemotions_mapping_only_targets_the_six_classes():
    assert set(DATA.GOEMOTIONS_TO_DAIR.values()) <= set(DATA.CLASSES)
    assert set(DATA.GOEMOTIONS_TO_DAIR) <= set(DATA.GOEMOTIONS_LABELS)
    assert "neutral" not in DATA.GOEMOTIONS_TO_DAIR and "disgust" not in DATA.GOEMOTIONS_TO_DAIR


def test_lexicon_reads_obvious_words_and_falls_back_to_the_prior():
    p = LEX.predict_proba(["i feel so lonely and hopeless", "i feel furious", "the bus was late"])
    assert p[0].argmax() == DATA.CLASSES.index("sadness")
    assert p[1].argmax() == DATA.CLASSES.index("anger")
    assert np.allclose(p[2], LEX.PRIOR)


def test_no_word_is_listed_under_two_emotions():
    seen = {}
    for c, words in LEX.LEXICON.items():
        for w in words:
            assert w not in seen, f"{w} in {seen.get(w)} and {c}"
            seen[w] = c


def test_masking_replaces_whole_tokens_only():
    out = C.mask_tokens(["i feel sad and saddened"], ["sad"])
    assert out == ["i feel ___ and saddened"]


@pytest.mark.skipif(not (DATA.RAW / "emotion_test.parquet").exists(), reason="corpora not downloaded")
def test_diary_pools_share_no_sentence_with_training_data():
    em = DATA.load_emotion()
    seen = set(em["train"].text.map(DATA.normalise))
    for split in ("validation", "test"):
        pool, _ = DATA.dedupe_against(em[split], em["train"])
        assert not pool.text.map(DATA.normalise).isin(seen).any()
    go, _ = DATA.dedupe_against(DATA.load_goemotions(), em["train"])
    assert not go.text.map(DATA.normalise).isin(seen).any()


@pytest.mark.skipif(not (DATA.RAW / "goemotions_test.parquet").exists(), reason="corpora not downloaded")
def test_reddit_pool_has_every_class_so_diaries_can_be_written_from_it():
    go = DATA.load_goemotions()
    assert set(go.label.unique()) == set(range(len(DATA.CLASSES)))
