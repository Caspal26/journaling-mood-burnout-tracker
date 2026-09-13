"""Public text corpora: loading, label mapping, and the leakage-safe sentence pools.

Two real, public, non-clinical corpora are used. Neither contains patient,
therapy or crisis-line data.

* dair-ai/emotion (Saravia et al., 2018) -- English tweets labelled with six
  emotions by hashtag distant supervision. 16,000 / 2,000 / 2,000.
  Card: "for educational and research purposes only".
* GoEmotions (Demszky et al., 2020, Apache-2.0) -- Reddit comments labelled
  by raters with 27 emotions + neutral. Used here ONLY as a register-shift
  test set, mapped onto the six dair-ai classes.

The raw parquet files are downloaded on first use into data/raw/ and are not
committed to git.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NEGATIVE = ("sadness", "anger", "fear")
POSITIVE = ("joy", "love")
NEG_IDX = [CLASSES.index(c) for c in NEGATIVE]
POS_IDX = [CLASSES.index(c) for c in POSITIVE]

_HF = "https://huggingface.co/datasets"
SOURCES = {
    "emotion_train.parquet": f"{_HF}/dair-ai/emotion/resolve/main/split/train-00000-of-00001.parquet",
    "emotion_validation.parquet": f"{_HF}/dair-ai/emotion/resolve/main/split/validation-00000-of-00001.parquet",
    "emotion_test.parquet": f"{_HF}/dair-ai/emotion/resolve/main/split/test-00000-of-00001.parquet",
    "goemotions_test.parquet": f"{_HF}/google-research-datasets/go_emotions/resolve/main/simplified/test-00000-of-00001.parquet",
    "goemotions_validation.parquet": f"{_HF}/google-research-datasets/go_emotions/resolve/main/simplified/validation-00000-of-00001.parquet",
}

# GoEmotions "simplified" label ids, in the order the dataset publishes them.
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]

# The GoEmotions paper's own Ekman grouping, with dair-ai's extra "love" class
# split back out of joy. Disgust and neutral have no dair-ai counterpart and are
# dropped rather than forced into a class.
GOEMOTIONS_TO_DAIR = {
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "fear": "fear", "nervousness": "fear",
    "joy": "joy", "amusement": "joy", "approval": "joy", "excitement": "joy",
    "gratitude": "joy", "optimism": "joy", "relief": "joy", "pride": "joy",
    "admiration": "joy",
    "love": "love", "caring": "love", "desire": "love",
    "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness",
    "grief": "sadness", "remorse": "sadness",
    "surprise": "surprise", "realization": "surprise", "confusion": "surprise",
    "curiosity": "surprise",
}


def ensure_raw() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        p = RAW / name
        if not p.exists():
            print(f"      downloading {name} ...")
            urllib.request.urlretrieve(url, p)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def load_emotion() -> dict[str, pd.DataFrame]:
    ensure_raw()
    out = {}
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(RAW / f"emotion_{split}.parquet")
        out[split] = pd.DataFrame({"text": df["text"].astype(str),
                                   "label": df["label"].astype(int)}).reset_index(drop=True)
    return out


def load_goemotions() -> pd.DataFrame:
    """Single-label GoEmotions rows whose label maps onto a dair-ai class."""
    ensure_raw()
    parts = [pd.read_parquet(RAW / f"goemotions_{s}.parquet") for s in ("validation", "test")]
    df = pd.concat(parts, ignore_index=True)
    rows = []
    for text, labels in zip(df["text"], df["labels"]):
        labels = list(labels)
        if len(labels) != 1:
            continue
        name = GOEMOTIONS_LABELS[int(labels[0])]
        if name not in GOEMOTIONS_TO_DAIR:
            continue
        rows.append((str(text), CLASSES.index(GOEMOTIONS_TO_DAIR[name]), name))
    return pd.DataFrame(rows, columns=["text", "label", "source_label"])


def dedupe_against(pool: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop pool sentences whose normalised text also occurs in training data.

    A diary built from a sentence the classifier was trained on would score that
    sentence from memory, not from reading it.
    """
    seen = set(train["text"].map(normalise))
    keep = ~pool["text"].map(normalise).isin(seen)
    return pool[keep].reset_index(drop=True), int((~keep).sum())


def valence(probs: np.ndarray) -> np.ndarray:
    """Positive minus negative probability mass, in [-1, 1]. Surprise is neutral."""
    probs = np.asarray(probs, float)
    return probs[:, POS_IDX].sum(1) - probs[:, NEG_IDX].sum(1)


def onehot(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, int)
    out = np.zeros((len(labels), len(CLASSES)))
    out[np.arange(len(labels)), labels] = 1.0
    return out
