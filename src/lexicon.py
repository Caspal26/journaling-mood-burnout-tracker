"""A fixed, hand-written emotion keyword lexicon -- the no-learning baseline.

Written once, before any model was scored, and never tuned against the test
split. It exists to answer one question: how far does "does the sentence
contain an emotion word?" get you on a corpus whose labels were themselves
derived from emotion hashtags?
"""

from __future__ import annotations

import re

import numpy as np

from .data import CLASSES

LEXICON = {
    "sadness": ["sad", "unhappy", "depressed", "lonely", "miserable", "hopeless", "heartbroken",
                "gloomy", "hurt", "crying", "empty", "drained", "exhausted", "worthless",
                "grief", "lost", "defeated", "helpless", "numb", "disappointed", "regret",
                "ashamed", "guilty", "burned", "burnt", "tired"],
    "joy": ["happy", "glad", "excited", "cheerful", "grateful", "content", "great", "wonderful",
            "delighted", "proud", "relieved", "pleased", "energetic", "hopeful", "joyful",
            "optimistic", "confident", "satisfied", "peaceful", "amazing", "fun", "good"],
    "love": ["love", "loving", "caring", "affectionate", "tender", "sweet", "romantic", "adore",
             "passionate", "supportive", "cherished", "loved", "fond", "warm", "sympathetic"],
    "anger": ["angry", "mad", "furious", "irritated", "annoyed", "frustrated", "resentful",
              "bitter", "hate", "rage", "pissed", "outraged", "hostile", "jealous", "offended",
              "cranky", "irritable", "grumpy"],
    "fear": ["afraid", "scared", "anxious", "nervous", "worried", "terrified", "frightened",
             "panicked", "overwhelmed", "uneasy", "insecure", "stressed", "tense", "dread",
             "paranoid", "restless", "vulnerable", "threatened"],
    "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned", "curious", "impressed",
                 "startled", "weird", "strange", "unexpected", "funny"],
}

_TOKEN = re.compile(r"[a-z']+")
_INDEX = {}
for _c, _words in LEXICON.items():
    for _w in _words:
        _INDEX.setdefault(_w, []).append(CLASSES.index(_c))

# Class frequencies of the dair-ai training split, used only to break ties and
# to give a sentence with no keyword the prior instead of a made-up answer.
PRIOR = np.array([0.2916, 0.3351, 0.0815, 0.1349, 0.1211, 0.0358])


def predict_proba(texts) -> np.ndarray:
    out = np.zeros((len(texts), len(CLASSES)))
    for i, t in enumerate(texts):
        counts = np.zeros(len(CLASSES))
        for tok in _TOKEN.findall(str(t).lower()):
            for c in _INDEX.get(tok, ()):
                counts[c] += 1.0
        if counts.sum() == 0:
            out[i] = PRIOR
        else:
            p = counts + 0.01 * PRIOR          # prior only breaks ties
            out[i] = p / p.sum()
    return out


def coverage(texts) -> float:
    """Share of sentences containing at least one lexicon word."""
    hit = 0
    for t in texts:
        if any(tok in _INDEX for tok in _TOKEN.findall(str(t).lower())):
            hit += 1
    return hit / max(1, len(texts))
