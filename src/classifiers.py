"""Sentence-level emotion classifiers.

Four arms, from no learning to a fine-tuned transformer:

* ``lexicon``   -- fixed keyword list (src/lexicon.py), no training.
* ``tfidf_lr``  -- TF-IDF unigrams+bigrams -> logistic regression.
* ``embed_lr``  -- frozen all-MiniLM-L6-v2 sentence embeddings -> logistic regression.
* ``minilm_ft`` -- all-MiniLM-L6-v2 fine-tuned end-to-end (mean pooling + linear
                   head), 2 epochs on CPU. This is the series' "fine-tuned
                   transformer" arm; DistilBERT is ~3x larger and was not
                   practical on this 7.5 GB CPU machine.

Hyper-parameters for the two linear arms are picked by 3-fold cross-validation
on the TRAINING split only (3 folds, small grids: this machine has 7.5 GB RAM
and 5-fold parallel CV exhausted it). The fine-tuning recipe is fixed in advance. The
validation and test splits are never used for selection -- they become the
sentence pools the synthetic diaries are written from.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from .data import CLASSES, ROOT

CACHE = ROOT / "artifacts" / "cache"
FT_DIR = ROOT / "artifacts" / "minilm_ft"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _key(texts, tag: str) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(tag.encode())
    for t in texts:
        h.update(str(t).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


# ----------------------------------------------------------------- linear arms
def fit_tfidf(texts, labels, seed=0, grid=(3.0, 10.0, 30.0, 100.0)):
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    rows = []
    for C in grid:
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                             LogisticRegression(C=C, max_iter=2000))
        # n_jobs=1 on purpose: parallel workers copy the data and exhausted RAM on 7.5 GB
        s = cross_val_score(pipe, list(texts), labels, cv=cv, scoring="f1_macro", n_jobs=1)
        rows.append(dict(C=C, cv_macro_f1=float(s.mean())))
    best = max(rows, key=lambda r: r["cv_macro_f1"])
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=best["C"], max_iter=2000))
    pipe.fit(list(texts), labels)
    return pipe, rows


def fit_embed_lr(X, labels, seed=0, grid=(1.0, 10.0)):
    """C chosen on a single 80/20 holdout carved from the TRAINING split.
    (3-fold CV on dense embeddings thrashed the page file on this machine.)"""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    labels = np.asarray(labels)
    ix_a, ix_b = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels,
                                  random_state=seed)
    rows = []
    for C in grid:
        clf = LogisticRegression(C=C, max_iter=1000).fit(X[ix_a], labels[ix_a])
        rows.append(dict(C=C, cv_macro_f1=float(f1_score(labels[ix_b], clf.predict(X[ix_b]),
                                                         average="macro"))))
    best = max(rows, key=lambda r: r["cv_macro_f1"])
    clf = LogisticRegression(C=best["C"], max_iter=3000).fit(X, labels)
    return clf, rows


_ST = None


def encoder():
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer(MODEL_NAME, device="cpu")
    return _ST


def embed(texts, use_cache=True) -> np.ndarray:
    texts = [str(t) for t in texts]
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"emb_{_key(texts, MODEL_NAME)}.npy"
    if use_cache and p.exists():
        return np.load(p)
    X = encoder().encode(texts, batch_size=128, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    if use_cache:
        np.save(p, X)
    return X


def discriminative_tokens(texts, labels, k=100) -> list[str]:
    """The k unigrams most associated with the label on the TRAINING split (chi-squared)."""
    cv = CountVectorizer(binary=True, min_df=5, token_pattern=r"[a-z']+")
    X = cv.fit_transform([str(t).lower() for t in texts])
    scores, _ = chi2(X, labels)
    vocab = np.array(cv.get_feature_names_out())
    order = np.argsort(-np.nan_to_num(scores))
    return vocab[order[:k]].tolist()


def mask_tokens(texts, tokens) -> list[str]:
    bad = set(tokens)
    return [" ".join("___" if w in bad else w for w in str(t).lower().split()) for t in texts]


# ----------------------------------------------------------- fine-tuned MiniLM
def _torch_net():
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self, enc, n_out):
            super().__init__()
            self.enc = enc
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(enc.config.hidden_size, n_out)

        def forward(self, input_ids, attention_mask):
            h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            return self.head(self.drop(pooled))

    return torch, Net


def finetune(texts, labels, epochs=2, lr=5e-5, batch=32, max_len=64, seed=0,
             out_dir: Path = FT_DIR, log=print) -> dict:
    torch, Net = _torch_net()
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(12, (os.cpu_count() or 4) - 2)))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    net = Net(AutoModel.from_pretrained(MODEL_NAME), len(CLASSES))
    texts = [str(t) for t in texts]
    y = torch.tensor(np.asarray(labels, int))
    n = len(texts)
    steps = epochs * int(np.ceil(n / batch))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    lossf = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    t0, step, hist = time.time(), 0, []
    net.train()
    for ep in range(epochs):
        order = rng.permutation(n)
        run = 0.0
        for b in range(0, n, batch):
            idx = order[b:b + batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            logits = net(enc["input_ids"], enc["attention_mask"])
            loss = lossf(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
            if step % 100 == 0:
                log(f"      fine-tune step {step}/{steps}  loss {run / 100:.3f}  "
                    f"{time.time() - t0:.0f}s")
                hist.append(dict(step=step, loss=run / 100))
                run = 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    net.enc.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    torch.save(net.head.state_dict(), out_dir / "head.pt")
    info = dict(epochs=epochs, lr=lr, batch=batch, max_len=max_len, steps=steps,
                seconds=round(time.time() - t0, 1), loss_history=hist)
    return info


_FT = None


def load_finetuned(model_dir: Path = FT_DIR):
    global _FT
    if _FT is None:
        torch, Net = _torch_net()
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        net = Net(AutoModel.from_pretrained(model_dir), len(CLASSES))
        net.head.load_state_dict(torch.load(model_dir / "head.pt", map_location="cpu"))
        net.eval()
        _FT = (torch, tok, net)
    return _FT


def finetuned_available(model_dir: Path = FT_DIR) -> bool:
    return (model_dir / "head.pt").exists()


def predict_finetuned(texts, model_dir: Path = FT_DIR, batch=128, max_len=64,
                      use_cache=True) -> np.ndarray:
    texts = [str(t) for t in texts]
    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = str((model_dir / "head.pt").stat().st_mtime_ns) if use_cache else ""
    p = CACHE / f"ft_{_key(texts, 'ft' + stamp)}.npy"
    if use_cache and p.exists():
        return np.load(p)
    torch, tok, net = load_finetuned(model_dir)
    out = []
    with torch.no_grad():
        for b in range(0, len(texts), batch):
            enc = tok(texts[b:b + batch], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt")
            out.append(torch.softmax(net(enc["input_ids"], enc["attention_mask"]), -1).numpy())
    P = np.concatenate(out) if out else np.zeros((0, len(CLASSES)))
    if use_cache:
        np.save(p, P)
    return P
