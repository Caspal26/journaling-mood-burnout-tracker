"""Render the study figures from artifacts/metrics.json + artifacts/study.pkl.gz."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG = ART / "figures"

INK, MUTED, ACCENT, WARN, HIGH, LINE = "#0f172a", "#5b6b82", "#6d28d9", "#b45309", "#dc2626", "#e6ebf2"
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#94a3b8", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.dpi": 110})


def load():
    m = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    with gzip.open(ART / "study.pkl.gz", "rb") as f:
        s = pickle.load(f)
    return m, s


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_ladder(m):
    lad = pd.DataFrame(m["ladder"])
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(lad))[::-1]
    cols = [ACCENT if a == m["config"]["shipped_arm"] else ("#94a3b8" if a == "random" else "#a78bfa")
            for a in lad.arm]
    ax.barh(y, lad.detection_rate * 100, color=cols)
    for yi, r in zip(y, lad.itertuples()):
        ax.text(r.detection_rate * 100 + 1, yi, f"{r.detection_rate:.0%}  ·  {r.stable_flag_rate:.0%} of "
                f"stable people flagged", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(y, lad.label)
    ax.set_xlim(0, 118)
    ax.set_xlabel("sustained declines caught (evaluation cohort)")
    ax.set_title("Where detection is lost: from true mood to a classifier", loc="left", color=INK)
    _save(fig, "ladder")


def fig_accuracy_vs_detection(m):
    lad = pd.DataFrame(m["ladder"]).dropna(subset=["sentence_accuracy"])
    lad = lad[lad.arm != "random"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter(lad.sentence_accuracy * 100, lad.detection_rate * 100, s=70, color=ACCENT, zorder=3)
    for r in lad.itertuples():
        ax.annotate(r.arm, (r.sentence_accuracy * 100, r.detection_rate * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=9, color=INK)
    ax.set_xlabel("sentence-level accuracy, 6 emotions (%)")
    ax.set_ylabel("declines caught (%)")
    ax.set_title("Sentence accuracy vs. what the tracker catches", loc="left")
    ax.grid(color=LINE)
    _save(fig, "accuracy_vs_detection")


def fig_classifier(m):
    rows = []
    for setting, title in [("in_domain", "tweets (in-domain)"), ("keywords_masked", "top-100 keywords masked"),
                           ("reddit_transport", "Reddit (register shift)")]:
        for arm, r in m["classification"][setting].items():
            rows.append(dict(setting=title, arm=arm, accuracy=r["accuracy"], valence_auc=r["valence_auc"]))
    df = pd.DataFrame(rows)
    arms = [a for a in ["random", "lexicon", "tfidf_lr", "embed_lr", "minilm_ft"] if a in set(df.arm)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    w = 0.8 / 3
    colors = [ACCENT, WARN, "#0ea5e9"]
    for ax, metric, lab in zip(axes, ["accuracy", "valence_auc"], ["6-class accuracy", "valence ROC-AUC"]):
        for j, st in enumerate(df.setting.unique()):
            sub = df[df.setting == st].set_index("arm").reindex(arms)
            ax.bar(np.arange(len(arms)) + (j - 1) * w, sub[metric], w, label=st, color=colors[j])
        ax.set_xticks(np.arange(len(arms)), arms, rotation=20)
        ax.set_ylim(0, 1.0)
        ax.set_title(lab, loc="left")
        ax.grid(axis="y", color=LINE)
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    _save(fig, "classifier")


def fig_person(m, s, pid=None):
    ppl, daily, sents = s["people"], s["daily"], s["sents"]
    D, B, arm = s["cfg"]["days"], s["cfg"]["baseline_days"], s["shipped"]
    if pid is None:
        flags = np.asarray(s["flags"][arm])
        cand = ppl[(ppl.drift == 1) & (flags >= ppl.onset) & (ppl.onset < 110)]
        pid = int(cand.person_id.iloc[0]) if len(cand) else int(ppl[ppl.drift == 1].person_id.iloc[0])
    fig = person_figure(s, pid)
    _save(fig, "person")
    return pid


def person_figure(s, pid):
    from src import tracker as T
    ppl, daily = s["people"], s["daily"]
    D, B, arm = s["cfg"]["days"], s["cfg"]["baseline_days"], s["shipped"]
    row = ppl.set_index("person_id").loc[pid]
    d = daily[daily.person_id == pid]
    M = s["M_ev"][arm]
    x = M[pid]
    mu, sd, _, _ = T.personal_baseline(M, B, pop_sd=None)
    S = T.cusum(M, mu, sd, B, s["k"])[pid]
    flag = int(np.asarray(s["flags"][arm])[pid])
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.1, 1.1, 0.9]))
    for ax in axes:
        for dd in d[d.dip == 1].day:
            ax.axvspan(dd - 0.5, dd + 0.5, color="#fde68a", alpha=0.45, lw=0)
        if row.drift:
            ax.axvspan(row.onset, D, color="#ede9fe", alpha=0.6, lw=0)
        ax.axvline(B, color="#94a3b8", ls=":", lw=1)
        if flag >= 0:
            ax.axvline(flag, color=HIGH, lw=1.6)
    axes[0].plot(d.day, d.latent, color=INK, lw=1.2)
    axes[0].set_ylabel("true mood\n(hidden)")
    axes[1].scatter(np.arange(D), x, s=9, color=ACCENT)
    axes[1].axhline(mu[pid], color=MUTED, lw=1, ls="--")
    axes[1].set_ylabel("entry valence\n(from text)")
    axes[2].plot(np.arange(D), S, color=ACCENT, lw=1.4)
    axes[2].axhline(s["threshold"], color=HIGH, ls="--", lw=1)
    axes[2].set_ylabel("CUSUM")
    axes[2].set_xlabel("day")
    title = (f"Journaler {pid} — {'sustained decline from day ' + str(int(row.onset)) if row.drift else 'no decline'}"
             f" · {'flagged day ' + str(flag) if flag >= 0 else 'never flagged'}")
    axes[0].set_title(title, loc="left")
    return fig


def fig_rules(m):
    r = pd.DataFrame(m["rules"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    names = {"cusum": "personal CUSUM", "personal_rolling": "personal 14-day drop",
             "population_rolling": "population threshold"}
    lab = [names[x] for x in r.rule]
    axes[0].bar(lab, r.detection_rate * 100, color=[ACCENT, "#a78bfa", WARN])
    axes[0].set_title("declines caught (%)", loc="left")
    axes[1].bar(lab, r.stable_flags_from_lowest_mood_quartile * 100, color=[ACCENT, "#a78bfa", WARN])
    axes[1].axhline(25, color=MUTED, ls="--", lw=1)
    axes[1].set_title("stable-people flags from the gloomiest 25% (%)", loc="left")
    for ax in axes:
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", color=LINE)
    _save(fig, "rules")


def fig_silence(m):
    sw = pd.DataFrame(m["silence_sweep"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    lab = [s.split(" (")[0].replace("as shipped: ", "shipped: ") for s in sw.setting]
    axes[0].bar(lab, sw.written_share_after_full_decline * 100, color=ACCENT)
    axes[0].set_title("days written once fully declined (%)", loc="left")
    axes[1].bar(lab, sw.detection_rate * 100, color="#a78bfa")
    axes[1].set_title("declines caught (%)", loc="left")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18, labelsize=8.5)
        ax.grid(axis="y", color=LINE)
    _save(fig, "silence")


def fig_coupling(m):
    cs = pd.DataFrame(m["coupling_sweep"])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(cs.gold_within_person_r, cs.gold_detection_rate * 100, "o-", color=WARN, label="gold labels")
    ax.plot(cs.shipped_within_person_r, cs.shipped_detection_rate * 100, "o-", color=ACCENT,
            label="fine-tuned classifier")
    lat = [r for r in m["ladder"] if r["arm"] == "latent_written"]
    if lat:
        ax.axhline(lat[0]["detection_rate"] * 100, color=MUTED, ls="--", lw=1)
        ax.text(ax.get_xlim()[0], lat[0]["detection_rate"] * 100 + 1.5,
                " ceiling: true mood on written days", fontsize=8.5, color=MUTED)
    ax.set_xlabel("within-person correlation of entry score with true mood")
    ax.set_ylabel("declines caught (%)")
    ax.set_title("How much must mood leak into words?", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=LINE)
    _save(fig, "coupling")


def fig_budget(m):
    b = pd.DataFrame(m["budget_sweep"])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(b.stable_flag_rate * 100, b.detection_rate * 100, "o-", color=ACCENT)
    for r in b.itertuples():
        ax.annotate(f"budget {r.budget:.0%}", (r.stable_flag_rate * 100, r.detection_rate * 100),
                    textcoords="offset points", xytext=(6, -10), fontsize=8.5, color=MUTED)
    ax.set_xlabel("never-declining people flagged over 5 months (%)")
    ax.set_ylabel("declines caught (%)")
    ax.set_title("The trade the budget buys", loc="left")
    ax.grid(color=LINE)
    _save(fig, "budget")


def make_all():
    m, s = load()
    fig_ladder(m)
    fig_accuracy_vs_detection(m)
    fig_classifier(m)
    fig_person(m, s)
    fig_rules(m)
    fig_silence(m)
    fig_coupling(m)
    fig_budget(m)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    make_all()
