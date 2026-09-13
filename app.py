"""Journaling Mood & Burnout Tracker -- Streamlit app.

    streamlit run app.py --server.port 8516
"""

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
import streamlit as st

from src import data as DATA
from src import evaluate as EV
from src import lexicon as LEX
from src import tracker as T

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

st.set_page_config(page_title="Journaling Mood & Burnout Tracker", page_icon="📓", layout="wide")

DISCLAIMER = (
    "**Not a medical device.** This is a research demonstration of a *mood-trend screening aid*. "
    "It does not diagnose burnout, depression or any condition, and it must not be used to make "
    "decisions about a real person. All journalers are **synthetic**; the sentences are public "
    "social-media text, not diaries. If you are struggling, please talk to someone you trust or "
    "contact your local emergency services or a crisis line."
)


@st.cache_data
def load_metrics():
    p = ART / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_resource
def load_study():
    p = ART / "study.pkl.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_tfidf():
    p = ART / "tfidf_lr.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


m = load_metrics()
s = load_study()

st.title("📓 Journaling Mood & Burnout Tracker")
st.caption("AI-in-Healthcare daily build · Project #16 of 30 · real public sentences, synthetic lives")
st.warning(DISCLAIMER)

if m is None or s is None:
    st.error("No study artifacts found. Run `python build.py --figures` first (see README).")
    st.stop()

H = m["headline"]
CFG = m["config"]
SHIPPED = CFG["shipped_arm"]
pct = lambda v: "—" if v is None else f"{100 * v:.0f}%"
pct1 = lambda v: "—" if v is None else f"{100 * v:.1f}%"

tabs = st.tabs(["Overview", "Try it on your own entries", "Journaler explorer", "Where detection is lost",
                "Sentence classifiers", "Personal vs population", "Silence", "Register shift",
                "Operating point (live)", "Method & honest limits"])


def df_fmt(df, pct_cols=(), f3_cols=(), f1_cols=()):
    df = pd.DataFrame(df).copy()
    for c in pct_cols:
        if c in df:
            df[c] = df[c].map(lambda v: None if v is None or pd.isna(v) else f"{100 * v:.1f}%")
    for c in f3_cols:
        if c in df:
            df[c] = df[c].map(lambda v: None if v is None or pd.isna(v) else f"{v:.3f}")
    for c in f1_cols:
        if c in df:
            df[c] = df[c].map(lambda v: None if v is None or pd.isna(v) else f"{v:.1f}")
    return df


# ------------------------------------------------------------------ Overview
with tabs[0]:
    st.markdown(
        "A journaling companion reads daily entries, scores the emotion in each sentence with a "
        "transformer, and asks one question per person: **has this person's mood shifted below "
        "their own normal, and stayed there?** A flag means *a gentle human check-in*, never a "
        "diagnosis. The deliverable is that flag, so everything is scored **per journaler** — "
        "how many sustained declines are caught, how late, and how many people who never declined "
        "get flagged anyway — not per sentence.")
    c = st.columns(5)
    c[0].metric("Sustained declines caught", pct(H["detection_rate"]),
                help=f"95% CI {pct(H['detection_ci'][0])}–{pct(H['detection_ci'][1])}")
    c[1].metric("Never-declining people flagged", pct(H["stable_flag_rate"]),
                help=f"budget {pct(CFG['budget'])}, fitted on a separate calibration cohort")
    c[2].metric("Median days from onset to flag", f"{H['median_latency']:.0f}")
    c[3].metric(f"Flags that are real, at {pct(CFG['real_prevalence'])} prevalence",
                pct(H["ppv_at_real_prevalence"]))
    c[4].metric("Sentence accuracy (6 emotions)",
                pct1(m["classification"]["in_domain"][SHIPPED]["accuracy"]))
    lad = pd.DataFrame(m["ladder"])
    st.subheader("Where detection is lost")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(lad))[::-1]
    ax.barh(y, lad.detection_rate * 100,
            color=["#6d28d9" if a == SHIPPED else "#c4b5fd" for a in lad.arm])
    ax.set_yticks(y, lad.label)
    ax.set_xlabel("declines caught (%)")
    for yi, v in zip(y, lad.detection_rate):
        ax.text(v * 100 + 1, yi, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Same people, same budget, every rung. The top rung reads true mood every day; each "
               "rung below removes something a real journaling app does not have.")

# ------------------------------------------------------------------ Try it
with tabs[1]:
    st.markdown("Write one entry per line — each line is one day. Nothing is stored or sent anywhere; "
                "scoring happens inside this app. This is a demonstration of the *mechanism*, "
                "and on a handful of entries it means very little.")
    example = "\n".join([
        "i feel really good about how the week started",
        "had a lovely dinner with friends and felt so grateful",
        "work was busy but i felt calm and on top of it",
        "i feel content today nothing special",
        "slept well and felt energetic this morning",
        "i feel a bit tired but mostly fine",
        "i feel stressed about the deadline",
        "i feel exhausted and a little hopeless about work",
        "i feel drained and irritable all day",
        "i feel lonely and like nothing i do matters",
        "i feel so tired of everything",
        "i feel anxious before bed again",
    ])
    txt = st.text_area("Entries (one per day)", value=example, height=260)
    scorer = st.radio("Sentence scorer", ["TF-IDF + LR (fast)", "fine-tuned MiniLM (loads a transformer)",
                                          "keyword lexicon"], horizontal=True)
    entries = [e.strip() for e in txt.splitlines() if e.strip()]
    if len(entries) >= 6:
        if scorer.startswith("fine"):
            from src import classifiers as C
            if C.finetuned_available():
                P = C.predict_finetuned(entries, use_cache=False)
            else:
                st.info("Fine-tuned model not found in artifacts/ — using TF-IDF instead.")
                P = load_tfidf().predict_proba(entries)
        elif scorer.startswith("TF"):
            P = load_tfidf().predict_proba(entries)
        else:
            P = LEX.predict_proba(entries)
        v = DATA.valence(P)
        nb = max(3, len(entries) // 2)
        M = v[None, :]
        mu, sd, _, _ = T.personal_baseline(M, nb, pop_sd=float(np.std(v[:nb], ddof=1) or 0.5))
        S = T.cusum(M, mu, sd, nb, m["k_selection"]["chosen"])[0]
        tab = pd.DataFrame(P, columns=DATA.CLASSES)
        tab.insert(0, "entry", entries)
        tab["valence"] = v
        tab["CUSUM"] = S
        st.dataframe(df_fmt(tab, f3_cols=DATA.CLASSES + ["valence", "CUSUM"]), width="stretch")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].plot(range(1, len(v) + 1), v, "o-", color="#6d28d9")
        axes[0].axhline(mu[0], color="#94a3b8", ls="--")
        axes[0].axvspan(0.5, nb + 0.5, color="#e2e8f0", alpha=.6)
        axes[0].set_title("entry valence (grey = baseline entries)", loc="left", fontsize=10)
        axes[1].plot(range(1, len(v) + 1), S, color="#6d28d9")
        axes[1].set_title("CUSUM of drop below your own baseline", loc="left", fontsize=10)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"The study's threshold ({s['threshold']:.1f}) was fitted on ~5 months of synthetic "
                   "entries with a 5-week baseline; it is shown for orientation only and is not "
                   "meaningful on a dozen lines.")
    else:
        st.info("Enter at least 6 entries.")

# ------------------------------------------------------------------ explorer
with tabs[2]:
    from src.figures import person_figure
    ppl = s["people"].copy()
    flags = np.asarray(s["flags"][SHIPPED])
    ppl["flag_day"] = flags
    ppl["outcome"] = np.select(
        [(ppl.drift == 1) & EV.detected_mask(ppl, flags, s["cfg"]["days"]),
         (ppl.drift == 1), (ppl.drift == 0) & (flags >= 0)],
        ["decline · caught", "decline · missed", "no decline · flagged anyway"], "no decline · not flagged")
    choice = st.selectbox("Show", ["decline · caught", "decline · missed", "no decline · flagged anyway",
                                   "no decline · not flagged"])
    ids = ppl[ppl.outcome == choice].person_id.tolist()
    if ids:
        pid = st.selectbox("Journaler", ids[:80])
        fig = person_figure(s, int(pid))
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Top: hidden true mood (purple = sustained decline, yellow = bad weeks). Middle: what "
                   "the text says each day an entry exists. Bottom: the CUSUM and the fitted threshold; "
                   "red line = the flag.")
        r = ppl.set_index("person_id").loc[pid]
        st.write(f"Typical mood {r.mu:+.2f} · writes on {r.adherence:.0%} of days (before mood effects) · "
                 f"{int(r.n_entries)} entries · {int(r.n_dips)} bad week(s)")
        sents = s["sents"][s["sents"].person_id == pid]
        day = st.slider("Read entries around day", 0, int(s["cfg"]["days"]) - 1,
                        int(r.onset) if r.drift else 90)
        near = sents[(sents.day >= day - 3) & (sents.day <= day + 3)]
        if len(near):
            pr = s["pool_probs"][SHIPPED][near.pool_idx.to_numpy()]
            show = pd.DataFrame(dict(day=near.day, sentence=[s["pool_text"][i] for i in near.pool_idx],
                                     gold=[DATA.CLASSES[c] for c in near.true_class],
                                     predicted=[DATA.CLASSES[i] for i in pr.argmax(1)],
                                     valence=DATA.valence(pr)))
            st.dataframe(df_fmt(show, f3_cols=["valence"]), width="stretch", hide_index=True)
        else:
            st.info("No entries within 3 days of that day — this person was not writing.")
    else:
        st.info("Nobody in this group.")

# ------------------------------------------------------------------ ladder
with tabs[3]:
    lad = pd.DataFrame(m["ladder"])
    st.dataframe(df_fmt(lad[["label", "sentence_accuracy", "sentence_valence_auc", "detection_rate",
                             "median_latency", "stable_flag_rate", "early_flag_rate", "cal_stable_flag_rate"]],
                        pct_cols=["sentence_accuracy", "detection_rate", "stable_flag_rate",
                                  "early_flag_rate", "cal_stable_flag_rate"],
                        f3_cols=["sentence_valence_auc"], f1_cols=["median_latency"]),
                 width="stretch", hide_index=True)
    st.markdown(f"Within-person correlation of the daily entry score with true mood: **gold labels "
                f"{H['within_person_r_gold']:.2f}**, **shipped classifier {H['within_person_r_shipped']:.2f}**.")
    be = H.get("baseline_error")
    if be and be.get("mean_true_gap_flagged") is not None:
        st.markdown(f"**The baseline is an estimate too.** Among never-declining people, how much happier "
                    f"their hidden true mood was in the first {CFG['baseline_days']} days than afterwards "
                    f"predicts being flagged with AUC **{be['auc_true_mood_gap_for_stable_flag']:.2f}**; "
                    f"fewer baseline entries predicts it with AUC "
                    f"{be['auc_fewer_baseline_entries_for_stable_flag']:.2f} (median "
                    f"{be['median_baseline_entries']:.0f} baseline entries).")
    cs = pd.DataFrame(m["coupling_sweep"])
    st.subheader("How much must mood leak into word choice?")
    st.dataframe(df_fmt(cs, pct_cols=[c for c in cs if c.endswith("rate")],
                        f3_cols=[c for c in cs if c.endswith("_r")]), width="stretch", hide_index=True)
    st.subheader("Negative controls")
    nc = m["negative_controls"]
    st.write(f"Random scorer: {pct1(nc['random_scorer']['detection_rate'])} caught · "
             f"days shuffled after baseline: {pct1(nc['shuffled_days']['detection_rate'])} caught "
             f"(vs {pct1(H['detection_rate'])} shipped).")

# ------------------------------------------------------------------ classifiers
with tabs[4]:
    rows = []
    for setting, title in [("in_domain", "tweets"), ("keywords_masked", "keywords masked"),
                           ("reddit_transport", "Reddit")]:
        for arm, r in m["classification"][setting].items():
            rows.append(dict(test_set=title, arm=arm, n=r["n"], accuracy=r["accuracy"], macro_f1=r["macro_f1"],
                             valence_auc=r["valence_auc"], valence_sign_accuracy=r["valence_sign_acc"]))
    st.dataframe(df_fmt(pd.DataFrame(rows), pct_cols=["accuracy", "valence_sign_accuracy"],
                        f3_cols=["macro_f1", "valence_auc"]), width="stretch", hide_index=True)
    sc = m["shortcut"]
    st.markdown(f"**Keyword probe.** The 100 unigrams most associated with the label on the training split "
                f"were masked in the test sentences; {pct(sc['share_sentences_touched'])} of sentences lost "
                f"at least one word. The hand-written lexicon matches a word in "
                f"{pct(sc['lexicon_coverage_tweets'])} of tweets vs {pct(sc['lexicon_coverage_reddit'])} of "
                f"Reddit comments; the word *feel* appears in {pct(sc['sentences_containing_feel'])} vs "
                f"{pct(sc['reddit_containing_feel'])}.")
    st.caption("First masked tokens: " + ", ".join(sc["masked_tokens"][:30]))
    cm = np.array(m["classification"]["in_domain"][SHIPPED]["confusion"])
    st.subheader(f"Confusion matrix — {SHIPPED}, tweets")
    st.dataframe(pd.DataFrame(cm, index=[f"true {c}" for c in DATA.CLASSES],
                              columns=[f"pred {c}" for c in DATA.CLASSES]), width="stretch")

# ------------------------------------------------------------------ rules
with tabs[5]:
    r = pd.DataFrame(m["rules"])
    st.dataframe(df_fmt(r[["rule", "detection_rate", "median_latency", "stable_flag_rate",
                           "stable_flags_from_lowest_mood_quartile", "typical_mood_auc_for_stable_flag",
                           "detection_cheerful_half", "detection_gloomy_half", "stable_flags_after_bad_week",
                           "bad_week_base_rate"]],
                        pct_cols=["detection_rate", "stable_flag_rate", "stable_flags_from_lowest_mood_quartile",
                                  "detection_cheerful_half", "detection_gloomy_half",
                                  "stable_flags_after_bad_week", "bad_week_base_rate"],
                        f3_cols=["typical_mood_auc_for_stable_flag"], f1_cols=["median_latency"]),
                 width="stretch", hide_index=True)
    st.caption("`typical_mood_auc_for_stable_flag` = how well a never-declining person's typical mood "
               "predicts that they get flagged (0.5 = the rule does not care who you are).")

# ------------------------------------------------------------------ silence
with tabs[6]:
    si = m["silence"]
    st.markdown(f"Once a decline is complete, people in this cohort write on "
                f"**{pct(si['written_share_after_full_decline'])}** of days, against "
                f"**{pct(si['written_share_stable_monitoring'])}** for people who never declined. "
                f"Averaged over the days they *did* write, their mood looks "
                f"{si['mean_dev_written_days_after_full_decline']:+.2f} below normal; over all days it is "
                f"{si['mean_dev_all_days_after_full_decline']:+.2f}.")
    st.dataframe(df_fmt(m["silence_sweep"], pct_cols=["written_share_after_full_decline", "detection_rate",
                                                      "stable_flag_rate"],
                        f3_cols=["entry_count_auc_for_decline"], f1_cols=["median_latency"]),
                 width="stretch", hide_index=True)
    st.subheader("By how often people write")
    st.dataframe(df_fmt(m["by_adherence"], pct_cols=["detection_rate"], f1_cols=["median_latency"]),
                 width="stretch", hide_index=True)
    st.subheader("By depth and speed of the decline")
    st.dataframe(df_fmt(m["by_depth"], pct_cols=["detection_rate"], f1_cols=["median_latency"]),
                 width="stretch", hide_index=True)
    st.dataframe(df_fmt(m["by_ramp"], pct_cols=["detection_rate"], f1_cols=["median_latency"]),
                 width="stretch", hide_index=True)

# ------------------------------------------------------------------ transport
with tabs[7]:
    st.markdown("The **same synthetic people** — same mood, same days written, same emotion in every "
                "sentence — rewritten with **Reddit comments** (GoEmotions) instead of tweets. Thresholds "
                "stay as fitted on the tweet calibration cohort.")
    st.dataframe(df_fmt(m["transport"], pct_cols=["detection_tweets", "detection_reddit",
                                                  "stable_flag_tweets", "stable_flag_reddit"],
                        f3_cols=["mean_valence_tweets", "mean_valence_reddit"]),
                 width="stretch", hide_index=True)

# ------------------------------------------------------------------ live operating point
with tabs[8]:
    st.markdown("Pick a false-flag budget. The threshold is re-fitted on the **calibration** cohort and "
                "applied unchanged to the **evaluation** cohort, using the same functions as the study.")
    budget = st.select_slider("Budget: share of never-declining people flagged over 5 months",
                              options=[0.02, 0.05, 0.10, 0.20, 0.30], value=CFG["budget"],
                              format_func=lambda v: f"{v:.0%}")
    from build import detect
    from src.journal import JournalConfig
    cfg = JournalConfig(**{k: v for k, v in s["cfg"].items()})
    res, ff, _, h = detect(s["M_cal"][SHIPPED], s["people_cal"], s["M_ev"][SHIPPED], s["people"], cfg,
                           "cusum", s["k"], budget)
    c = st.columns(4)
    c[0].metric("declines caught", pct1(res["detection_rate"]))
    c[1].metric("never-declining flagged", pct1(res["stable_flag_rate"]))
    c[2].metric("median latency (days)", f"{res['median_latency']:.0f}")
    c[3].metric(f"real flags at {pct(CFG['real_prevalence'])} prevalence",
                pct1(EV.ppv_at_prevalence(res["detection_rate"], res["stable_flag_rate"], CFG["real_prevalence"])))
    st.dataframe(df_fmt(m["budget_sweep"][:], pct_cols=["budget", "detection_rate", "stable_flag_rate",
                                                         "early_flag_rate", "ppv_at_real_prevalence"],
                        f1_cols=["median_latency", "threshold"])[
        ["budget", "threshold", "detection_rate", "median_latency", "stable_flag_rate", "early_flag_rate",
         "ppv_at_real_prevalence"]], width="stretch", hide_index=True)
    tr = H["travel"]
    pr = m["pool_replicates"]
    st.markdown(f"**Does the budget travel?** Calibrate on cohort A → evaluate on B: "
                f"{pct1(tr['cal16_to_eval17']['stable_flag_rate'])} flagged; the other way round: "
                f"{pct1(tr['cal17_to_eval16']['stable_flag_rate'])}. Rewriting the same evaluation people "
                f"from disjoint halves of the sentence pool moves detection between "
                f"{pct1(pr['detection_min'])} and {pct1(pr['detection_max'])}.")

# ------------------------------------------------------------------ method
with tabs[9]:
    st.markdown(f"""
**Text.** dair-ai/emotion (tweets, six emotions, labels from hashtag distant supervision) trains four
sentence scorers: a fixed keyword lexicon, TF-IDF + logistic regression, frozen MiniLM embeddings +
logistic regression, and MiniLM fine-tuned end-to-end on CPU. Linear hyper-parameters come from
cross-validation on the training split only. Validation and test sentences — after dropping any that
also occur in training — become the pools diaries are written from. GoEmotions (Reddit) is used only as
a register-shift test.

**Lives.** {CFG['n_people']} synthetic journalers per cohort over {CFG['days']} days: a personal typical
mood (between-person sd {CFG['baseline_sd']}), day-to-day AR(1) wobble (φ = {CFG['ar_phi']}),
bad weeks (~{CFG['dip_rate']} per 6 months) and — for {pct(CFG['drift_frac'])} — a sustained decline
starting between day {CFG['onset_min']} and {CFG['onset_max']}, ramping over 2–6 weeks. A bad week is
**exactly as deep** as a decline; only duration separates them. Whether someone writes, and which
emotion each sentence carries, depends on mood alone — never on the label (a test enforces it).

**Tracker.** Daily score = mean sentence valence (P(joy)+P(love) − P(sadness)+P(anger)+P(fear)).
Personal baseline from the first {CFG['baseline_days']} days, then a one-sided CUSUM of the standardised
drop. Threshold = the quantile of per-person maxima that flags {pct(CFG['budget'])} of never-declining
people in a **separate calibration cohort**. The CUSUM allowance k was chosen on calibration too.

**Honest limits.**
- The sentences are tweets and Reddit comments, not diaries, and the lives are simulated. The
  text–mood coupling is an assumption; it is swept, not claimed.
- "Burnout" here is a sustained drop in expressed mood. Real burnout is a syndrome (exhaustion,
  cynicism, reduced efficacy) that no sentiment score measures.
- The in-study prevalence of decline ({pct(CFG['drift_frac'])}) is set for statistical power; precision
  is re-expressed at an assumed {pct(CFG['real_prevalence'])}.
- Sentences are reused across synthetic people, so person-level bootstrap intervals understate the
  uncertainty that comes from the classifier's errors; the pool-replicate spread is shown for that.
- A flag should only ever open a supportive, consented conversation. Nothing here has been validated
  with real users.
""")
