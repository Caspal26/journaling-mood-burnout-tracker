"""Run the whole study and write artifacts/metrics.json.

    python build.py                # full study (fine-tune reused if present)
    python build.py --quick        # 120 people per cohort, no fine-tuning, fast check
    python build.py --retrain-ft   # force the MiniLM fine-tune to re-run (~10 min CPU)
    python build.py --figures      # also render figures + docs/index.html

Everything the app and docs/index.html display comes out of this one run, so the
page cannot drift away from the numbers.
"""

from __future__ import annotations

import os

# 14 BLAS threads fighting over paged-out memory made one logistic regression take
# 10+ minutes on this 7.5 GB machine; a few threads are much faster in practice.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import gzip
import json
import pickle
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src import classifiers as C
from src import data as DATA
from src import evaluate as EV
from src import lexicon as LEX
from src import tracker as T
from src.journal import JournalConfig, build_cohort

warnings.filterwarnings("ignore")
ART = Path(__file__).parent / "artifacts"
ART.mkdir(exist_ok=True)

SHIPPED_ARM = "minilm_ft"      # declared before any detection result was seen
BUDGET = 0.10                  # at most 10% of never-declining people flagged over 145 days
REAL_PREVALENCE = 0.10         # assumed share developing a sustained decline in 6 months
ARM_LABELS = {
    "latent_all": "true mood, every day (no text, no gaps)",
    "latent_written": "true mood, only on days they wrote",
    "gold_text": "gold emotion labels of the sentences written",
    "minilm_ft": "fine-tuned MiniLM (shipped)",
    "embed_lr": "frozen MiniLM embeddings + LR",
    "tfidf_lr": "TF-IDF + LR",
    "lexicon": "keyword lexicon (no learning)",
    "random": "random scorer (negative control)",
}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return jsonable(o.replace({np.nan: None}).to_dict("records"))
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def matrices(cohort, probs_by_arm, cfg):
    people, daily, sents = cohort
    P, D = len(people), cfg.days
    out = {"latent_all": T.latent_matrix(daily, P, D),
           "latent_written": T.latent_matrix(daily, P, D, written_only=True)}
    for arm, probs in probs_by_arm.items():
        out[arm] = T.score_matrix(people, sents, probs, D)
    return out


def detect(M_cal, ppl_cal, M_ev, ppl_ev, cfg, rule="cusum", k=0.5, budget=BUDGET):
    start = cfg.baseline_days
    S_cal, b_cal = T.run(M_cal, cfg, rule, k)
    h = T.fit_threshold(S_cal, ppl_cal.drift.to_numpy() == 0, budget, start)
    S_ev, _ = T.run(M_ev, cfg, rule, k, pop_sd=b_cal["pop_sd"])
    ff = T.first_flag(S_ev, h, start)
    res = EV.score_detection(ppl_ev, ff, cfg.days, start)
    cal = EV.score_detection(ppl_cal, T.first_flag(S_cal, h, start), cfg.days, start)
    res.update(threshold=h, cal_detection_rate=cal["detection_rate"],
               cal_stable_flag_rate=cal["stable_flag_rate"])
    return res, ff, S_ev, h


def main(quick=False, retrain_ft=False):
    t0 = time.time()
    m: dict = {}
    n_people = 120 if quick else 300

    # ------------------------------------------------------------------ 1
    print("[1/11] corpora + leakage-safe sentence pools ...")
    em = DATA.load_emotion()
    train = em["train"]
    cal_pool, cal_dup = DATA.dedupe_against(em["validation"], train)
    ev_pool, ev_dup = DATA.dedupe_against(em["test"], train)
    go = DATA.load_goemotions()
    tr_pool, tr_dup = DATA.dedupe_against(go, train)
    m["corpora"] = dict(
        train_n=len(train), validation_n=len(em["validation"]), test_n=len(em["test"]),
        calibration_pool_n=len(cal_pool), calibration_dropped_as_train_duplicates=cal_dup,
        evaluation_pool_n=len(ev_pool), evaluation_dropped_as_train_duplicates=ev_dup,
        transport_pool_n=len(tr_pool), transport_dropped_as_train_duplicates=tr_dup,
        train_class_share={c: float(np.mean(train.label == i)) for i, c in enumerate(DATA.CLASSES)},
        evaluation_pool_class_n={c: int(np.sum(ev_pool.label == i)) for i, c in enumerate(DATA.CLASSES)},
        transport_pool_class_n={c: int(np.sum(tr_pool.label == i)) for i, c in enumerate(DATA.CLASSES)},
        transport_source_labels=go.source_label.value_counts().to_dict(),
    )
    print(f"       pools: calibration {len(cal_pool)}, evaluation {len(ev_pool)}, "
          f"transport {len(tr_pool)}; train-duplicates dropped {cal_dup}/{ev_dup}/{tr_dup}")

    # ------------------------------------------------------------------ 2
    print("[2/11] sentence classifiers ...")
    t2 = time.time()
    tfidf, tfidf_cv = C.fit_tfidf(train.text, train.label)
    print(f"       tfidf cv ({time.time() - t2:.0f}s): {tfidf_cv}")
    t2 = time.time()
    X_tr = C.embed(train.text)
    emb_lr, emb_cv = C.fit_embed_lr(X_tr, train.label.to_numpy())
    print(f"       embed cv ({time.time() - t2:.0f}s): {emb_cv}")
    mask_tok = C.discriminative_tokens(train.text, train.label, k=100)

    ft_info = None
    use_ft = not quick
    if use_ft:
        if retrain_ft or not C.finetuned_available():
            print("       fine-tuning MiniLM (2 epochs, CPU) ...")
            ft_info = C.finetune(train.text, train.label.to_numpy())
            (C.FT_DIR / "train_info.json").write_text(json.dumps(ft_info, indent=2))
        else:
            info_p = C.FT_DIR / "train_info.json"
            ft_info = json.loads(info_p.read_text()) if info_p.exists() else {}
            ft_info["reused"] = True
            print("       reusing fine-tuned MiniLM from artifacts/minilm_ft")

    rng = np.random.default_rng(1601)
    texts = dict(calibration=cal_pool.text.tolist(), evaluation=ev_pool.text.tolist(),
                 transport=tr_pool.text.tolist(),
                 evaluation_masked=C.mask_tokens(ev_pool.text, mask_tok))
    labels = dict(calibration=cal_pool.label.to_numpy(), evaluation=ev_pool.label.to_numpy(),
                  transport=tr_pool.label.to_numpy(), evaluation_masked=ev_pool.label.to_numpy())

    def predict(arm, tx):
        if arm == "lexicon":
            return LEX.predict_proba(tx)
        if arm == "tfidf_lr":
            return tfidf.predict_proba(tx)
        if arm == "embed_lr":
            return emb_lr.predict_proba(C.embed(tx))
        if arm == "minilm_ft":
            return C.predict_finetuned(tx)
        raise ValueError(arm)

    arms = ["lexicon", "tfidf_lr", "embed_lr"] + (["minilm_ft"] if use_ft else [])
    probs = {s: {a: predict(a, tx) for a in arms} for s, tx in texts.items()}
    for s in probs:
        probs[s]["gold_text"] = DATA.onehot(labels[s])
        probs[s]["random"] = rng.dirichlet(np.ones(len(DATA.CLASSES)), len(texts[s]))
    shipped = SHIPPED_ARM if use_ft else "tfidf_lr"

    # ------------------------------------------------------------------ 3
    print("[3/11] sentence-level metrics (in-domain / keywords masked / Reddit) ...")
    m["classifier_selection"] = dict(tfidf_cv=tfidf_cv, embed_cv=emb_cv, finetune=ft_info)
    m["classification"] = {}
    for s, title in [("evaluation", "in_domain"), ("evaluation_masked", "keywords_masked"),
                     ("transport", "reddit_transport")]:
        m["classification"][title] = {a: EV.classification(labels[s], probs[s][a])
                                      for a in arms + ["random"]}
    masked_share = float(np.mean([any(w == "___" for w in t.split())
                                  for t in texts["evaluation_masked"]]))
    m["shortcut"] = dict(masked_tokens=mask_tok[:30], n_masked_tokens=len(mask_tok),
                         share_sentences_touched=masked_share,
                         lexicon_coverage_tweets=LEX.coverage(texts["evaluation"]),
                         lexicon_coverage_reddit=LEX.coverage(texts["transport"]),
                         sentences_containing_feel=float(np.mean(
                             ["feel" in t.lower().split() for t in texts["evaluation"]])),
                         reddit_containing_feel=float(np.mean(
                             ["feel" in t.lower().split() for t in texts["transport"]])))

    # ------------------------------------------------------------------ 4
    print("[4/11] synthetic journalers (real sentences, synthetic lives) ...")
    cfg_cal = JournalConfig(n_people=n_people, seed=16)
    cfg_ev = JournalConfig(n_people=n_people, seed=17)
    coh_cal = build_cohort(cfg_cal, labels["calibration"])
    coh_ev = build_cohort(cfg_ev, labels["evaluation"])
    coh_tr = build_cohort(cfg_ev, labels["transport"])          # SAME people, Reddit sentences
    ppl_cal, ppl_ev = coh_cal[0], coh_ev[0]
    assert (coh_tr[1].wrote.to_numpy() == coh_ev[1].wrote.to_numpy()).all()
    assert (coh_tr[2].true_class.to_numpy() == coh_ev[2].true_class.to_numpy()).all()

    def cohort_stats(coh, cfg):
        p, d, s = coh
        base = s[s.day < cfg.baseline_days].true_class
        drift_days = d[(d.drifting == 1)]
        dd = s.merge(drift_days[["person_id", "day"]], on=["person_id", "day"])
        late = dd.merge(p[["person_id", "onset", "ramp"]], on="person_id")
        late = late[late.day >= late.onset + late.ramp]
        return dict(
            n_people=len(p), n_drift=int(p.drift.sum()), drift_share=float(p.drift.mean()),
            entries=int(d.wrote.sum()), sentences=int(len(s)),
            days_written_share=float(d.wrote.mean()),
            sentences_per_entry=float(len(s) / max(1, d.wrote.sum())),
            entries_per_person_median=float(p.n_entries.median()),
            bad_weeks_per_person=float(p.n_dips.mean()),
            negative_share_baseline=float(base.isin(DATA.NEG_IDX).mean()),
            negative_share_after_full_decline=float(late.true_class.isin(DATA.NEG_IDX).mean())
            if len(late) else None,
            written_share_stable=float(d[d.person_id.isin(p[p.drift == 0].person_id)].wrote.mean()),
        )

    m["config"] = dict(cfg_ev.as_dict(), shipped_arm=shipped, budget=BUDGET,
                       real_prevalence=REAL_PREVALENCE, grace_days=EV.GRACE_DAYS,
                       monitored_days=cfg_ev.days - cfg_ev.baseline_days, quick=quick)
    m["cohorts"] = dict(calibration=cohort_stats(coh_cal, cfg_cal),
                        evaluation=cohort_stats(coh_ev, cfg_ev))

    arms_det = ["latent_all", "latent_written", "gold_text"] + arms[::-1] + ["random"]
    Mc = matrices(coh_cal, probs["calibration"], cfg_cal)
    Me = matrices(coh_ev, probs["evaluation"], cfg_ev)
    Mt = matrices(coh_tr, probs["transport"], cfg_ev)

    # ------------------------------------------------------------------ 5
    print("[5/11] CUSUM allowance chosen on the calibration cohort ...")
    k_rows = []
    for k in (0.25, 0.5, 1.0):
        S_cal, _ = T.run(Mc[shipped], cfg_cal, "cusum", k)
        h = T.fit_threshold(S_cal, ppl_cal.drift.to_numpy() == 0, BUDGET, cfg_cal.baseline_days)
        r = EV.score_detection(ppl_cal, T.first_flag(S_cal, h, cfg_cal.baseline_days),
                               cfg_cal.days, cfg_cal.baseline_days)
        k_rows.append(dict(k=k, cal_detection_rate=r["detection_rate"],
                           cal_median_latency=r["median_latency"]))
    best_k = sorted(k_rows, key=lambda r: (-r["cal_detection_rate"], r["cal_median_latency"]))[0]["k"]
    m["k_selection"] = dict(rows=k_rows, chosen=best_k)
    K = best_k

    # ------------------------------------------------------------------ 6
    print("[6/11] where detection is lost: the ladder ...")
    ladder, flags = [], {}
    for arm in arms_det:
        res, ff, S, h = detect(Mc[arm], ppl_cal, Me[arm], ppl_ev, cfg_ev, "cusum", K)
        flags[arm] = ff
        acc = m["classification"]["in_domain"].get(arm, {})
        ladder.append(dict(arm=arm, label=ARM_LABELS[arm], sentence_accuracy=acc.get("accuracy"),
                           sentence_valence_auc=acc.get("valence_auc"), **res))
    m["ladder"] = pd.DataFrame(ladder)
    head, ff_ship, S_ship, h_ship = detect(Mc[shipped], ppl_cal, Me[shipped], ppl_ev, cfg_ev, "cusum", K)
    head["detection_ci"] = EV.bootstrap_ci(ppl_ev, ff_ship, cfg_ev.days, cfg_ev.baseline_days)
    head["stable_flag_ci"] = EV.bootstrap_ci(ppl_ev, ff_ship, cfg_ev.days, cfg_ev.baseline_days,
                                             key="stable_flag_rate")
    head["ppv_at_real_prevalence"] = EV.ppv_at_prevalence(
        head["detection_rate"], head["stable_flag_rate"], REAL_PREVALENCE)
    L_ev = T.latent_matrix(coh_ev[1], len(ppl_ev), cfg_ev.days)
    head["within_person_r_shipped"] = EV.within_person_r(Me[shipped], L_ev)
    head["within_person_r_gold"] = EV.within_person_r(Me["gold_text"], L_ev)
    # the budget is a quantile of per-person maxima: how well does it travel?
    swap, _, _, _ = detect(Me[shipped], ppl_ev, Mc[shipped], ppl_cal, cfg_cal, "cusum", K)
    head["travel"] = dict(
        cal16_to_eval17=dict(stable_flag_rate=head["stable_flag_rate"],
                             detection_rate=head["detection_rate"]),
        cal17_to_eval16=dict(stable_flag_rate=swap["stable_flag_rate"],
                             detection_rate=swap["detection_rate"]))
    m["headline"] = head

    # how much does mood have to leak into word choice before text is useful?
    csw = []
    for coup in (0.5, 1.0, 2.0, 4.0):
        c2c, c2e = replace(cfg_cal, text_coupling=coup), replace(cfg_ev, text_coupling=coup)
        k2c, k2e = build_cohort(c2c, labels["calibration"]), build_cohort(c2e, labels["evaluation"])
        assert np.allclose(k2e[1].latent.to_numpy(), coh_ev[1].latent.to_numpy())
        row = dict(text_coupling=coup)
        for arm in ("gold_text", shipped):
            M2c = T.score_matrix(k2c[0], k2c[2], probs["calibration"][arm], c2c.days)
            M2e = T.score_matrix(k2e[0], k2e[2], probs["evaluation"][arm], c2e.days)
            r2, _, _, _ = detect(M2c, k2c[0], M2e, k2e[0], c2e, "cusum", K)
            tag = "gold" if arm == "gold_text" else "shipped"
            row[f"{tag}_within_person_r"] = EV.within_person_r(M2e, L_ev)
            row[f"{tag}_detection_rate"] = r2["detection_rate"]
            row[f"{tag}_stable_flag_rate"] = r2["stable_flag_rate"]
        csw.append(row)
    m["coupling_sweep"] = pd.DataFrame(csw)

    # ------------------------------------------------------------------ 7
    print("[7/11] personal baseline vs population threshold ...")
    rules = []
    mu_true = ppl_ev.mu.to_numpy()
    stable = ppl_ev.drift.to_numpy() == 0
    q1 = mu_true <= np.quantile(mu_true[stable], 0.25)
    for rule in ("cusum", "personal_rolling", "population_rolling"):
        res, ff, S, h = detect(Mc[shipped], ppl_cal, Me[shipped], ppl_ev, cfg_ev, rule, K)
        fl = ff >= cfg_ev.baseline_days
        stable_fl = fl & stable
        drift = ~stable
        hit = EV.detected_mask(ppl_ev, ff, cfg_ev.days)
        bw = EV.what_stable_flags_hit(ppl_ev, coh_ev[1], ff, cfg_ev.days)
        rules.append(dict(
            rule=rule, **res,
            stable_flags_from_lowest_mood_quartile=float((stable_fl & q1).sum() / max(1, stable_fl.sum())),
            typical_mood_auc_for_stable_flag=EV.roc_auc(stable_fl[stable], -mu_true[stable]),
            detection_cheerful_half=float(hit[drift & (mu_true > np.median(mu_true))].mean()),
            detection_gloomy_half=float(hit[drift & (mu_true <= np.median(mu_true))].mean()),
            stable_flags_after_bad_week=bw["share_after_bad_week"],
            bad_week_base_rate=bw["base_rate_near_bad_week"]))
        if rule == "cusum":
            m["bad_weeks"] = bw
    m["rules"] = pd.DataFrame(rules)

    # ------------------------------------------------------------------ 8
    print("[8/11] silence: who writes less, and what it costs ...")
    d_ev = coh_ev[1].merge(ppl_ev[["person_id", "onset", "ramp", "drift"]], on="person_id")
    full = d_ev[(d_ev.drift == 1) & (d_ev.day >= d_ev.onset + d_ev.ramp)]
    m["silence"] = dict(
        mean_dev_all_days_after_full_decline=float(full.dev.mean()),
        mean_dev_written_days_after_full_decline=float(full[full.wrote == 1].dev.mean()),
        written_share_after_full_decline=float(full.wrote.mean()),
        written_share_stable_monitoring=float(
            d_ev[(d_ev.drift == 0) & (d_ev.day >= cfg_ev.baseline_days)].wrote.mean()),
    )
    m["by_adherence"] = EV.by_group(ppl_ev, ff_ship, cfg_ev.days, cfg_ev.baseline_days,
                                    "adherence", [0, 0.45, 0.65, 1.0],
                                    ["writes rarely (<45%)", "sometimes (45-65%)", "most days (>65%)"])
    m["by_depth"] = EV.by_group(ppl_ev, ff_ship, cfg_ev.days, cfg_ev.baseline_days,
                                "depth", [0.6, 0.8, 1.0, 1.2, 1.4],
                                ["0.6-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.4"])
    m["by_ramp"] = EV.by_group(ppl_ev, ff_ship, cfg_ev.days, cfg_ev.baseline_days,
                               "ramp", [14, 23, 32, 42], ["2-3 weeks", "3-4.5 weeks", "4.5-6 weeks"])

    sweeps = []
    for label, kw in [("as shipped: write less when low (0.8)", {}),
                      ("writing independent of mood (0.0)", dict(mood_dropout=0.0)),
                      ("strongly withdraw when low (1.6)", dict(mood_dropout=1.6)),
                      ("write MORE when low (-0.8)", dict(mood_dropout=-0.8))]:
        c2c, c2e = replace(cfg_cal, **kw), replace(cfg_ev, **kw)
        k2c, k2e = build_cohort(c2c, labels["calibration"]), build_cohort(c2e, labels["evaluation"])
        assert np.allclose(k2e[1].latent.to_numpy(), coh_ev[1].latent.to_numpy()), "sweep changed mood"
        M2c = T.score_matrix(k2c[0], k2c[2], probs["calibration"][shipped], c2c.days)
        M2e = T.score_matrix(k2e[0], k2e[2], probs["evaluation"][shipped], c2e.days)
        r2, ff2, _, _ = detect(M2c, k2c[0], M2e, k2e[0], c2e, "cusum", K)
        dd = k2e[1].merge(k2e[0][["person_id", "onset", "ramp", "drift"]], on="person_id")
        fl2 = dd[(dd.drift == 1) & (dd.day >= dd.onset + dd.ramp)]
        entries_auc = EV.roc_auc(k2e[0].drift, -k2e[0].n_entries)
        sweeps.append(dict(setting=label, mood_dropout=c2e.mood_dropout,
                           written_share_after_full_decline=float(fl2.wrote.mean()),
                           detection_rate=r2["detection_rate"], median_latency=r2["median_latency"],
                           stable_flag_rate=r2["stable_flag_rate"],
                           entry_count_auc_for_decline=entries_auc))
    m["silence_sweep"] = pd.DataFrame(sweeps)

    # ------------------------------------------------------------------ 9
    print("[9/11] register shift: same people, Reddit sentences ...")
    trows = []
    for arm in arms + ["gold_text"]:
        for rule in ("cusum", "population_rolling"):
            r_in, _, _, _ = detect(Mc[arm], ppl_cal, Me[arm], ppl_ev, cfg_ev, rule, K)
            r_tr, _, _, _ = detect(Mc[arm], ppl_cal, Mt[arm], coh_tr[0], cfg_ev, rule, K)
            trows.append(dict(arm=arm, rule=rule,
                              detection_tweets=r_in["detection_rate"],
                              detection_reddit=r_tr["detection_rate"],
                              stable_flag_tweets=r_in["stable_flag_rate"],
                              stable_flag_reddit=r_tr["stable_flag_rate"],
                              mean_valence_tweets=float(np.nanmean(Me[arm][:, :cfg_ev.baseline_days])),
                              mean_valence_reddit=float(np.nanmean(Mt[arm][:, :cfg_ev.baseline_days]))))
    m["transport"] = pd.DataFrame(trows)

    # ------------------------------------------------------------------ 10
    print("[10/11] budget sweep, pool replicates, negative controls ...")
    m["budget_sweep"] = pd.DataFrame([
        dict(budget=b, **{k: v for k, v in detect(Mc[shipped], ppl_cal, Me[shipped], ppl_ev,
                                                  cfg_ev, "cusum", K, b)[0].items()},
             ppv_at_real_prevalence=None) for b in (0.02, 0.05, 0.10, 0.20, 0.30)])
    m["budget_sweep"]["ppv_at_real_prevalence"] = [
        EV.ppv_at_prevalence(r.detection_rate, r.stable_flag_rate, REAL_PREVALENCE)
        for r in m["budget_sweep"].itertuples()]

    # the same people written from two disjoint halves of the sentence pool
    reps = []
    for rep in range(4):
        r_ = np.random.default_rng(900 + rep)
        half = r_.random(len(ev_pool)) < 0.5
        for side, sel in (("A", half), ("B", ~half)):
            idx = np.flatnonzero(sel)
            coh_h = build_cohort(cfg_ev, labels["evaluation"][idx])
            s_h = coh_h[2].copy()
            s_h["pool_idx"] = idx[s_h.pool_idx.to_numpy()]
            M_h = T.score_matrix(coh_h[0], s_h, probs["evaluation"][shipped], cfg_ev.days)
            r_h, _, _, _ = detect(Mc[shipped], ppl_cal, M_h, coh_h[0], cfg_ev, "cusum", K)
            reps.append(dict(replicate=rep, half=side, detection_rate=r_h["detection_rate"],
                             stable_flag_rate=r_h["stable_flag_rate"],
                             median_latency=r_h["median_latency"]))
    reps = pd.DataFrame(reps)
    m["pool_replicates"] = dict(rows=reps,
                                detection_min=float(reps.detection_rate.min()),
                                detection_max=float(reps.detection_rate.max()),
                                stable_flag_min=float(reps.stable_flag_rate.min()),
                                stable_flag_max=float(reps.stable_flag_rate.max()))

    # shuffling the monitored days destroys the sustained shape but keeps every entry
    rs = np.random.default_rng(5)
    Msh = Me[shipped].copy()
    B = cfg_ev.baseline_days
    for i in range(len(Msh)):
        Msh[i, B:] = Msh[i, B:][rs.permutation(cfg_ev.days - B)]
    r_sh, _, _, _ = detect(Mc[shipped], ppl_cal, Msh, ppl_ev, cfg_ev, "cusum", K)
    m["negative_controls"] = dict(
        shuffled_days=r_sh,
        random_scorer={k: v for k, v in m["ladder"].set_index("arm").loc["random"].items()})

    # ------------------------------------------------------------------ 11
    print("[11/11] writing artifacts ...")
    with open(ART / "tfidf_lr.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(ART / "embed_lr.pkl", "wb") as f:
        pickle.dump(emb_lr, f)
    study = dict(
        cfg=cfg_ev.as_dict(), k=K, shipped=shipped, threshold=h_ship,
        people=ppl_ev, daily=coh_ev[1], sents=coh_ev[2],
        people_cal=ppl_cal, daily_cal=coh_cal[1], sents_cal=coh_cal[2],
        pool_text=ev_pool.text.tolist(), pool_label=labels["evaluation"],
        pool_probs={a: probs["evaluation"][a] for a in (shipped, "tfidf_lr", "lexicon")},
        cal_pool_probs={shipped: probs["calibration"][shipped]},
        M_ev={a: Me[a] for a in (shipped, "latent_all", "gold_text")},
        M_cal={shipped: Mc[shipped]},
        flags={a: flags[a] for a in flags},
    )
    with gzip.open(ART / "study.pkl.gz", "wb", compresslevel=6) as f:
        pickle.dump(study, f)
    m["runtime_seconds"] = round(time.time() - t0, 1)
    m["generated"] = time.strftime("%Y-%m-%d %H:%M")
    (ART / "metrics.json").write_text(json.dumps(jsonable(m), indent=2), encoding="utf-8")
    print(f"done in {m['runtime_seconds']}s -> artifacts/metrics.json")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain-ft", action="store_true")
    ap.add_argument("--figures", action="store_true", help="also render docs figures + page")
    a = ap.parse_args()
    main(quick=a.quick, retrain_ft=a.retrain_ft)
    if a.figures:
        from src import figures
        figures.make_all()
        from src import docs
        docs.build()
