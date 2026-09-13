"""Generate docs/index.html from artifacts/metrics.json.

Written from the metrics file, never by hand, with figures inlined as base64:
self-contained, and unable to drift from the numbers the study produced. Every
comparison's *direction* is computed here, not asserted in prose (project #12).
"""

from __future__ import annotations

import base64
import html as H
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG = ART / "figures"
OUT = ROOT / "docs" / "index.html"

CSS = """
:root{--accent:#6d28d9;--accent-weak:#f3effd;--ink:#0f172a;--muted:#5b6b82;
--line:#e6ebf2;--bg:#fff;--panel:#f7f9fc}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:var(--bg);line-height:1.62}
.wrap{max-width:900px;margin:0 auto;padding:0 24px}
header{background:linear-gradient(135deg,#0f172a 0%,#3b0764 58%,#6d28d9 100%);color:#fff;padding:56px 0 48px}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:700;opacity:.85}
h1{font-size:34px;line-height:1.15;margin:10px 0 8px}
.sub{font-size:17px;opacity:.93;max-width:680px}
.badges{margin-top:22px;display:flex;flex-wrap:wrap;gap:8px}
.badge{font-size:12px;font-weight:600;padding:5px 11px;border-radius:999px;
background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.22)}
section{padding:34px 0;border-bottom:1px solid var(--line)}
h2{font-size:22px;margin:0 0 14px;display:flex;align-items:center;gap:10px}
h2 .dot{width:9px;height:9px;border-radius:50%;background:var(--accent)}
h3{font-size:16.5px;margin:22px 0 8px}
p{margin:0 0 14px;color:#26364d}
.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin:6px 0 18px}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.metric .n{font-size:26px;font-weight:750;color:var(--accent)}
.metric .l{font-size:13px;color:var(--muted);margin-top:2px}
.tablewrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;margin:8px 0 18px;font-size:14.5px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--panel);font-size:12.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
code,pre{font-family:"SF Mono",Menlo,Consolas,monospace}
pre{background:#0f172a;color:#e2e8f0;padding:16px 18px;border-radius:12px;overflow-x:auto;font-size:13.5px;line-height:1.55}
ul{margin:0 0 14px;padding-left:20px}li{margin:7px 0;color:#26364d}
.callout{background:var(--accent-weak);border:1px solid #ddd6fe;border-radius:12px;padding:16px 18px;color:#3b0764;margin:0 0 16px}
.disclaimer{background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:16px 18px;color:#7c2d12;margin:0 0 16px}
.refuted{background:#fef2f2;border:1px solid #fecaca;border-radius:12px;padding:16px 18px;color:#7f1d1d;margin:0 0 16px}
figure{margin:16px 0}
figure img{width:100%;border:1px solid var(--line);border-radius:12px}
figcaption{font-size:13px;color:var(--muted);margin-top:7px}
footer{padding:28px 0 52px;color:var(--muted);font-size:14px}
a{color:var(--accent)}
"""


def img(name: str, caption: str) -> str:
    p = FIG / f"{name}.png"
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img alt="{H.escape(caption)}" src="data:image/png;base64,{b64}">'
            f'<figcaption>{caption}</figcaption></figure>')


def table(df: pd.DataFrame, fmt: dict | None = None, num_cols=()) -> str:
    fmt = fmt or {}
    head = "".join(f'<th class="{"num" if c in num_cols else ""}">{H.escape(str(c))}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                v = "—"
            elif c in fmt:
                v = fmt[c](v)
            tds.append(f'<td class="{"num" if c in num_cols else ""}">{v}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return ('<div class="tablewrap"><table><thead><tr>' + head + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


pct = lambda v: "—" if v is None else f"{100 * v:.0f}%"
pct1 = lambda v: "—" if v is None else f"{100 * v:.1f}%"
f3 = lambda v: f"{v:.3f}"
f2 = lambda v: f"{v:.2f}"
f0 = lambda v: f"{v:.0f}"


def build() -> Path:
    m = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    cfg, h, corp = m["config"], m["headline"], m["corpora"]
    ship = cfg["shipped_arm"]
    lad = pd.DataFrame(m["ladder"]).set_index("arm")
    cls = m["classification"]
    coh = m["cohorts"]["evaluation"]

    det = lambda a: float(lad.loc[a, "detection_rate"])
    loss_gaps = det("latent_all") - det("latent_written")
    loss_text = det("latent_written") - det("gold_text")
    loss_clf = det("gold_text") - det(ship)
    total = det("latent_all") - det(ship)
    biggest = max([("missing days", loss_gaps), ("text as a sample of mood", loss_text),
                   ("the classifier", loss_clf)], key=lambda x: x[1])

    trained = [a for a in ("lexicon", "tfidf_lr", "embed_lr", ship) if a in lad.index]
    acc = {a: cls["in_domain"][a]["accuracy"] for a in trained}
    dets = {a: det(a) for a in trained}
    acc_span = max(acc.values()) - min(acc.values())
    det_span = max(dets.values()) - min(dets.values())
    best_acc = max(acc, key=acc.get)
    best_det = max(dets, key=dets.get)

    # classifier table
    crow = []
    for a in trained + ["random"]:
        crow.append(dict(arm=a,
                         tweets=cls["in_domain"][a]["accuracy"],
                         masked=cls["keywords_masked"][a]["accuracy"],
                         reddit=cls["reddit_transport"][a]["accuracy"],
                         val_tweets=cls["in_domain"][a]["valence_auc"],
                         val_reddit=cls["reddit_transport"][a]["valence_auc"]))
    ctab = pd.DataFrame(crow)
    mask_drop = {a: cls["in_domain"][a]["accuracy"] - cls["keywords_masked"][a]["accuracy"] for a in trained}
    red_drop = {a: cls["in_domain"][a]["accuracy"] - cls["reddit_transport"][a]["accuracy"] for a in trained}
    ctab.columns = ["scorer", "accuracy · tweets", "accuracy · keywords masked", "accuracy · Reddit",
                    "valence AUC · tweets", "valence AUC · Reddit"]

    ltab = pd.DataFrame(m["ladder"])[["label", "sentence_accuracy", "detection_rate", "median_latency",
                                      "stable_flag_rate", "early_flag_rate"]]
    ltab.columns = ["daily signal", "sentence accuracy", "declines caught", "median days to flag",
                    "never-declining flagged", "flagged before onset"]

    rules = pd.DataFrame(m["rules"]).set_index("rule")
    rtab = pd.DataFrame(m["rules"])[["rule", "detection_rate", "stable_flag_rate",
                                     "stable_flags_from_lowest_mood_quartile", "typical_mood_auc_for_stable_flag",
                                     "detection_cheerful_half", "detection_gloomy_half"]]
    rtab["rule"] = rtab.rule.map({"cusum": "personal CUSUM (shipped)", "personal_rolling": "personal 14-day drop",
                                  "population_rolling": "population threshold (14-day mean)"})
    rtab.columns = ["rule", "declines caught", "never-declining flagged", "flags from gloomiest 25%",
                    "typical mood → flag AUC", "caught · cheerful half", "caught · gloomy half"]
    pop, cus = rules.loc["population_rolling"], rules.loc["cusum"]

    sw = pd.DataFrame(m["silence_sweep"])
    swtab = sw[["setting", "written_share_after_full_decline", "detection_rate", "median_latency",
                "stable_flag_rate", "entry_count_auc_for_decline"]].copy()
    swtab.columns = ["writing behaviour", "days written once declined", "declines caught",
                     "median days to flag", "never-declining flagged", "entry count → decline AUC"]
    si = m["silence"]
    adh = pd.DataFrame(m["by_adherence"])
    adh.columns = ["how often they write", "n", "declines caught", "median days to flag"]
    dep = pd.DataFrame(m["by_depth"])
    dep.columns = ["decline depth (latent units)", "n", "declines caught", "median days to flag"]

    tr = pd.DataFrame(m["transport"])
    trs = tr[tr.arm == ship].set_index("rule")
    ttab = tr[["arm", "rule", "detection_tweets", "detection_reddit", "stable_flag_tweets",
               "stable_flag_reddit", "mean_valence_tweets", "mean_valence_reddit"]].copy()
    ttab.columns = ["scorer", "rule", "caught · tweets", "caught · Reddit", "false flags · tweets",
                    "false flags · Reddit", "mean valence · tweets", "mean valence · Reddit"]
    shift_c = trs.loc["cusum", "stable_flag_reddit"] - trs.loc["cusum", "stable_flag_tweets"]
    shift_p = trs.loc["population_rolling", "stable_flag_reddit"] - trs.loc["population_rolling", "stable_flag_tweets"]

    cs = pd.DataFrame(m["coupling_sweep"])
    cstab = cs[["text_coupling", "gold_within_person_r", "gold_detection_rate",
                "shipped_within_person_r", "shipped_detection_rate"]].copy()
    cstab.columns = ["mood → word coupling", "r · gold labels", "caught · gold labels",
                     "r · classifier", "caught · classifier"]

    bs = pd.DataFrame(m["budget_sweep"])[["budget", "threshold", "detection_rate", "median_latency",
                                          "stable_flag_rate", "ppv_at_real_prevalence"]]
    bs.columns = ["budget", "threshold", "declines caught", "median days to flag",
                  "never-declining flagged", f"real flags at {pct(cfg['real_prevalence'])} prevalence"]

    trv, pr, nc, bw = h["travel"], m["pool_replicates"], m["negative_controls"], m.get("bad_weeks", {})
    kk = m["k_selection"]

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Journaling Mood &amp; Burnout Tracker — Project #16</title>
<style>{CSS}</style></head><body>
<header><div class="wrap">
  <div class="eyebrow">AI-in-Healthcare · Daily Build · Project #16 of 30</div>
  <h1>📓 Journaling Mood &amp; Burnout Tracker</h1>
  <div class="sub">A journaling companion scores the emotion in every sentence with a fine-tuned
  transformer and asks, per person: <strong>has their mood dropped below their own normal and stayed
  there?</strong> The deliverable is a gentle check-in flag, so the evaluation is per journaler —
  not per sentence — and it asks how much sentence accuracy actually buys.</div>
  <div class="badges">
    <span class="badge">Web · Streamlit</span>
    <span class="badge">Fine-tuned MiniLM · TF-IDF · lexicon</span>
    <span class="badge">Real public sentences, synthetic lives</span>
    <span class="badge">Personal-baseline CUSUM</span>
    <span class="badge">Screening aid, not a diagnosis</span>
  </div>
</div></header>
<main class="wrap">

<section>
  <div class="disclaimer"><strong>Not a medical device.</strong> A research demonstration of a
  mood-trend screening aid. It does not diagnose burnout, depression or any condition and must not be
  used to make decisions about a real person. Every journaler is <strong>synthetic</strong>; the
  sentences are public social-media text, not diaries. No patient, therapy or crisis-line data is used.</div>
  <h2><span class="dot"></span>Headline</h2>
  <div class="grid">
    <div class="metric"><div class="n">{pct(h['detection_rate'])}</div><div class="l">sustained declines caught
      (95% CI {pct(h['detection_ci'][0])}–{pct(h['detection_ci'][1])})</div></div>
    <div class="metric"><div class="n">{pct(h['stable_flag_rate'])}</div><div class="l">of never-declining people
      flagged over {cfg['monitored_days']} days (budget {pct(cfg['budget'])})</div></div>
    <div class="metric"><div class="n">{f0(h['median_latency'])} d</div><div class="l">median onset → flag
      (IQR {f0(h['latency_p25'])}–{f0(h['latency_p75'])})</div></div>
    <div class="metric"><div class="n">{pct(h['ppv_at_real_prevalence'])}</div><div class="l">of flags real at an
      assumed {pct(cfg['real_prevalence'])} prevalence</div></div>
    <div class="metric"><div class="n">{pct1(cls['in_domain'][ship]['accuracy'])}</div><div class="l">sentence
      accuracy, 6 emotions (tweets)</div></div>
  </div>
  <div class="callout">Perfect knowledge of mood every day would catch <strong>{pct(det('latent_all'))}</strong>
  of declines at this budget. The shipped tracker catches <strong>{pct(det(ship))}</strong>. Of the
  {f0(100 * total)}-point gap, the largest share is <strong>{biggest[0]}</strong>
  ({f0(100 * biggest[1])} points); the classifier itself accounts for {f0(100 * loss_clf)}.</div>
</section>

<section>
  <h2><span class="dot"></span>The problem</h2>
  <p>Journaling apps increasingly offer "mood insights". The tempting build is a sentiment classifier and
  a chart. The useful question is different: a single bad week is normal and nobody should be flagged
  for one, but a slow slide that does not lift is worth a gentle, consented check-in. That makes the
  unit of evaluation <strong>the person over months</strong> — how many declines are caught, how late,
  and how many people who were fine get flagged anyway — and it makes sentence accuracy an input whose
  value has to be measured rather than assumed.</p>
</section>

<section>
  <h2><span class="dot"></span>Data: real sentences, synthetic lives</h2>
  <ul>
    <li><strong>dair-ai/emotion</strong> (Saravia et al., 2018): English tweets labelled with six emotions by
    hashtag distant supervision — {corp['train_n']:,} train / {corp['validation_n']:,} validation /
    {corp['test_n']:,} test. Research/educational use.</li>
    <li><strong>GoEmotions</strong> (Demszky et al., 2020, Apache-2.0): Reddit comments rated for 27 emotions;
    {corp['transport_pool_n']:,} single-label comments mapped onto the six classes via the paper's Ekman
    grouping. Used only as a register-shift test.</li>
    <li>Validation and test sentences become the <em>pools</em> diaries are written from, after dropping any
    whose text also appears in training ({corp['calibration_dropped_as_train_duplicates']} and
    {corp['evaluation_dropped_as_train_duplicates']} duplicates removed).</li>
    <li><strong>{cfg['n_people']} synthetic journalers per cohort × {cfg['days']} days</strong>, three cohorts
    (calibration, evaluation, and the evaluation people rewritten in Reddit sentences). Each has a
    personal typical mood, day-to-day wobble (AR(1), φ = {cfg['ar_phi']}), bad weeks, and for
    {pct(cfg['drift_frac'])} a sustained decline starting between day {cfg['onset_min']} and
    {cfg['onset_max']}. <strong>A bad week is exactly as deep as a decline</strong>; only its duration differs.</li>
    <li>People write on {pct(coh['days_written_share'])} of days, ~{coh['sentences_per_entry']:.1f} sentences
    per entry. Whether they write and which emotion each sentence carries depends on mood alone — the
    generator never sees the label, and a test requires byte-identical diaries when the label is swapped.</li>
  </ul>
</section>

<section>
  <h2><span class="dot"></span>Approach</h2>
  <p><strong>Sentence scorers</strong> (trained on tweets only): a fixed keyword lexicon; TF-IDF + logistic
  regression; frozen all-MiniLM-L6-v2 embeddings + logistic regression; and MiniLM fine-tuned end-to-end on
  CPU (the shipped scorer, declared before any detection result). <strong>Daily score</strong> = mean
  sentence valence, P(joy)+P(love) − P(sadness)−P(anger)−P(fear). <strong>Tracker</strong>: a personal
  baseline from the first {cfg['baseline_days']} days, then a one-sided CUSUM of the standardised drop
  (allowance k = {kk['chosen']}, chosen on the calibration cohort). <strong>Threshold</strong>: the
  quantile of per-person maxima that flags {pct(cfg['budget'])} of never-declining people in the
  <em>calibration</em> cohort, applied unchanged to evaluation. A flag counts as a detection only between
  onset and six weeks after the decline completes.</p>
</section>

<section>
  <h2><span class="dot"></span>1 · Where detection is lost</h2>
  {img('ladder', 'Same people, same budget; each rung removes one thing a real app does not have.')}
  {table(ltab, {"sentence accuracy": pct1, "declines caught": pct, "median days to flag": f0,
                "never-declining flagged": pct, "flagged before onset": pct},
         num_cols=ltab.columns[1:])}
  <p>From true mood every day ({pct(det('latent_all'))}) to true mood only on days people wrote
  ({pct(det('latent_written'))}) costs <strong>{f0(100 * loss_gaps)} points</strong>. Reading that mood through
  the emotions of the sentences written — with <em>perfect</em> labels — costs another
  <strong>{f0(100 * loss_text)}</strong> ({pct(det('gold_text'))}). Replacing gold labels with the fine-tuned
  classifier costs <strong>{f0(100 * loss_clf)}</strong> ({pct(det(ship))}). Within-person correlation of
  the daily score with true mood: {f2(h['within_person_r_gold'])} with gold labels,
  {f2(h['within_person_r_shipped'])} with the classifier.</p>
</section>

<section>
  <h2><span class="dot"></span>2 · Sentence accuracy is not the bottleneck</h2>
  {img('accuracy_vs_detection', 'Each trained scorer: sentence accuracy against declines caught.')}
  <p>Across the four scorers, sentence accuracy spans <strong>{f0(100 * acc_span)} points</strong>
  (best: {best_acc}, {pct1(acc[best_acc])}); detection spans <strong>{f0(100 * det_span)} points</strong>
  (best: {best_det}, {pct(dets[best_det])}). Averaging several sentences over several weeks forgives a
  great deal of per-sentence error, because the tracker only needs the <em>sign</em> of the mood mass, not
  the emotion.</p>
  {img('classifier', 'Sentence-level accuracy and valence AUC, in-domain, with keywords masked, and on Reddit.')}
  {table(ctab, {c: pct1 for c in ctab.columns[1:4]} | {c: f3 for c in ctab.columns[4:]}, num_cols=ctab.columns[1:])}
  <p><strong>The keyword shortcut.</strong> dair-ai labels come from emotion hashtags, and
  {pct(m['shortcut']['sentences_containing_feel'])} of test tweets contain the word <em>feel</em>. Masking the
  100 training unigrams most associated with the label (touching {pct(m['shortcut']['share_sentences_touched'])}
  of test sentences) drops accuracy by {", ".join(f"{a} {f0(100 * v)} pts" for a, v in mask_drop.items())}.
  On Reddit comments the drop is {", ".join(f"{a} {f0(100 * v)} pts" for a, v in red_drop.items())}.</p>
</section>

<section>
  <h2><span class="dot"></span>3 · Compare people with themselves</h2>
  {img('rules', 'Personal CUSUM vs a personal rolling drop vs a population threshold, same scorer and budget.')}
  {table(rtab, {c: pct for c in rtab.columns[1:4]} | {"typical mood → flag AUC": f3,
               "caught · cheerful half": pct, "caught · gloomy half": pct}, num_cols=rtab.columns[1:])}
  <p>A population threshold ("recent average below X") catches {pct(pop['detection_rate'])} of declines
  against {pct(cus['detection_rate'])} for the personal CUSUM, and
  {pct(pop['stable_flags_from_lowest_mood_quartile'])} of its false flags land on the gloomiest quarter of
  people who never declined (personal CUSUM: {pct(cus['stable_flags_from_lowest_mood_quartile'])}). Among
  never-declining people the population rule flags, {pct(pop['stable_flags_after_bad_week'])} were within
  a week of a bad week; for the CUSUM, {pct(cus['stable_flags_after_bad_week'])} (base rate of such days:
  {pct(cus['bad_week_base_rate'])}).</p>
</section>

<section>
  <h2><span class="dot"></span>4 · The people who most need noticing write the least</h2>
  {img('silence', 'Four writing behaviours, identical moods (seed held fixed).')}
  <p>Once a decline is complete, people write on {pct(si['written_share_after_full_decline'])} of days
  vs {pct(si['written_share_stable_monitoring'])} for people who never declined. On the days they do write
  their mood averages {si['mean_dev_written_days_after_full_decline']:+.2f} below normal; across all days it
  is {si['mean_dev_all_days_after_full_decline']:+.2f} — silence hides the worst days.</p>
  {table(swtab, {"days written once declined": pct, "declines caught": pct, "median days to flag": f0,
                 "never-declining flagged": pct, "entry count → decline AUC": f3}, num_cols=swtab.columns[1:])}
  {table(adh, {"declines caught": pct, "median days to flag": f0}, num_cols=adh.columns[1:])}
  {table(dep, {"declines caught": pct, "median days to flag": f0}, num_cols=dep.columns[1:])}
</section>

<section>
  <h2><span class="dot"></span>5 · Register shift: same people, Reddit sentences</h2>
  <p>The evaluation people are rewritten with Reddit comments carrying the <em>same</em> emotion in the same
  sentence slot. Thresholds stay as fitted on tweets. For the shipped scorer, the share of never-declining
  people flagged moves by <strong>{100 * shift_c:+.0f} points</strong> under the personal CUSUM and by
  <strong>{100 * shift_p:+.0f} points</strong> under the population threshold.</p>
  {table(ttab, {c: pct for c in ttab.columns[2:6]} | {c: f3 for c in ttab.columns[6:]}, num_cols=ttab.columns[2:])}
</section>

<section>
  <h2><span class="dot"></span>6 · How much must mood leak into words?</h2>
  {img('coupling', 'Sweeping the generator’s mood→word coupling, moods held fixed.')}
  {table(cstab, {"r · gold labels": f2, "r · classifier": f2, "caught · gold labels": pct,
                 "caught · classifier": pct}, num_cols=cstab.columns)}
  <p>The shipped setting (coupling 1.0) gives a within-person correlation of
  {f2(h['within_person_r_gold'])} between an entry's gold-label score and true mood. That is an assumption,
  and a generous one; it is why this table exists.</p>
</section>

<section>
  <h2><span class="dot"></span>7 · The operating point</h2>
  {img('budget', 'Declines caught against never-declining people flagged, across budgets.')}
  {table(bs, {"budget": pct, "threshold": f2, "declines caught": pct, "median days to flag": f0,
              "never-declining flagged": pct, bs.columns[-1]: pct}, num_cols=bs.columns)}
  <p><strong>Does the budget travel?</strong> Calibrating on one cohort and evaluating on the other flags
  {pct1(trv['cal16_to_eval17']['stable_flag_rate'])} of never-declining people; swapping the roles gives
  {pct1(trv['cal17_to_eval16']['stable_flag_rate'])}, both against a {pct(cfg['budget'])} target. Rewriting the same
  people from disjoint halves of the sentence pool moves detection between {pct(pr['detection_min'])} and
  {pct(pr['detection_max'])}. At an assumed {pct(cfg['real_prevalence'])} prevalence,
  {pct(h['ppv_at_real_prevalence'])} of flags would be real; the in-study {pct(cfg['drift_frac'])} would
  flatter that figure.</p>
  <p><strong>Negative controls.</strong> A random scorer catches {pct1(nc['random_scorer']['detection_rate'])};
  shuffling each person's monitored days (same entries, no sustained shape) catches
  {pct1(nc['shuffled_days']['detection_rate'])}.</p>
</section>

<section>
  <h2><span class="dot"></span>How to run</h2>
<pre>git clone https://github.com/Caspal26/journaling-mood-burnout-tracker
cd journaling-mood-burnout-tracker
python -m venv .venv &amp;&amp; .venv\\Scripts\\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python build.py --figures        # downloads corpora, fine-tunes on CPU (~25 min first time)
streamlit run app.py --server.port 8516
pytest -q</pre>
  <p class="muted">Or double-click <code>run.bat</code>. The study ran in {m['runtime_seconds']:.0f} s
  (fine-tune reused) on {m['generated']}.</p>
</section>

<section>
  <h2><span class="dot"></span>Honest limits</h2>
  <ul>
    <li>Tweets and Reddit comments are not diaries, and the lives are simulated. The mood→word coupling,
    writing behaviour and decline shapes are assumptions; the ones that matter are swept.</li>
    <li>"Burnout" here is a sustained drop in expressed mood. Burnout proper is a syndrome — exhaustion,
    cynicism, reduced efficacy — that no sentiment score measures.</li>
    <li>dair-ai labels are hashtag-derived and keyword-heavy; in-domain accuracy overstates what any of these
    scorers would do on real journal prose.</li>
    <li>Sentences are reused across synthetic people, so person-level bootstrap intervals understate
    classifier-error uncertainty; the pool-replicate spread is reported for that reason.</li>
    <li>Nothing here has been tested with real users. A flag should only ever open a supportive, consented
    conversation; it is not a clinical signal.</li>
  </ul>
</section>
</main>
<footer><div class="wrap">Project #16 of the AI-in-Healthcare daily build series · generated from
<code>artifacts/metrics.json</code> · not a medical device.</div></footer>
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"docs -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    return OUT


if __name__ == "__main__":
    build()
