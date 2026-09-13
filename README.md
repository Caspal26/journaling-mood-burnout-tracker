# 📓 Journaling Mood & Burnout Tracker

**AI-in-Healthcare daily build · Project #16 of 30**

> **Not a medical device.** A research demonstration of a *mood-trend screening aid*. It does not
> diagnose burnout, depression or any condition and must not be used to make decisions about a real
> person. All journalers are **synthetic**; the sentences are public social-media text, not diaries.
> No patient, therapy or crisis-line data is used anywhere.

A journaling companion scores the emotion in every sentence with a fine-tuned transformer and asks,
per person: **has their mood dropped below their own normal and stayed there?** A flag means a gentle,
consented check-in. Because that flag is the deliverable, everything is scored **per journaler over
five months** — declines caught, days to flag, people flagged who never declined — and the study
measures how much sentence-level accuracy actually buys.

## Headline (evaluation cohort: 300 journalers, 125 with a sustained decline)

| | |
|---|---|
| Sustained declines caught | **22.4%** (95% CI 15–30%) |
| Never-declining people flagged over 145 days | **8.0%** (budget 10%, fitted on a separate cohort) |
| Median days from onset to flag | **40** (IQR 28–55) |
| Flags that are real at an assumed 10% prevalence | **24%** (in-study 51% is flattered by 40% prevalence) |
| Sentence accuracy, 6 emotions (fine-tuned MiniLM, tweets) | **92.8%**, macro-F1 0.883 |

**A 93%-accurate emotion classifier makes a tracker that catches about one decline in five.** That is
the result, not a failure to report around.

## What the study found

**1 · Most detection is lost before the classifier sees a word.** Same people, same 10% budget:

| daily signal | declines caught | never-declining flagged |
|---|---:|---:|
| true mood, every day (no text, no gaps) | 77.6% | 8.6% |
| true mood, only on days they wrote | 45.6% | 7.4% |
| gold emotion labels of the sentences written | 24.8% | 9.1% |
| **fine-tuned MiniLM (shipped)** | **22.4%** | 8.0% |
| TF-IDF + LR | 20.0% | 9.1% |
| frozen MiniLM embeddings + LR | 18.4% | 4.6% |
| keyword lexicon (no learning) | 12.0% | 10.9% |
| random scorer (negative control) | 4.8% | 13.7% |

Missing days cost 32 points, reading mood through sentence emotions (even with *perfect* labels) 21,
and the classifier 2.4. Across the four scorers sentence accuracy spans 40 points; detection spans
10, and the transformer-vs-TF-IDF gap (+5.6 accuracy, +2.4 detection) sits inside the ±7-point
bootstrap interval.

**2 · The people who most need noticing write the least.** Once a decline is complete people write on
33% of days vs 54% for people who never declined, and the days they skip are their worst (mood −0.85
below normal on written days vs −1.10 over all days). Holding every mood and every sentence's emotion
fixed and changing *only* writing behaviour: writing independent of mood → **37.6%** caught; writing
*more* when low → 38.4%; withdrawing strongly → 13.6%. Silence that follows mood costs ~15 points —
six times what the classifier costs. People who write most days: 48% caught; rarely: 11%.

**3 · Compare people with themselves.** A population threshold ("recent average below X") catches a
similar 23.2%, but **94%** of its false flags land on the gloomiest quarter of people who never
declined (typical mood predicts its false flags with AUC 0.92); for the personal CUSUM it is 14%
(AUC 0.39). The personal rule has its own blind spot, partly built into the generator's logistic
mood→word link: it catches 32% of declines among the more cheerful half and 13% among the gloomier.

**4 · A personal baseline is an estimate too.** Among never-declining people, how much happier their
hidden true mood was during the five baseline weeks than afterwards predicts a false flag with
**AUC 0.79**; having fewer baseline entries does not (0.44, median 20 entries). The first draft of
this diagnostic compared baseline text with later text, scored AUC 0.97, and was **circular** — the
CUSUM is a running sum of exactly that gap — so it was replaced with quantities the tracker never sees.

**5 · Register shift: same people, Reddit sentences.** The fine-tuned model drops from 92.8% to
**50.7%** sentence accuracy on GoEmotions Reddit comments — level with the hand-written lexicon
(51.9%), though it still ranks valence better (AUC 0.805 vs 0.655). The word *feel* is in 66% of the
tweets and 1.9% of the Reddit comments. Rewriting the evaluation people in Reddit sentences:

| shipped scorer | caught · tweets → Reddit | never-declining flagged · tweets → Reddit |
|---|---|---|
| personal CUSUM | 22.4% → 10.4% | 8.0% → 6.9% |
| population threshold | 23.2% → 0.8% | 9.1% → 0.6% |

Why: the same emotion moves the score half as far on Reddit (positive-minus-negative separation
1.93 → 0.92), so the tweet-calibrated absolute cut-point is almost never reached. The personal
baseline re-standardises and protects the false-flag rate — but not sensitivity.

**6 · How much must mood leak into words?** Sweeping the generator's mood→word coupling (moods held
fixed) moves the within-person correlation of a gold-labelled entry with true mood from 0.44 to 0.77,
and gold-label detection only from 21% to 25% — against 45.6% for true mood on the same written days.
The shipped coupling (r = 0.64) is an assumption, and a generous one.

**7 · The operating point.** Keyword probe: masking the 100 training words most tied to the label
(37% of test sentences touched) drops accuracy by 24.5 points for the transformer and 20.7 for TF-IDF.
The 10% budget lands at 8.0% on evaluation and at 13.4% with the cohorts' roles swapped. Rewriting
the same people from disjoint halves of the sentence pool moves detection between 22.4% and 27.2%.
Negative controls: random scorer 4.8%; shuffling each person's monitored days **16.0%** — the shuffle
keeps a declining person's lower later average and destroys only the timing, so a good part of what
the tracker catches is "later entries are lower", not the moment of onset.

## Data — real sentences, synthetic lives
- **dair-ai/emotion** (Saravia et al., 2018): tweets, six emotions, hashtag-derived labels,
  16,000 / 2,000 / 2,000. Trains the scorers. Its validation and test sentences — minus 5 and 11 that
  duplicate training text — are the pools the diaries are written from. Research/educational use.
- **GoEmotions** (Demszky et al., 2020, Apache-2.0): 5,803 single-label Reddit comments mapped onto
  the same six classes via the paper's Ekman grouping. Register-shift test only.
- **Synthetic journalers**, 300 per cohort × 180 days: a personal typical mood, AR(1) day-to-day wobble
  (φ = 0.4), ~1.5 bad weeks per 6 months, and for 40% a sustained decline starting on day 60–130 and
  ramping over 2–6 weeks. A bad week is **exactly as deep** as a decline; only duration separates them.
  ~4 sentences per entry, written on ~51% of days. Writing and word choice depend on mood alone — a test
  requires byte-identical diaries when the label is swapped.

Corpora download on first run into `data/raw/` and are not committed.

## Approach
Four sentence scorers trained on tweets only: keyword lexicon, TF-IDF + LR, frozen all-MiniLM-L6-v2
embeddings + LR, and MiniLM **fine-tuned end-to-end on CPU** (2 epochs, ~17 min; shipped, declared
before any detection result). Daily score = mean sentence valence, P(joy)+P(love) − P(sadness)−P(anger)−P(fear).
Personal baseline from the first 35 days, then a one-sided CUSUM of the standardised drop (k = 0.25,
chosen on the calibration cohort). Threshold = the quantile of **per-person** maxima that flags 10% of
never-declining people in a separate calibration cohort, applied unchanged to evaluation. A flag counts
only between onset and six weeks after the decline completes.

## Build notes worth keeping
- The first generator (daily mood inertia φ = 0.7, bad weeks deeper than declines) let even perfect,
  every-day mood catch only 54% of declines. φ was moved to 0.4, inside the range daily-diary studies
  report, **before** any classifier was scored.
- On this 7.5 GB machine a fine-tune and a 5-worker cross-validation killed each other with
  out-of-memory errors, and a single dense logistic regression thrashed for 13 minutes with 14 BLAS
  threads. Heavy steps now run one at a time with threads capped.

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python build.py --figures        # downloads corpora; fine-tunes on CPU the first time (~20 min)
streamlit run app.py --server.port 8516
pytest -q
```
Or double-click `run.bat`. The app has ten tabs, including a live operating-point slider that re-runs the
study's own functions and a "try it on your own entries" demo that stores nothing.

`docs/index.html` is generated from `artifacts/metrics.json` with figures inlined, so it cannot drift from
the numbers above.

## Honest limits
- Tweets and Reddit comments are not diaries, and the lives are simulated. The mood→word coupling,
  writing behaviour and decline shapes are assumptions; the ones that matter are swept.
- "Burnout" here is a sustained drop in expressed mood. Burnout proper is a syndrome — exhaustion,
  cynicism, reduced efficacy — that no sentiment score measures.
- dair-ai labels are hashtag-derived and keyword-heavy; in-domain accuracy overstates what these
  scorers would do on real journal prose (the Reddit result is the better guide).
- Sentences are reused across synthetic people, so person-level bootstrap intervals understate
  classifier-error uncertainty; the pool-replicate spread is reported for that reason.
- Nothing here has been tested with real users. A flag should only ever open a supportive, consented
  conversation.
