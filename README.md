# 📓 Journaling Mood & Burnout Tracker

**AI-in-Healthcare daily build · Project #16 of 30**

> **Not a medical device.** A research demonstration of a *mood-trend screening aid*. It does not
> diagnose burnout, depression or any condition and must not be used to make decisions about a real
> person. All journalers are **synthetic**; the sentences are public social-media text, not diaries.
> No patient, therapy or crisis-line data is used anywhere.

A journaling companion scores the emotion in every sentence and asks, per person: **has their mood
dropped below their own normal and stayed there?** The deliverable is a gentle check-in flag, so the
evaluation is per journaler over months — declines caught, days to flag, people flagged who never
declined — and it measures how much sentence-level accuracy actually buys.

*Status: code complete; full study (with the fine-tuned transformer) in progress. Results will be
added here from `artifacts/metrics.json`.*

## Data — real sentences, synthetic lives
- **dair-ai/emotion** (tweets, 6 emotions, hashtag-derived labels) trains the sentence scorers; its
  validation/test sentences (minus any duplicated in training) are the pools diaries are written from.
- **GoEmotions** (Reddit, Apache-2.0), mapped to the same 6 classes, is a register-shift test only.
- Synthetic journalers: personal typical mood, day-to-day wobble, bad weeks, and for 40% a sustained
  decline. Writing behaviour and word choice depend on mood only — never on the label (tested).

Corpora are downloaded on first run into `data/raw/` and are not committed.

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python build.py --figures
streamlit run app.py --server.port 8516
pytest -q
```
Or double-click `run.bat`.
