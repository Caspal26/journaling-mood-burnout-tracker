"""Headless run of the whole Streamlit app: every tab must render without an exception."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not ((ROOT / "artifacts" / "metrics.json").exists() and (ROOT / "artifacts" / "study.pkl.gz").exists()),
    reason="study artifacts not built")


def test_app_renders_every_tab_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 10
    assert any("Not a medical device" in w.value for w in at.warning)
    assert len(at.metric) >= 9
    assert len(at.dataframe) >= 8


def test_live_budget_reproduces_the_shipped_headline():
    import json
    from streamlit.testing.v1 import AppTest

    m = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    caught = [x for x in at.metric if x.label == "declines caught"][0]
    assert caught.value == f"{100 * m['headline']['detection_rate']:.1f}%"
