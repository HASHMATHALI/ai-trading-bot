import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evidence


def frame(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes, "ATR": 1.0}, index=idx)


def test_perfect_foresight_beats_hold_and_is_significant():
    closes = np.array([100, 110, 99, 108.9, 98.01, 107.8, 97.0, 106.7] * 8, dtype=float)
    f = frame(closes)
    went_up = pd.Series(np.append(np.diff(closes) > 0, False), index=f.index)
    bt = evidence.run_backtest(f, went_up.astype(float), confidence=0.5, max_atr_pct=100)
    assert bt.hit_rate == 1.0 and bt.strategy_return > bt.hold_return
    assert bt.max_drawdown == 0.0 and bt.p_value < 0.001


def test_always_long_equals_buy_and_hold():
    f = frame(np.linspace(100, 150, 60))
    bt = evidence.run_backtest(f, pd.Series(1.0, index=f.index), confidence=0.5, max_atr_pct=100)
    assert bt.strategy_return == pytest.approx(bt.hold_return)
    assert bt.exposure == 1.0 and bt.p_value == 1.0  # base rate is 100%, nothing to prove


def test_volatility_cap_keeps_strategy_flat():
    f = frame(np.linspace(100, 150, 60))
    bt = evidence.run_backtest(f, pd.Series(1.0, index=f.index), confidence=0.5, max_atr_pct=0.1)
    assert bt.buy_calls == 0 and bt.strategy_return == 0.0


def test_calibration_buckets():
    f = frame(np.array([1, 2, 3, 2, 1, 2, 3, 4], dtype=float))
    p = pd.Series([0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.5], index=f.index)
    cal = evidence.calibration(f, p, bins=2)
    assert cal.loc[cal.predicted < 0.5, "actual"].iloc[0] == 0.0
    assert cal.loc[cal.predicted > 0.5, "actual"].iloc[0] == 1.0


def test_drift_uses_tree_thresholds():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 10, size=(400, 7)), columns=evidence.FEATURES)
    y = (X["RSI"] > 5).astype(int)
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    ranges = evidence.known_ranges(model).set_index("feature")
    assert 0 <= ranges.loc["RSI", "low"] <= ranges.loc["RSI", "high"] <= 10
    inside = pd.Series({f: 5.0 for f in evidence.FEATURES})
    assert not evidence.drift_report(model, inside).set_index("feature").loc["RSI", "outside"]
    far = inside.copy(); far["RSI"] = 500.0
    assert evidence.drift_report(model, far).set_index("feature").loc["RSI", "outside"]


def test_verdict_levels():
    drift_ok = pd.DataFrame({"feature": ["RSI"], "outside": [False]})
    drift_bad = pd.DataFrame({"feature": ["EMA_9"], "outside": [True]})
    good = evidence.Backtest(pd.DataFrame(), 0.5, 0.2, -0.1, -0.2, 0.5, 200, 0.62, 0.53, 0.004)
    poor = evidence.Backtest(pd.DataFrame(), 0.05, 0.6, -0.1, -0.2, 0.3, 200, 0.48, 0.53, 0.93)
    assert evidence.verdict(good, drift_ok)[0] == "EARNED"
    assert evidence.verdict(poor, drift_bad)[0] == "NOT EARNED"
