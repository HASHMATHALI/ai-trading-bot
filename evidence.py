"""Evidence engine: everything that decides whether the model deserves trust.

Pure functions over pandas/numpy so they can be unit-tested without Streamlit
or network access.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

FEATURES = ["RSI", "MACD", "MACD_Signal", "EMA_9", "EMA_21", "ATR", "Return"]
TRADING_DAYS = 252


@dataclass
class Backtest:
    curve: pd.DataFrame          # equity of strategy and buy-and-hold, both starting at 1.0
    strategy_return: float
    hold_return: float
    max_drawdown: float
    hold_max_drawdown: float
    exposure: float              # share of days the strategy was in the market
    buy_calls: int
    hit_rate: float              # share of BUY days followed by an up day
    base_rate: float             # share of all days followed by an up day
    p_value: float               # chance of a hit rate this good if BUY days were picked at random


def atr_percent(features: pd.DataFrame) -> pd.Series:
    """ATR as a percentage of price, comparable across tickers and decades."""
    return features["ATR"] / features["Close"] * 100


def model_probabilities(model, features: pd.DataFrame) -> pd.Series:
    """P(next bar closes up) for every row."""
    cols = list(getattr(model, "feature_names_in_", FEATURES))
    up_index = list(model.classes_).index(1)
    return pd.Series(model.predict_proba(features[cols])[:, up_index], index=features.index, name="p_up")


def positions(p_up: pd.Series, atr_pct: pd.Series, confidence: float, max_atr_pct: float) -> pd.Series:
    """1 when the rules would be long for the next bar, else 0 (flat; no shorting).

    Mirrors risk_management.evaluate_risk: a BUY needs P(up) >= confidence and
    volatility under the cap. Everything else (SELL or HOLD) is out of the market.
    """
    return ((p_up >= confidence) & (atr_pct <= max_atr_pct)).astype(int)


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def run_backtest(features: pd.DataFrame, p_up: pd.Series, confidence: float, max_atr_pct: float) -> Backtest:
    """Decide at each close, hold through the next bar. No costs, no leverage."""
    next_ret = features["Close"].pct_change().shift(-1)
    pos = positions(p_up, atr_percent(features), confidence, max_atr_pct)
    frame = pd.DataFrame({"pos": pos, "next_ret": next_ret}).dropna()

    strategy = (1 + frame["pos"] * frame["next_ret"]).cumprod()
    hold = (1 + frame["next_ret"]).cumprod()
    curve = pd.DataFrame({"Model + risk rules": strategy, "Buy and hold": hold})

    up = frame["next_ret"] > 0
    buys = frame["pos"] == 1
    n_buys, hits = int(buys.sum()), int((buys & up).sum())
    base = float(up.mean()) if len(frame) else 0.0
    p_value = float(binomtest(hits, n_buys, base, alternative="greater").pvalue) if n_buys and 0 < base < 1 else 1.0

    return Backtest(
        curve=curve,
        strategy_return=float(strategy.iloc[-1] - 1) if len(strategy) else 0.0,
        hold_return=float(hold.iloc[-1] - 1) if len(hold) else 0.0,
        max_drawdown=_max_drawdown(strategy),
        hold_max_drawdown=_max_drawdown(hold),
        exposure=float(buys.mean()) if len(frame) else 0.0,
        buy_calls=n_buys,
        hit_rate=hits / n_buys if n_buys else 0.0,
        base_rate=base,
        p_value=p_value,
    )


def calibration(features: pd.DataFrame, p_up: pd.Series, bins: int = 5) -> pd.DataFrame:
    """When the model said X% up, how often did the next bar actually close up?"""
    went_up = (features["Close"].pct_change().shift(-1) > 0)
    frame = pd.DataFrame({"p": p_up, "up": went_up}).iloc[:-1]
    if frame.empty:
        return pd.DataFrame(columns=["predicted", "actual", "days"])
    edges = np.linspace(0, 1, bins + 1)
    frame["bin"] = pd.cut(frame["p"], edges, include_lowest=True)
    out = frame.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("up", "mean"), days=("up", "size"))
    return out.reset_index(drop=True)


def known_ranges(model) -> pd.DataFrame:
    """The span of values each feature was split on while the forest was trained.

    A decision tree learns nothing about values beyond its outermost split
    thresholds: everything past them lands in the same leaf. So these thresholds
    are a record, stored in the model itself, of the range it can tell apart.
    """
    cols = list(getattr(model, "feature_names_in_", FEATURES))
    lows, highs = {c: np.inf for c in cols}, {c: -np.inf for c in cols}
    for est in model.estimators_:
        tree = est.tree_
        for f, t in zip(tree.feature, tree.threshold):
            if f >= 0:
                name = cols[f]
                lows[name], highs[name] = min(lows[name], t), max(highs[name], t)
    return pd.DataFrame({"feature": cols, "low": [lows[c] for c in cols], "high": [highs[c] for c in cols]})


def drift_report(model, latest: pd.Series) -> pd.DataFrame:
    """Compare today's inputs with the range the model can distinguish."""
    report = known_ranges(model)
    report["today"] = [float(latest[f]) for f in report["feature"]]
    report["outside"] = (report["today"] < report["low"]) | (report["today"] > report["high"])
    return report


def verdict(bt: Backtest, drift: pd.DataFrame) -> tuple[str, list[str]]:
    """Plain-language trust level with the reasons behind it."""
    reasons: list[str] = []
    score = 0
    if bt.buy_calls < 30:
        reasons.append(f"Only {bt.buy_calls} BUY calls in the test window: too few to judge.")
    elif bt.p_value < 0.05:
        score += 1
        reasons.append(f"BUY days were followed by an up day {bt.hit_rate:.0%} of the time against a {bt.base_rate:.0%} base rate (p = {bt.p_value:.3f}).")
    else:
        reasons.append(f"BUY-day hit rate of {bt.hit_rate:.0%} is not distinguishable from the {bt.base_rate:.0%} base rate (p = {bt.p_value:.2f}).")

    if bt.strategy_return > bt.hold_return:
        score += 1
        reasons.append(f"Returned {bt.strategy_return:+.0%} against {bt.hold_return:+.0%} for buy and hold.")
    else:
        reasons.append(f"Returned {bt.strategy_return:+.0%}, behind buy and hold at {bt.hold_return:+.0%}.")

    outside = drift.loc[drift["outside"], "feature"].tolist()
    if outside:
        score -= 1
        reasons.append("Today's " + ", ".join(outside) + " lie outside the range the model can tell apart, so it is extrapolating.")
    else:
        score += 1
        reasons.append("All of today's inputs are inside the range the model was trained to distinguish.")

    level = "EARNED" if score >= 3 else "UNPROVEN" if score >= 1 else "NOT EARNED"
    return level, reasons
