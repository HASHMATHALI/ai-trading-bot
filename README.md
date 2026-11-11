# Null Hypothesis — a trading desk that cross-examines its own model

Most ML trading dashboards open with a big green **BUY**. This one opens with a verdict on
whether the model has earned the right to be listened to — **EARNED**, **UNPROVEN** or
**NOT EARNED** — and shows the evidence. The signal comes second.

It is a paper-trading simulation for learning. It places no real orders and is not financial advice.

## The three exhibits

| | Question | How it is answered |
| :-- | :-- | :-- |
| **A. Track record** | Would following the model and your risk rules have beaten doing nothing? | Walk through up to 10 years of bars, decide at each close, hold one bar. Equity curve against buy and hold, drawdowns, time in market, and a binomial test of the BUY-day hit rate against the base rate of up days. |
| **B. Calibration** | When the model says 60%, does it happen 60% of the time? | Reliability curve of predicted probability against realised frequency. |
| **C. Familiar ground** | Is today inside the range the model can tell apart? | The outermost split thresholds of every tree are read straight out of the forest. An input beyond them means the model is extrapolating. |

Change the confidence threshold or the volatility cap in the sidebar and all three are recomputed
under the new rules.

### What it finds about the bundled model

On SPY over five years (run in November 2026) the bundled RandomForest returned about +10%
against +68% for buy and hold, its BUY days were right 52% of the time against a 53% base rate,
and Exhibit C shows why: it was trained when the 9- and 21-day EMAs were between roughly 158 and
446, and they are now above 750. Raw price levels are poor features because they drift out of
range. The verdict is **NOT EARNED**, and the app says so. Your numbers will differ as the market moves.

## Other things it does

- **Volatility cap as a percentage of price**, so one rule works for a $20 stock and a $700 ETF.
  (The old dollar threshold put SPY on permanent HOLD.)
- **Position sizing** that risks 2% of the paper account with a 1.5 × ATR stop and never exceeds
  95% of capital.
- **Paper journal** that stores the trust verdict with every simulated trade and exports to CSV.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

```bash
pip install pytest && pytest -q    # 11 tests: backtest maths, calibration, drift, risk rules
```

`trading_model.pkl` was trained with scikit-learn 1.6.1, which `requirements.txt` pins. If you
retrain, replace the file and update the pin.

## Layout

```text
app.py               Streamlit UI (broadsheet theme in .streamlit/config.toml)
evidence.py          Backtest, calibration, drift detection, verdict — pure functions
risk_management.py   Signal overrides and position sizing
data_pipeline.py     Market data (yfinance) and indicators
tests/               pytest suite
```

## Limits

No transaction costs or slippage. The model's training dates are unknown and probably overlap the
evidence window, which flatters it. Free yfinance data can be delayed.
