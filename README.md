# Autonomous Trading API - Hybrid Cloud Deployment

This repository contains a full production-grade autonomous trading system designed to circumvent local GPU constraints by offloading heavy computation training to Google Colab, and utilizing Streamlit Cloud for lightweight CPU-inference execution.

## System Architecture

1. **Google Colab (`colab_training/`)**: Heavy lifting. Fetches massive historical datasets, calculates rolling window technical indicators, trains a robust Scikit-Learn `RandomForestClassifier`, evaluates risk matrix, and exports the serialized model payload (`.pkl`).
2. **Streamlit Cloud (`streamlit_app/`)**: The execution layer. Pulls live `yfinance` market data conditionally, dynamically reconstructs feature engineering pipelines, loads the pickled model into memory, runs rapid CPU inference, applies a strict risk management engine, and surfaces a visual dashboard with simulated paper-trading execution.

---

## 🛠️ Repository Layout

```text
├── app.py               # Streamlit dashboard (entry point)
├── data_pipeline.py     # Market data fetch + feature engineering
├── risk_management.py   # Rule-based overrides and position sizing
├── trading_model.pkl    # RandomForest trained with scikit-learn 1.6.1
├── requirements.txt
└── tests/               # pytest suite for the risk engine
```

> **Disclaimer:** this is a paper-trading simulation for learning purposes. It places no real
> orders and is not financial advice.

---

## 🚀 Run & Deploy

```bash
pip install -r requirements.txt
streamlit run app.py
```

To deploy on [Streamlit Community Cloud](https://share.streamlit.io/): **New app** → pick this
repository → main file `app.py` → **Deploy**.

If you retrain the model, export it as `trading_model.pkl` in the repository root and update the
`scikit-learn` pin in `requirements.txt` to the version used for training (pickles are not
portable across scikit-learn versions).

```bash
pip install pytest && pytest -q   # run the tests
```

---

## ✅ Testing & Error-Proofing Checklist

To guarantee zero runtime crashes, the following safeguards are implemented:

- [x] **API Resilience**: `yfinance` wrapped in `try/except` with a Streamlit UI visual error fallback if rate-limited.
- [x] **NaN State Protection**: `data.dropna(inplace=True)` ensures indicators generating NaN (due to rolling windows) do not reach model inference.
- [x] **Model Handling**: If `trading_model.pkl` is not found, the dashboard fails gracefully into an "Observation Only" mode rather than experiencing a Fatal Exception.
- [x] **Memory Stability**: `st.cache_data` and `.copy()` ensure garbage collection of large pandas dataframes and prevent memory leaks on continuous state reruns.

---

## ⚠️ Known Limitations

1. **Intraday Data Delay**: Free `yfinance` APIs do not guarantee millisecond precision and can be delayed by 15-20 minutes on 1m/5m intervals depending on the exchange.
2. **Ephemeral Memory Storage**: Simulated execution logs stored via `st.session_state` reset upon closing the browser tab or app sleep. 
3. **Slippage Ignored**: The evaluation matrix does not consider heavy realistic trading spread slippage. Do not connect real capital without forward-testing paper logs first!
