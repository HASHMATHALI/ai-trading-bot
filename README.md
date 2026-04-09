# Autonomous Trading API - Hybrid Cloud Deployment

This repository contains a full production-grade autonomous trading system designed to circumvent local GPU constraints by offloading heavy computation training to Google Colab, and utilizing Streamlit Cloud for lightweight CPU-inference execution.

## System Architecture

1. **Google Colab (`colab_training/`)**: Heavy lifting. Fetches massive historical datasets, calculates rolling window technical indicators, trains a robust Scikit-Learn `RandomForestClassifier`, evaluates risk matrix, and exports the serialized model payload (`.pkl`).
2. **Streamlit Cloud (`streamlit_app/`)**: The execution layer. Pulls live `yfinance` market data conditionally, dynamically reconstructs feature engineering pipelines, loads the pickled model into memory, runs rapid CPU inference, applies a strict risk management engine, and surfaces a visual dashboard with simulated paper-trading execution.

---

## 🛠️ Folder Structure

```text
autonomous_trading_system/
├── colab_training/
│   └── Trading_Model_Training.ipynb   # 1. Run this first in Colab
├── streamlit_app/
│   ├── app.py                         # 3. Main Streamlit application
│   ├── requirements.txt               # Streamlit Cloud dependencies
│   ├── models/                        
│   │   └── trading_model.pkl          # 2. Place Colab output here
│   └── modules/
│       ├── data_pipeline.py           # Feature engineering & Data Fetch
│       └── risk_management.py         # AI Override constraints
└── README.md
```

---

## 🚀 Deployment Guide & Setup (Step-by-Step)

### Phase 1: Train Model on Google Colab

1. Open `colab_training/Trading_Model_Training.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run all cells. It will pip install required libraries.
3. Observe the generated Evaluation Matrix and Classification report.
4. The final cell will generate and automatically download `trading_model.pkl` to your local machine.

### Phase 2: Local Verification

Before deploying to the cloud, verify and structure the app locally.

1. Navigate to the `streamlit_app/` directory:
   ```bash
   cd streamlit_app
   ```
2. Create a folder named `models/` inside `streamlit_app/` if it doesn't exist, and place `trading_model.pkl` inside it.
   ```bash
   mkdir models
   mv path/to/downloads/trading_model.pkl models/
   ```
3. Install dependencies locally (optional but recommended):
   ```bash
   pip install -r requirements.txt
   ```
4. Run locally to test for errors:
   ```bash
   streamlit run app.py
   ```

### Phase 3: Deploy to Streamlit Cloud

1. Create a GitHub repository and push the **entire `streamlit_app/` folder** to the main branch. Ensure `requirements.txt` and `app.py` are at the root level of your repo (or appropriately configured). Make sure `models/trading_model.pkl` is also committed (it should be small, usually <10MB).
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your newly created GitHub repository.
5. Set the "Main file path" to `app.py` (if it's in the root) or `streamlit_app/app.py`.
6. Click **Deploy**.

Streamlit Cloud will automatically install packages from `requirements.txt` and spin up your execution dashboard.

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
