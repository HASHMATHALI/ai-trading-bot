import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as GO
from pathlib import Path

# Local imports
from modules.data_pipeline import fetch_live_data, engineer_features
from modules.risk_management import evaluate_risk, calculate_position_size

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Autonomous Trading AI", page_icon="📈", layout="wide")
st.title("📈 Autonomous Trading AI Dashboard")

# --- INITIALIZE SESSION STATE ---
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 10000.0 # Starting capital

# --- SIDEBAR CONFIG ---
st.sidebar.header("System Configuration")
symbol = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1wk", "1mo"], index=0)
confidence_thresh = st.sidebar.slider("Minimum Confidence", 0.50, 0.99, 0.60)
atr_thresh = st.sidebar.number_input("Max ATR Risk Threshold", value=10.0)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Use pathlib to dynamically find model in parent or current dir
    model_path = Path("models/trading_model.pkl")
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("⚠️ **Model not found!** Please ensure you have run the Colab Training notebook, downloaded `trading_model.pkl`, and placed it in `models/trading_model.pkl` relative to this app.")
    st.info("System running in 'Observation Mode' without ML inference.")

# --- MAIN EXECUTION PIPELINE ---

col1, col2 = st.columns([2, 1])

with st.spinner(f"Fetching market data for {symbol}..."):
    raw_df = fetch_live_data(symbol, interval=timeframe, period="2y")

if not raw_df.empty:
    with st.spinner("Engineering features..."):
        features_df = engineer_features(raw_df)
    
    if not features_df.empty:
        # Extract features for latest candle
        latest_data = features_df.iloc[-1]
        
        # Dashboard Matrix Setup
        col1.subheader(f"Price Action ({symbol})")
        
        # Plotly Candlestick
        fig = GO.Figure(data=[GO.Candlestick(x=features_df.index,
                                             open=features_df['Open'],
                                             high=features_df['High'],
                                             low=features_df['Low'],
                                             close=features_df['Close'],
                                             name='Price')])
        fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=0, r=0, t=30, b=0))
        col1.plotly_chart(fig, use_container_width=True)

        # Dashboard Logic Execution
        col2.subheader("AI System Status")
        
        # Display current metrics
        current_price = latest_data['Close']
        current_atr = latest_data['ATR']
        
        col2.metric("Latest Price", f"${current_price:.2f}", f"{latest_data['Return']*100:.2f}%")
        col2.metric("Current Volatility (ATR)", f"${current_atr:.2f}")

        # INFERENCE
        if model is not None:
            feature_cols = ['RSI', 'MACD', 'MACD_Signal', 'EMA_9', 'EMA_21', 'ATR', 'Return']
            X_latest = pd.DataFrame([latest_data[feature_cols]])
            
            try:
                # Predict
                pred_class = model.predict(X_latest)[0]
                pred_proba = model.predict_proba(X_latest)[0]
                confidence = max(pred_proba) # Get probability of chosen class
                
                # Rule-based Execution
                action = evaluate_risk(
                    signal=pred_class, 
                    confidence=confidence, 
                    current_atr=current_atr,
                    atr_threshold=atr_thresh,
                    confidence_threshold=confidence_thresh
                )
                
                # Styling outcome
                if action == "BUY":
                    st.success(f"🔥 **ACTION: {action}** | Confidence: {confidence*100:.1f}%")
                elif action == "SELL":
                    st.error(f"📉 **ACTION: {action}** | Confidence: {confidence*100:.1f}%")
                else:
                    st.warning(f"🛡️ **ACTION: {action}**")
                    
                # Paper trading log
                if st.button("Manual Trade Execution (Simulated)"):
                    size = calculate_position_size(st.session_state.portfolio_value, 0.02, current_atr)
                    st.session_state.trade_history.append({
                        "Symbol": symbol,
                        "Action": action,
                        "Price": current_price,
                        "Size": round(size, 2),
                        "Confidence": round(confidence, 2)
                    })
                    st.toast("Trade recorded in session state.")

            except Exception as e:
                col2.error(f"Inference error: {str(e)}")

        # Execution History Log
        st.subheader("System Logs & Execution History")
        if st.session_state.trade_history:
            history_df = pd.DataFrame(st.session_state.trade_history)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No trades executed this session.")
        
    else:
        st.warning("Insufficient data length to compute specific technical indicators (e.g. 21 EMA). Try increasing period.")
