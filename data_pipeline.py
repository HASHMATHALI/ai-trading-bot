import yfinance as yf
import pandas as pd
import ta
import streamlit as st
import datetime

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_live_data(symbol: str, interval: str = "1d", period: str = "1y") -> pd.DataFrame:
    """
    Fetches real-time market data robustly.
    Cached by Streamlit to prevent API rate limits.
    """
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        if df.empty:
            st.error(f"Failed to fetch data for {symbol}. Try another ticker or check yfinance connection.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the exact same feature engineering as the Colab Training Notebook.
    Includes dropna() to ensure no NaNs are passed to the model.
    """
    try:
        # Prevent mutating original df
        data = df.copy()
        
        # Momentum Indicators
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
        
        # Trend Indicators
        data['EMA_9'] = ta.trend.EMAIndicator(data['Close'], window=9).ema_indicator()
        data['EMA_21'] = ta.trend.EMAIndicator(data['Close'], window=21).ema_indicator()
        
        # Volatility
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
        
        # Price Action
        data['Return'] = data['Close'].pct_change()
        
        # Drop NaN values dynamically created by rolling windows
        data.dropna(inplace=True)
        return data
        
    except Exception as e:
        st.error(f"Error engineering features: {str(e)}")
        return pd.DataFrame()
