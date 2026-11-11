"""Null Hypothesis - a trading desk that cross-examines its own model."""
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import evidence
from data_pipeline import engineer_features, fetch_live_data
from risk_management import calculate_position_size, evaluate_risk

INK, PAPER, RULE, GREEN, RED, MUTED = "#1B1B18", "#F6F1E7", "#CFC6B4", "#1F6B4A", "#9B2226", "#6B665C"

st.set_page_config(page_title="Null Hypothesis", page_icon="∅", layout="wide")
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp {{ background:{PAPER}; }}
html, body, [class*="css"], .stMarkdown p, label {{ font-family:'IBM Plex Sans',sans-serif; color:{INK}; }}
h1,h2,h3 {{ font-family:'Fraunces',serif !important; font-weight:600 !important; color:{INK} !important; letter-spacing:-.01em; }}
.nameplate {{ text-align:center; border-top:3px double {INK}; border-bottom:3px double {INK}; padding:10px 0 12px; margin-bottom:6px; }}
.nameplate .title {{ font-family:'Fraunces',serif; font-weight:900; font-size:3rem; line-height:1; color:{INK}; }}
.nameplate .strap {{ font-size:.82rem; letter-spacing:.18em; text-transform:uppercase; color:{MUTED}; margin-top:6px; }}
.verdict {{ border:1px solid {INK}; padding:18px 22px; background:#FFFDF7; }}
.verdict .stamp {{ display:inline-block; font-family:'Fraunces',serif; font-weight:900; font-size:1.5rem; letter-spacing:.06em;
                  padding:2px 14px; border:3px solid currentColor; transform:rotate(-2deg); }}
.verdict ul {{ margin:12px 0 0 18px; padding:0; }} .verdict li {{ margin-bottom:4px; }}
.kicker {{ font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:{MUTED}; border-bottom:1px solid {RULE}; padding-bottom:4px; margin:18px 0 8px; }}
[data-testid="stMetric"] {{ border-top:1px solid {INK}; padding-top:8px; }}
[data-testid="stMetricValue"] {{ font-family:'Fraunces',serif; }}
.fine {{ color:{MUTED}; font-size:.8rem; }}
</style>
<div class="nameplate">
  <div class="title">Null Hypothesis</div>
  <div class="strap">The model has to beat doing nothing before it gets a say · paper trading only · not financial advice</div>
</div>
""",
    unsafe_allow_html=True,
)


def styled(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=24, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=12), legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor=RULE, linecolor=INK)
    fig.update_yaxes(gridcolor=RULE, linecolor=INK)
    return fig


@st.cache_resource
def load_model():
    try:
        return joblib.load(Path(__file__).resolve().parent / "trading_model.pkl")
    except FileNotFoundError:
        return None
    except Exception as exc:  # e.g. incompatible scikit-learn version
        st.error(f"Model file could not be loaded: {exc}")
        return None


if "trade_history" not in st.session_state:
    st.session_state.trade_history = []
if "portfolio_value" not in st.session_state:
    st.session_state.portfolio_value = 10000.0

# ------------------------------------------------------------------ controls
st.sidebar.header("The case")
symbol = st.sidebar.text_input("Ticker", value="SPY").strip().upper()
if not symbol or not all(c.isalnum() or c in ".-^=" for c in symbol) or len(symbol) > 15:
    st.sidebar.error("Enter a valid ticker symbol, e.g. SPY, AAPL or RELIANCE.NS")
    st.stop()
timeframe = st.sidebar.selectbox("Bar size", ["1d", "1wk", "1mo"], index=0)
window = st.sidebar.selectbox("Evidence window", ["2y", "5y", "10y"], index=1)
st.sidebar.header("The rules")
confidence_thresh = st.sidebar.slider("Minimum confidence to act", 0.50, 0.90, 0.55, 0.01)
max_atr_pct = st.sidebar.slider("Volatility cap (ATR as % of price)", 0.5, 8.0, 2.5, 0.1)
st.sidebar.caption("Change a rule and the whole track record below is recomputed under it.")

model = load_model()
with st.spinner(f"Pulling {symbol}…"):
    raw_df = fetch_live_data(symbol, interval=timeframe, period=window)
if raw_df.empty:
    st.stop()
features = engineer_features(raw_df)
if features.empty:
    st.warning("Not enough history to compute the indicators. Choose a longer evidence window.")
    st.stop()
if model is None:
    st.error("trading_model.pkl was not found, so there is no model to examine.")
    st.stop()

latest = features.iloc[-1]
p_up = evidence.model_probabilities(model, features)
atr_pct = evidence.atr_percent(features)
bt = evidence.run_backtest(features, p_up, confidence_thresh, max_atr_pct)
drift = evidence.drift_report(model, latest)
level, reasons = evidence.verdict(bt, drift)

p_today = float(p_up.iloc[-1])
signal = int(p_today >= 0.5)
action = evaluate_risk(signal, max(p_today, 1 - p_today), float(atr_pct.iloc[-1]), max_atr_pct, confidence_thresh)

# ------------------------------------------------------------------- verdict
colour = {"EARNED": GREEN, "UNPROVEN": "#8A6D1A", "NOT EARNED": RED}[level]
left, right = st.columns([3, 2])
left.markdown(
    f"<div class='verdict'><span class='fine'>TRUST IN THIS MODEL ON {symbol}</span><br>"
    f"<span class='stamp' style='color:{colour}'>{level}</span><ul>" + "".join(f"<li>{r}</li>" for r in reasons) + "</ul></div>",
    unsafe_allow_html=True,
)
with right:
    st.markdown("<div class='kicker'>Today's call, for what it is worth</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    a.metric("Model says", action.split(" (")[0], f"P(up) {p_today:.0%}", delta_color="off")
    b.metric(f"{symbol} last", f"{latest['Close']:,.2f}", f"{latest['Return'] * 100:+.2f}%")
    if "(" in action:
        st.caption("Overridden: " + action[action.index("(") + 1:-1])
    size = calculate_position_size(st.session_state.portfolio_value, 0.02, float(latest["ATR"]), float(latest["Close"]))
    st.caption(f"If acted on: {size:,.1f} shares risks 2% of a ${st.session_state.portfolio_value:,.0f} paper account (stop 1.5 × ATR).")

# ------------------------------------------------------------- track record
st.markdown("<div class='kicker'>Exhibit A · the track record under your rules</div>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Model + rules", f"{bt.strategy_return:+.0%}")
m2.metric("Buy and hold", f"{bt.hold_return:+.0%}")
m3.metric("Worst drawdown", f"{bt.max_drawdown:.0%}", f"hold {bt.hold_max_drawdown:.0%}", delta_color="off")
m4.metric("BUY-day hit rate", f"{bt.hit_rate:.0%}", f"base rate {bt.base_rate:.0%}", delta_color="off")
m5.metric("Time in market", f"{bt.exposure:.0%}", f"{bt.buy_calls} BUY calls", delta_color="off")

fig = go.Figure()
fig.add_scatter(x=bt.curve.index, y=bt.curve["Buy and hold"], name="Buy and hold", line=dict(color=MUTED, width=1.5, dash="dot"))
fig.add_scatter(x=bt.curve.index, y=bt.curve["Model + risk rules"], name="Model + risk rules", line=dict(color=INK, width=2))
fig.update_yaxes(title="growth of 1.00")
st.plotly_chart(styled(fig, 340), width="stretch")
st.markdown(
    "<p class='fine'>Decisions are taken at each close and held for one bar; flat when the rules say SELL or HOLD. No costs or slippage. "
    "The model's training dates are unknown and probably overlap this window, which flatters it - a result that still loses to buy and hold is damning.</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------- calibration and drift
cal_col, drift_col = st.columns(2)
with cal_col:
    st.markdown("<div class='kicker'>Exhibit B · does 60% mean 60%?</div>", unsafe_allow_html=True)
    cal = evidence.calibration(features, p_up)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfectly calibrated", line=dict(color=MUTED, dash="dot"))
    fig.add_scatter(x=cal["predicted"], y=cal["actual"], mode="markers+lines", name="this model",
                    marker=dict(size=(cal["days"] ** 0.5).clip(6, 28), color=INK), line=dict(color=INK, width=1),
                    hovertemplate="said %{x:.0%} · happened %{y:.0%}<extra></extra>")
    fig.update_xaxes(title="model's P(up)", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="share of days that closed up", range=[0, 1], tickformat=".0%")
    st.plotly_chart(styled(fig, 300), width="stretch")
    st.markdown("<p class='fine'>Points on the dotted line mean the probabilities can be taken at face value. Dot size shows how many days fell in the bucket.</p>", unsafe_allow_html=True)

with drift_col:
    st.markdown("<div class='kicker'>Exhibit C · is the model on familiar ground?</div>", unsafe_allow_html=True)
    table = drift.assign(status=drift["outside"].map({True: "outside", False: "inside"}))[["feature", "low", "high", "today", "status"]]
    st.dataframe(table.round(3), width="stretch", hide_index=True)
    st.markdown(
        "<p class='fine'>Low and high are the outermost split thresholds across every tree in the forest. A tree cannot tell apart two values beyond "
        "them, so an input marked outside means the model is treating today like the most extreme day it ever saw. Price-level inputs such as the EMAs "
        "go stale this way as a market trends.</p>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ journal
st.markdown("<div class='kicker'>Paper journal</div>", unsafe_allow_html=True)
actionable = action in ("BUY", "SELL")
if st.button(f"Record simulated {action if actionable else 'trade'}", disabled=not actionable):
    st.session_state.trade_history.append({
        "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "Symbol": symbol, "Action": action,
        "Price": round(float(latest["Close"]), 2), "Shares": round(size, 2), "Notional ($)": round(size * float(latest["Close"]), 2),
        "P(up)": round(p_today, 2), "Trust at entry": level,
    })
    st.toast("Recorded in this session's journal.")
if st.session_state.trade_history:
    journal = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(journal, width="stretch", hide_index=True)
    st.download_button("Download journal (CSV)", journal.to_csv(index=False), "paper_journal.csv", "text/csv")
else:
    st.caption("No simulated trades yet. Each entry stores the trust verdict at the time, so you can later check whether ignoring it paid.")
