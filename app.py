import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import yfinance as yf

st.set_page_config(page_title="Tesla AI Stock Forecaster", page_icon="📈", layout="wide")

@st.cache_data(ttl=60)
def get_tesla_data():
    data = yf.download("TSLA", start="2014-01-01", end="2026-01-01", progress=False)
    if data.empty:
        raise ValueError("No data fetched")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    close_prices = data["Close"].dropna().values.astype(float)
    if len(close_prices) < 100:
        raise ValueError(f"Insufficient data: {len(close_prices)}")
    return close_prices, data
def prepare_data(prices, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices = scaler.fit_transform(prices.reshape(-1, 1))
    X, y = [], []
    for i in range(window_size, len(prices)):
        X.append(prices[i-window_size:i, 0])
        y.append(prices[i, 0])
    return np.array(X), np.array(y), scaler

def build_randomforest_model():
    return RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)

def build_gradientboost_model():
    return GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42)
# CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
* { font-family: 'Rajdhani', sans-serif; }
@keyframes gradientAnim { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes neonGlow { 0%,100%{text-shadow:0 0 10px #00ffff,0 0 20px #00ffff,0 0 30px #00ffff,0 0 40px #00ffff} 50%{text-shadow:0 0 5px #ff00ff,0 0 10px #ff00ff,0 0 15px #ff00ff,0 0 20px #ff00ff} }
@keyframes cardFloat { 0%,100%{transform:translateY(0px)} 50%{transform:translateY(-5px)} }
body { background: linear-gradient(45deg, #0a0a1a, #0d1b2a, #1a0a2e, #0a0a1a); background-size: 400% 400%; animation: gradientAnim 15s ease infinite; }
.stApp { background-color: transparent !important; }
.main-title { font-family: 'Orbitron', sans-serif; font-size: 3.5rem; font-weight: 900; text-align: center; background: linear-gradient(90deg, #00ffff, #ff00ff, #ffff00, #00ffff); background-size: 300% 300%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: gradientAnim 5s ease infinite, neonGlow 3s ease infinite; margin-bottom: 1rem; }
.subtitle { text-align: center; font-size: 1.3rem; color: #aaaacc; margin-bottom: 2rem; letter-spacing: 2px; }
.glow-card { background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1)); border: 1px solid rgba(0,255,255,0.3); border-radius: 20px; padding: 20px; transition: all 0.3s ease; animation: cardFloat 3s ease infinite; }
.glow-card:hover { border-color: #00ffff; box-shadow: 0 0 30px rgba(0,255,255,0.5), 0 0 60px rgba(0,255,255,0.3); transform: translateY(-8px) scale(1.02); }
.stat-number { font-family: 'Orbitron', sans-serif; font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #00ffff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
.stat-label { text-align: center; color: #aaaacc; font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase; }
.sidebar-card { background: rgba(255,255,255,0.03); border-radius: 15px; padding: 15px; margin-bottom: 20px; border-left: 3px solid #00ffff; }
.sidebar-title { color: #00ffff; font-weight: 700; font-size: 1.1rem; margin-bottom: 10px; letter-spacing: 1px; }
.stButton>button { background: linear-gradient(90deg, #00ffff, #ff00ff); border: none; border-radius: 10px; padding: 10px 30px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; transition: all 0.3s ease; }
.stButton>button:hover { box-shadow: 0 0 20px rgba(0,255,255,0.5); transform: scale(1.05); }
.css-1l02zno { background: rgba(0,0,0,0.2); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(10,10,30,0.95), rgba(20,10,40,0.95)); border-right: 1px solid rgba(0,255,255,0.2); }
</style>
""", unsafe_allow_html=True)
# SIDEBAR
with st.sidebar:
    st.markdown('''

#### ⚙️ CONFIG

''', unsafe_allow_html=True)
    st.markdown("Last 60 days of TSLA prices", unsafe_allow_html=True)
    st.divider()
    st.markdown('''

#### 📊 MODEL SELECTOR

''', unsafe_allow_html=True)
    model_choice = st.selectbox("Select Model", ["Random Forest", "Gradient Boost"], label_visibility="collapsed")
    st.divider()
    st.markdown('''

#### 🔮 PREDICT HORIZON

''', unsafe_allow_html=True)
    forecast_days = st.slider("Days to Forecast", min_value=1, max_value=10, value=5, label_visibility="collapsed")
    st.divider()
    st.markdown('''

#### ⚡ TRAINING

''', unsafe_allow_html=True)
    st.markdown("Training is fast with CPU", unsafe_allow_html=True)
# MAIN TITLE
st.markdown('''

# ⚡ TESLA AI STOCK FORECASTER

''', unsafe_allow_html=True)
st.markdown('''

Powered by AI | Random Forest & Gradient Boosting

''', unsafe_allow_html=True)

# STAT CARDS
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('''

### 📊

**Training Data**

''', unsafe_allow_html=True)
    st.markdown("### 2,400+ Days", unsafe_allow_html=True)
    st.markdown("", unsafe_allow_html=True)
    st.markdown("\n\n", unsafe_allow_html=True)
with col2:
    st.markdown('''

### 🔮

**Forecast Horizons**

''', unsafe_allow_html=True)
    st.markdown("### 1 / 5 / 10 Days", unsafe_allow_html=True)
    st.markdown("", unsafe_allow_html=True)
    st.markdown("\n\n", unsafe_allow_html=True)
with col3:
    st.markdown('''

### 🤖

**AI Models**

''', unsafe_allow_html=True)
    st.markdown("### Random Forest + Gradient Boost", unsafe_allow_html=True)
    st.markdown("", unsafe_allow_html=True)
    st.markdown("\n\n", unsafe_allow_html=True)
# DATA LOADING
with st.spinner("⏳ Fetching Tesla stock data from Yahoo Finance..."):
    try:
        close_prices, data = get_tesla_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if "predictions" not in st.session_state:
    st.session_state.predictions = None
    st.session_state.model = None
    st.session_state.window_size = 60
    st.session_state.scaler = None

if st.session_state.predictions is None:
    with st.spinner("🔥 Preparing data and building models..."):
        window_size = st.session_state.window_size
        X, y, scaler = prepare_data(close_prices, window_size)
        st.session_state.scaler = scaler
        split = int(len(X) * 0.9)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        with st.spinner("🚀 Ready! Train the model to start predicting..."):
            st.divider()
            if st.button("🚀 TRAIN & PREDICT NOW"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                with st.spinner(f"🔥 Training {model_choice} model..."):
                    if model_choice == "Random Forest":
                        model = build_randomforest_model()
                    else:
                        model = build_gradientboost_model()
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    progress_bar.progress(50)
                    status_text.text("🎯 Generating predictions...")
                    # Predictions
                    last_window = close_prices[-window_size:].reshape(1, window_size)
                    predictions = []
                    current_window = scaler.transform(last_window.reshape(window_size, 1)).flatten()
                    for _ in range(forecast_days):
                        X_pred = current_window.reshape(1, -1)
                        pred = model.predict(X_pred)[0]
                        predictions.append(pred)
                        current_window = np.roll(current_window, -1)
                        current_window[-1] = pred
                    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                    st.session_state.predictions = predictions
                    progress_bar.progress(100)
                    status_text.text("✅ Prediction Complete!")
                    st.balloons()
# DISPLAY PREDICTIONS
if st.session_state.predictions is not None:
    st.divider()
    st.markdown("## 🎯 FORECAST RESULTS")
    # Prediction Table
    pred_df = pd.DataFrame({
        "Day": list(range(1, forecast_days + 1)),
        "Predicted Price (USD)": st.session_state.predictions.round(2)
    })
    st.table(pred_df)
    # Price Trend Analysis
    last_price = close_prices[-1]
    pred_change = ((st.session_state.predictions[0] - last_price) / last_price) * 100
    trend = "🟢 BULLISH" if pred_change > 0 else "🔴 BEARISH"
    st.markdown(f"""

### {trend} OUTLOOK

**Current Price:** ${last_price:.2f} | **Predicted:** ${st.session_state.predictions[0]:.2f} | **Change:** {pred_change:+.2f}%

""")
    # Chart 1: Historical Price
    st.markdown("## 📈 HISTORICAL TSLA PRICE")
    dates = pd.date_range(end=pd.Timestamp.now().date(), periods=len(close_prices))
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dates, y=close_prices, mode="lines", name="Close Price", line=dict(color="#00ffff", width=2, shape="spline")))
    fig1.update_layout(title="Tesla (TSLA) Historical Close Price", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark", height=500)
    st.plotly_chart(fig1, use_container_width=True)
    # Chart 2: Prediction Visualization
    st.markdown("## 🔮 AI PREDICTION VISUALIZATION")
    pred_dates = pd.date_range(start=pd.Timestamp.now().date() + pd.Timedelta(days=1), periods=forecast_days)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates[-100:], y=close_prices[-100:], mode="lines", name="Historical", line=dict(color="#00ffff", width=2, shape="spline")))
    fig2.add_trace(go.Scatter(x=pred_dates, y=st.session_state.predictions, mode="lines+markers", name="Predicted", line=dict(color="#ff00ff", width=3, dash="dot"), marker=dict(size=10, color="#ff00ff")))
    fig2.update_layout(title=f"TSLA Price Forecast - Next {forecast_days} Days", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark", height=500, showlegend=True)
    fig2.add_shape(type="line", x0=pred_dates[0], x1=pred_dates[0], y0=min(close_prices[-100:]), y1=max(st.session_state.predictions)*1.1, line=dict(color="#ffff00", width=2, dash="dash"))
    st.plotly_chart(fig2, use_container_width=True)
    # Reset button
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("🔄 Reset Predictions"):
            st.session_state.predictions = None
            st.session_state.model = None
            st.rerun()
# DISCLAIMER & FOOTER
st.divider()
st.markdown("""

#### ⚠️ IMPORTANT DISCLAIMER

This application is for **educational purposes only** and should NOT be used for actual trading or investment decisions. Stock market predictions using AI models carry significant uncertainty. Always consult a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

Built with ❤️ using Streamlit + Scikit-Learn AI | Tesla AI Stock Forecaster 2026  
⚡ Powered by Random Forest & Gradient Boosting Models

""")
