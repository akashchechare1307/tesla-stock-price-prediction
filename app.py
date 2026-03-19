import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
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

def build_simplernn_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
    model = Sequential()
    model.add(SimpleRNN(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def build_lstm_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


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
    st.markdown('<div class="sidebar-card"><div class="sidebar-title">⚙️ CONFIG</div></div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#aaaacc'>Last 60 days of TSLA prices</small>", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sidebar-card"><div class="sidebar-title">📊 MODEL SELECTOR</div></div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Select Model", ["SimpleRNN", "LSTM"], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="sidebar-card"><div class="sidebar-title">🔮 PREDICT HORIZON</div></div>', unsafe_allow_html=True)
    forecast_days = st.slider("Days to Forecast", min_value=1, max_value=10, value=5, label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="sidebar-card"><div class="sidebar-title">⚡ TRAINING</div></div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#aaaacc'>Training is fast with GPU/CPU</small>", unsafe_allow_html=True)

# MAIN TITLE
st.markdown('<h1 class="main-title">⚡ TESLA AI STOCK FORECASTER</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Deep Learning | SimpleRNN & LSTM Neural Networks</p>', unsafe_allow_html=True)

# STAT CARDS
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="glow-card"><div class="stat-number">📊</div><div class="stat-label">Training Data</div><br>', unsafe_allow_html=True)
    st.markdown("### 2,400+ Days", unsafe_allow_html=True)
    st.markdown("", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="glow-card"><div class="stat-number">🔮</div><div class="stat-label">Forecast Horizons</div><br>', unsafe_allow_html=True)
    st.markdown("### 1 / 5 / 10 Days", unsafe_allow_html=True)
    st.markdown("", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown('<div class="glow-card"><div class="stat-number">🤖</div><div class="stat-label">AI Models</div><br>', unsafe_allow_html=True)
    st.markdown("### SimpleRNN + LSTM", unsafe_allow_html=True)
    st.markdown("", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


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
    st.session_state.history = None
    st.session_state.window_size = 60
    st.session_state.scaler = None

if st.session_state.predictions is None:
    with st.spinner("🔥 Preparing data and building models..."):
        window_size = st.session_state.window_size
        X, y, scaler = prepare_data(close_prices, window_size)
        st.session_state.scaler = scaler
        X = X.reshape((X.shape[0], X.shape[1], 1))
        split = int(len(X) * 0.9)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

with st.spinner("🚀 Ready! Train the model to start predicting..."):
    st.divider()
    if st.button("🚀 TRAIN & PREDICT NOW"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner(f"🔥 Training {model_choice} model..."):
            if model_choice == "SimpleRNN":
                model = build_simplernn_model((window_size, 1))
            else:
                model = build_lstm_model((window_size, 1))
            
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))
            st.session_state.model = model
            st.session_state.history = model.history.history
            progress_bar.progress(50)
            status_text.text("🎯 Generating predictions...")
        
        # Predictions
        last_window = close_prices[-window_size:].reshape(1, window_size, 1)
        last_window_scaled = scaler.transform(last_window.reshape(window_size, 1)).reshape(1, window_size, 1)
        
        predictions = []
        current_window = last_window_scaled.copy()
        for _ in range(forecast_days):
            pred = model.predict(current_window, verbose=0)
            predictions.append(pred[0, 0])
            current_window = np.roll(current_window, -1, axis=1)
            current_window[0, -1, 0] = pred[0, 0]
        
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
    pred_df = pd.DataFrame({"Day": list(range(1, forecast_days + 1)), "Predicted Price (USD)": st.session_state.predictions.round(2)})
    st.table(pred_df.style.apply(lambda x: ['background: linear-gradient(90deg, rgba(0,255,255,0.2), rgba(255,0,255,0.2))'] * len(x), axis=1))
    
    # Price Trend Analysis
    last_price = close_prices[-1]
    pred_change = ((st.session_state.predictions[0] - last_price) / last_price) * 100
    trend = "🟢 BULLISH" if pred_change > 0 else "🔴 BEARISH"
    trend_color = "#00ff00" if pred_change > 0 else "#ff0000"
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1)); border: 1px solid {trend_color}; border-radius: 15px; padding: 20px; margin: 20px 0;'>
        <h3 style='color: {trend_color}; font-family: Orbitron;'>{trend} OUTLOOK</h3>
        <p style='color: #aaaacc; font-size: 1.2rem;'>
            Current Price: <strong style='color: #00ffff'>${last_price:.2f}</strong> | 
            Predicted: <strong style='color: {trend_color}'>${st.session_state.predictions[0]:.2f}</strong> | 
            Change: <strong style='color: {trend_color}'>{pred_change:+.2f}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Chart 3: Training Loss
    if st.session_state.history:
        st.markdown("## 📊 MODEL TRAINING METRICS")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=st.session_state.history['loss'], mode="lines", name="Training Loss", line=dict(color="#00ffff", width=2, shape="spline")))
        fig3.add_trace(go.Scatter(y=st.session_state.history['val_loss'], mode="lines", name="Validation Loss", line=dict(color="#ff00ff", width=2, shape="spline")))
        fig3.update_layout(title=f"{model_choice} Training Progress", xaxis_title="Epoch", yaxis_title="MSE Loss", template="plotly_dark", height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    # Reset button
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("🔄 Reset Predictions"):
            st.session_state.predictions = None
            st.session_state.model = None
            st.session_state.history = None
            st.rerun()


# DISCLAIMER & FOOTER
st.divider()
st.markdown("""
<div style='background: rgba(255,100,100,0.1); border: 1px solid rgba(255,100,100,0.5); border-radius: 15px; padding: 20px; margin: 20px 0;'>
    <h4 style='color: #ff6666; font-family: Orbitron;'>⚠️ IMPORTANT DISCLAIMER</h4>
    <p style='color: #aaaacc; font-size: 0.95rem;'>
        This application is for <strong>educational purposes only</strong> and should NOT be used for actual trading or investment decisions. 
        Stock market predictions using AI models carry significant uncertainty. Always consult a qualified financial advisor before making investment decisions.
        Past performance does not guarantee future results.
    </p>
</div>
<div style='text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid rgba(0,255,255,0.3);'>
    <p style='color: #666; font-size: 0.85rem;'>
        Built with ❤️ using Streamlit + TensorFlow Deep Learning | Tesla AI Stock Forecaster 2026<br>
        <span style='color: #00ffff'>⚡ Powered by SimpleRNN & LSTM Neural Networks</span>
    </p>
</div>
""", unsafe_allow_html=True)
