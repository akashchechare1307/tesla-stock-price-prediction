st.markdown("""<style>body{background-color:#000;color:#fff}.stApp{background-color:#000!important;color:#fff!important}*{color:#fff!important}</style>""",unsafe_allow_html=True)
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

def build_gradientboosting_model():
    return GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

try:
    close_prices, full_data = get_tesla_data()
except ValueError as e:
    st.error(f"Error loading data: {e}")
    st.stop()

X, y, scaler = prepare_data(close_prices)

rf_model = build_randomforest_model()
gb_model = build_gradientboosting_model()

rf_model.fit(X, y)
gb_model.fit(X, y)

rf_pred = rf_model.predict(X)
gb_pred = gb_model.predict(X)

st.title("Tesla AI Stock Forecaster")
st.write("This app predicts Tesla stock prices using ML models.")

col1, col2 = st.columns(2)

with col1:
    st.metric("RF Model R² Score", f"{rf_model.score(X, y):.4f}")

with col2:
    st.metric("GB Model R² Score", f"{gb_model.score(X, y):.4f}")

fig = go.Figure()
fig.add_trace(go.Scatter(y=y, mode='lines', name='Actual', line=dict(color='#00d4ff')))
fig.add_trace(go.Scatter(y=rf_pred, mode='lines', name='RF Pred', line=dict(color='#ffff00')))
fig.add_trace(go.Scatter(y=gb_pred, mode='lines', name='GB Pred', line=dict(color='#ff00ff')))
fig.update_layout(title='Stock Price Predictions', xaxis_title='Time', yaxis_title='Normalized Price', height=400, template='plotly_dark')
st.plotly_chart(fig, use_container_width=True)

st.write("Model comparison completed.")
