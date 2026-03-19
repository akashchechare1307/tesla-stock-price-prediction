import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Tesla Stock Price Predictor", page_icon="chart_with_upwards_trend", layout="wide")

st.title("Tesla Stock Price Prediction")
st.subheader("Deep Learning Powered Predictions with PyTorch LSTM")

# Sidebar
st.sidebar.header("Settings")

# Fetch recent TSLA data
@st.cache_data

def get_tsla_data():
    ticker = yf.Ticker("TSLA")
    hist = ticker.history(period="2y")
    return hist

def train_and_predict(data, window_size=60, hidden_size=50, epochs=50):
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    # Create sequences
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split - last 20% for validation
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    # LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(2, x.size(0), 50)
            c0 = torch.zeros(2, x.size(0), 50)
            out, _ = self.lstm(x, (h0, c0))
            return self.fc(out[:, -1, :])

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

    # Predict next day
    model.eval()
    with torch.no_grad():
        last_seq = X_val_t[-1:]      
        pred_scaled = model(last_seq).numpy()[0, 0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0, 0]

    # Model metrics on validation set
    with torch.no_grad():
        val_pred = model(X_val_t).numpy().flatten()
    mse = np.mean((val_pred - y_val) ** 2)
    ss_res = np.sum((y_val - val_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return pred_price, mse, r2, scaler, X_val_t, model

# Display TSLA data
with st.expander("View Recent TSLA Data"):
    data = get_tsla_data()
    st.write(f"**Latest TSLA Price:** ${data['Close'].iloc[-1]:.2f}")
    st.dataframe(data.tail(10), use_container_width=True)

st.markdown("---")

# Prediction
if st.button("Train Model & Predict Next Day Price", type="primary"):
    with st.spinner("Training LSTM model with PyTorch..."):
        pred_price, mse, r2, scaler, X_val, model = train_and_predict(get_tsla_data())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="1-Day Prediction", value=f"${pred_price:.2f}")
    with col2:
        data = get_tsla_data()
        current_price = data['Close'].iloc[-1]
        change = pred_price - current_price
        st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"${change:+.2f}")
    with col3:
        st.metric(label="Validation R2", value=f"{r2:.3f}")

    st.success(f"Model trained! MSE: {mse:.2f}, R2: {r2:.3f}")

    # Plot actual vs predicted for validation period
    with torch.no_grad():
        val_pred_scaled = model(X_val).numpy().flatten()

    # Inverse transform
    prices = get_tsla_data()['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(prices)

    # Get the dates for validation period
    split = int(len(prices) * 0.8)
    dates_val = data.index[-(len(data) - split):]
    dates_pred = data.index[-(len(data) - split):]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates_val, prices[-(len(data) - split):], label="Actual Price", color="blue", linewidth=2)
    ax.plot(dates_pred, scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)), label="Predicted Price", color="red", linestyle="--", linewidth=2)
    ax.set_title("Actual vs Predicted Tesla Stock Price (Validation Set)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Model Comparison
st.header("Model Performance Comparison")
comparison_data = {
    "Model": ["SimpleRNN (1-day)", "LSTM (1-day)", "SimpleRNN (5-day)",
              "LSTM (5-day)", "SimpleRNN (10-day)", "LSTM (10-day)"],
    "MSE": [311.13, 760.10, 7774.65, 1427.25, 8437.21, 1937.09],
    "R2": [0.941, 0.856, -0.472, 0.730, -0.595, 0.634]
}
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, hide_index=True, use_container_width=True)


fig, ax = plt.subplots(figsize=(10, 5))
models = comparison_data["Model"]
mse_values = comparison_data["MSE"]
ax.barh(models, mse_values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
ax.set_xlabel("MSE (Lower is Better)")
ax.set_title("Model Performance Comparison - Mean Squared Error")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""<div style="text-align: center; color: gray;">
    <p>Created with ❤️ using Python, PyTorch, yfinance, and Streamlit</p>
    <p><b>Project:</b> Tesla Stock Price Prediction using Deep Learning (SimpleRNN & LSTM)</p>
    <p><i>Disclaimer: For educational purposes only. Not financial advice.</i></p></div>""", unsafe_allow_html=True)
