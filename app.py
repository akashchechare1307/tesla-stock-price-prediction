import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

st.markdown('<style>body{background-color:#000;color:#fff}.stApp{background-color:#000!important;color:#fff!important}*{color:#fff!important}</style>',unsafe_allow_html=True)

st.set_page_config(page_title="Tesla AI Stock Forecaster", page_icon="📈", layout="wide")

@st.cache_data(ttl=3600)
def get_tesla_data():
    data = yf.download("TSLA", start="2014-01-01", end="2026-01-01", progress=False)
    if data.empty:
        raise ValueError("No data fetched")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def prepare_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_simplernn(X_train, y_train, X_test, y_test):
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
    predictions = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    return predictions, rmse, mae

def train_lstm(X_train, y_train, X_test, y_test):
    model = Sequential([
        LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
    predictions = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    return predictions, rmse, mae

try:
    data = get_tesla_data()
    close_prices = data['Close'].values.astype(float)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))
    
    st.title("🚀 Tesla AI Stock Forecaster - Deep Learning Edition")
    st.subheader("SimpleRNN & LSTM Models for Stock Price Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${close_prices[-1]:.2f}")
    with col2:
        st.metric("Highest Price", f"${close_prices.max():.2f}")
    with col3:
        st.metric("Lowest Price", f"${close_prices.min():.2f}")
    
    st.write("---")
    
    prediction_days = st.selectbox("Select Prediction Window", [1, 5, 10])
    
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    X_train, y_train = prepare_sequences(train_data, prediction_days)
    X_test, y_test = prepare_sequences(test_data, prediction_days)
    
    st.write(f"### Training {prediction_days}-Day Prediction Models...")
    progress_bar = st.progress(0)
    
    progress_bar.progress(30)
    simplernn_pred, simplernn_rmse, simplernn_mae = train_simplernn(X_train, y_train, X_test, y_test)
    
    progress_bar.progress(70)
    lstm_pred, lstm_rmse, lstm_mae = train_lstm(X_train, y_train, X_test, y_test)
    
    progress_bar.progress(100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SimpleRNN Performance")
        st.metric("RMSE", f"{simplernn_rmse:.4f}")
        st.metric("MAE", f"{simplernn_mae:.4f}")
    
    with col2:
        st.subheader("LSTM Performance")
        st.metric("RMSE", f"{lstm_rmse:.4f}")
        st.metric("MAE", f"{lstm_mae:.4f}")
    
    st.write("---")
    
    y_test_unscaled = scaler.inverse_transform(y_test)
    simplernn_unscaled = scaler.inverse_transform(simplernn_pred)
    lstm_unscaled = scaler.inverse_transform(lstm_pred)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test_unscaled.flatten(), mode='lines', name='Actual Price', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(y=simplernn_unscaled.flatten(), mode='lines', name='SimpleRNN Prediction', line=dict(color='#ffff00', width=2)))
    fig.add_trace(go.Scatter(y=lstm_unscaled.flatten(), mode='lines', name='LSTM Prediction', line=dict(color='#ff00ff', width=2)))
    fig.update_layout(title=f'{prediction_days}-Day Stock Price Predictions', xaxis_title='Time Period', yaxis_title='Price ($)', height=500, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("---")
    st.subheader("Model Comparison")
    
    comparison_data = {
        'Model': ['SimpleRNN', 'LSTM'],
        'RMSE': [simplernn_rmse, lstm_rmse],
        'MAE': [simplernn_mae, lstm_mae]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.success(f"✅ {prediction_days}-Day prediction models trained and evaluated successfully!")
    
    st.write("\n### Key Insights:")
    if lstm_rmse < simplernn_rmse:
        st.info(f"🏆 LSTM model performs better with RMSE: {lstm_rmse:.4f} vs SimpleRNN: {simplernn_rmse:.4f}")
    else:
        st.info(f"🏆 SimpleRNN model performs better with RMSE: {simplernn_rmse:.4f} vs LSTM: {lstm_rmse:.4f}")
    
    st.write("\n### Project Details:")
    with st.expander("📊 About the Models"):
        st.write("""
        **SimpleRNN (Simple Recurrent Neural Network):**
        - Basic RNN architecture for sequence modeling
        - Processes sequential data by maintaining hidden state
        - Prone to vanishing gradient problem over long sequences
        - Faster training but less effective for long-term dependencies
        
        **LSTM (Long Short-Term Memory):**
        - Advanced RNN with memory cells and gates
        - Better at capturing long-term dependencies
        - Uses forget gate, input gate, and output gate mechanisms
        - More parameters but better performance on complex sequences
        """)
    
    with st.expander("🔧 Data Processing"):
        st.write(f"""
        - **Dataset**: Tesla (TSLA) stock data from 2014 to 2026
        - **Train/Test Split**: 80/20
        - **Scaling**: MinMaxScaler (0-1 normalization)
        - **Sequence Length**: {prediction_days} days
        - **Training Samples**: {len(X_train)}
        - **Testing Samples**: {len(X_test)}
        """)
    
    with st.expander("📈 Model Architecture"):
        st.write("""
        **SimpleRNN:**
        - SimpleRNN Layer (50 units, ReLU activation)
        - Dropout (20%)
        - Dense Layer (25 units, ReLU activation)
        - Output Layer (1 unit)
        
        **LSTM:**
        - LSTM Layer (100 units, ReLU activation, return_sequences=True)
        - Dropout (20%)
        - LSTM Layer (50 units, ReLU activation)
        - Dropout (20%)
        - Dense Layer (25 units, ReLU activation)
        - Output Layer (1 unit)
        """)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Please ensure all dependencies are installed.")
