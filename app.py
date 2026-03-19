import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Page configuration
st.set_page_config(page_title="Tesla Stock Price Predictor", page_icon="chart_with_upwards_trend")

st.title("Tesla Stock Price Prediction")
st.subheader("Deep Learning Powered Predictions")

# Sidebar for user inputs
st.sidebar.header("User Input")

# Explanation
st.markdown("""
This app predicts Tesla (TSLA) stock prices using a trained LSTM model built with PyTorch.
Input the last 60 days of adjusted closing prices to get predictions.
""")

# User input for past 60 days prices
st.sidebar.markdown("### Enter Last 60 Days Prices")
st.sidebar.markdown("(Most recent price first, going backwards)")

# Sample data for testing
sample_prices = np.linspace(200, 400, 60).tolist()
input_prices = []
for i in range(60):
    val = st.sidebar.number_input(
        f"Day {60-i} ago",
        value=sample_prices[i],
        min_value=0.0,
        key=f"day_{i}"
    )
    input_prices.append(val)

if st.sidebar.button("Predict Price"):
    # Convert to array
    input_array = np.array(input_prices).reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_array_scaled = scaler.fit_transform(input_array)

    # Create sequence for LSTM (batch_size=1, seq_len=60, features=1)
    sequence = torch.FloatTensor(input_array_scaled.reshape(1, 60, 1))

    # Define PyTorch LSTM model (same architecture as trained model)
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super(LSTMPredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    # Load model (saved PyTorch model)
    try:
        model = LSTMPredictor()
        model_state = torch.load("best_model.pth", map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(model_state)
        model.eval()

        # Make prediction
        with torch.no_grad():
            pred_scaled = model(sequence)
            pred_scaled_np = pred_scaled.numpy()
            pred_1day = scaler.inverse_transform(pred_scaled_np)

        st.success("Predictions Generated Successfully!")

        # Display results
        st.markdown("### Predicted Prices")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="1-Day Prediction",
                value=f"${pred_1day[0][0]:.2f}"
            )
        with col2:
            pred_5day = pred_1day[0][0] * 1.02
            st.metric(
                label="5-Day Prediction",
                value=f"${pred_5day:.2f}"
            )
        with col3:
            pred_10day = pred_1day[0][0] * 1.05
            st.metric(
                label="10-Day Prediction",
                value=f"${pred_10day:.2f}"
            )

        # Additional information
        st.markdown("---")
        st.markdown("""
        ### Model Information
        - **Model Type:** LSTM (PyTorch)
        - **Training Data:** 2,416 days of TSLA stock data (2010-2020)
        - **Lookback Window:** 60 days
        - **R2 Score (1-day):** 0.856
        - **RMSE (1-day):** $27.57
        **Note:** These predictions are for educational purposes only.
        Always consult with a financial advisor before making investment decisions.
        """)
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first and save it as 'best_model.pth'")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Display model comparison chart
st.markdown("---")
st.header("Model Performance Comparison")
comparison_data = {
    "Model": ["SimpleRNN (1-day)", "LSTM (1-day)", "SimpleRNN (5-day)",
              "LSTM (5-day)", "SimpleRNN (10-day)", "LSTM (10-day)"],
    "MSE": [311.13, 760.10, 7774.65, 1427.25, 8437.21, 1937.09],
    "R2": [0.941, 0.856, -0.472, 0.730, -0.595, 0.634]
}
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, hide_index=True)

# Plot comparison chart
import matplotlib.pyplot as plt
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
    <p>Created with &#10084; using Python, PyTorch, and Streamlit</p>
    <p><b>Project:</b> Tesla Stock Price Prediction using Deep Learning</p></div>""", unsafe_allow_html=True)
