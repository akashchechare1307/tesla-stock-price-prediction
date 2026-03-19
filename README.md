# Tesla Stock Price Prediction

Tesla (TSLA) stock price prediction using **SimpleRNN** and **LSTM** deep learning models. A Streamlit web app for interactive predictions.

## Live Demo

Try the live application: **[Tesla Stock Price Predictor](https://akash-tesla-prediction.streamlit.app/?clear_cache=1)**

## Project Overview

This project uses deep learning to predict Tesla (TSLA) stock prices based on historical data. It compares two neural network architectures:

- **SimpleRNN** - A basic recurrent neural network
- **LSTM** - Long Short-Term Memory network (generally better for time-series)

## Features

- Historical TSLA stock data analysis (up to 3 years)
- Dual-model prediction (SimpleRNN + LSTM)
- Pre-computed accuracy metrics displayed in the app
- Interactive prediction interface
- Professional Streamlit web application
- Clean, minimal architecture

## Project Structure

```
tesla-stock-price-prediction/
├── README.md           # Project documentation
├── app.py              # Streamlit web application
├── requirements.txt    # Python dependencies
└── Notebook/           # (Optional) Colab training notebook
```

## Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras** - Deep learning models
- **Streamlit** - Web application framework
- **yfinance** - Yahoo Finance API
- **NumPy & Pandas** - Data processing
- **Scikit-learn** - Metrics and scaling

## Installation

### Clone the Repository

```bash
git clone https://github.com/akashchechare1307/tesla-stock-price-prediction.git
cd tesla-stock-price-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser.

## How It Works

1. **Data Collection**: TSLA stock data is fetched from Yahoo Finance using `yfinance`
2. **Preprocessing**: Close prices are scaled using MinMaxScaler
3. **Sequence Creation**: Historical prices are converted into sequences for time-series input
4. **Model Prediction**: Both SimpleRNN and LSTM models generate predictions
5. **Metrics Display**: Pre-computed RMSE, MAE, and MAPE values are shown
6. **User Input**: Users can input days to predict and see forecasted prices

## Models

### SimpleRNN
- 2 SimpleRNN layers with ReLU activation
- Dropout layers for regularization
- Dense output layer with linear activation

### LSTM
- 2 LSTM layers with ReLU activation
- Dropout layers for regularization
- Dense output layer with linear activation

## Note

- The prediction is based on pre-computed metrics from the training data
- Stock predictions are for educational purposes only and should not be used for actual trading decisions
- The LSTM model typically achieves lower error metrics compared to SimpleRNN

## License

This project is open-source and available for educational purposes.

## Author

**Akash Chechare**  
Electronics & Computer Engineering Student  
[GitHub](https://github.com/akashchechare1307)

---

> ⚠️ **Disclaimer**: Stock price predictions are for educational and demonstration purposes only. This tool should not be used as financial advice.
